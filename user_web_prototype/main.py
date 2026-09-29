from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import csv
import hashlib
import io
import json
import os
import shlex
import sqlite3
import subprocess
import threading
import uuid
import zipfile

import numpy as np
import pydicom
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydicom.pixels import apply_modality_lut, apply_voi_lut

ROOT = Path(os.getenv("USER_WEB_DATA", "/data")).resolve()
UPLOADS = ROOT / "uploads"
PREVIEWS = ROOT / "previews"
RUNS = ROOT / "training_runs"
DB_PATH = ROOT / "prototype.db"
ALLOWED = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_FILES = 1000
MAX_FILE_BYTES = 512 * 1024 * 1024
for directory in (ROOT, UPLOADS, PREVIEWS, RUNS):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DXA User Web prototype")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connection():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    with connection() as db:
        db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS sessions (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS assets (
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, name TEXT NOT NULL,
          source_path TEXT NOT NULL, preview_path TEXT NOT NULL,
          width INTEGER NOT NULL, height INTEGER NOT NULL, created_at TEXT NOT NULL,
          FOREIGN KEY(session_id) REFERENCES sessions(id)
        );
        CREATE TABLE IF NOT EXISTS annotations (
          asset_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL,
          FOREIGN KEY(asset_id) REFERENCES assets(id)
        );
        CREATE TABLE IF NOT EXISTS retrain_queue (
          asset_id TEXT PRIMARY KEY, status TEXT NOT NULL, queued_at TEXT NOT NULL,
          FOREIGN KEY(asset_id) REFERENCES assets(id)
        );
        CREATE TABLE IF NOT EXISTS train_jobs (
          id TEXT PRIMARY KEY, status TEXT NOT NULL, manifest_path TEXT NOT NULL,
          output_path TEXT NOT NULL, command TEXT NOT NULL, log TEXT NOT NULL,
          created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
        );
        """)


init_db()


def safe_name(value: str) -> str:
    name = PurePosixPath(str(value).replace("\\", "/")).name
    cleaned = "".join(c for c in name if c.isalnum() or c in "._- ()[]")[:180]
    return cleaned or "image"


def dicom_preview(data: bytes) -> Image.Image:
    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = np.asarray(ds.pixel_array)
    arr = np.asarray(apply_modality_lut(arr, ds), dtype=np.float32)
    try:
        arr = np.asarray(apply_voi_lut(arr, ds), dtype=np.float32)
    except Exception:
        pass
    finite = np.isfinite(arr)
    if arr.ndim != 2 or not finite.any():
        raise ValueError("DICOM does not contain a finite 2D image")
    lo, hi = np.nanpercentile(arr[finite], [1, 99])
    if hi <= lo:
        lo, hi = np.nanmin(arr[finite]), np.nanmax(arr[finite])
    if hi <= lo:
        pixels = np.zeros(arr.shape, dtype=np.uint8)
    else:
        pixels = (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        pixels = 255 - pixels
    return Image.fromarray(pixels, mode="L")


def decode_image(name: str, data: bytes) -> Image.Image:
    suffix = Path(name).suffix.lower()
    image = dicom_preview(data) if suffix in {".dcm", ".dicom"} else Image.open(io.BytesIO(data))
    image.load()
    if image.mode not in {"L", "RGB"}:
        image = image.convert("RGB")
    image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    return image


def ingest(session_id: str, name: str, data: bytes):
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{name}: file is larger than 512 MB")
    name = safe_name(name)
    if Path(name).suffix.lower() not in ALLOWED:
        return None
    asset_id = uuid.uuid4().hex
    source_dir = UPLOADS / session_id
    preview_dir = PREVIEWS / session_id
    source_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    source = source_dir / f"{asset_id}_{name}"
    preview = preview_dir / f"{asset_id}.png"
    source.write_bytes(data)
    try:
        image = decode_image(name, data)
        width, height = image.size
        image.save(preview, "PNG", optimize=True)
    except Exception:
        source.unlink(missing_ok=True)
        raise
    with connection() as db:
        db.execute(
            "INSERT INTO assets VALUES (?,?,?,?,?,?,?,?)",
            (asset_id, session_id, name, str(source), str(preview), width, height, now()),
        )
    return asset_id


def row_dict(row):
    return dict(row) if row else None


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/sessions")
async def create_session(files: list[UploadFile] = File(...)):
    session_id = uuid.uuid4().hex
    with connection() as db:
        db.execute("INSERT INTO sessions VALUES (?,?,?)", (session_id, f"Загрузка {now()}", now()))
    accepted, errors = [], []
    candidates = []
    for upload in files:
        raw = await upload.read(MAX_FILE_BYTES + 1)
        if (upload.filename or "").lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    for info in archive.infolist():
                        if info.is_dir() or Path(info.filename).suffix.lower() not in ALLOWED:
                            continue
                        if info.file_size > MAX_FILE_BYTES:
                            errors.append(f"{info.filename}: слишком большой файл")
                            continue
                        candidates.append((info.filename, archive.read(info)))
            except Exception as exc:
                errors.append(f"{upload.filename}: некорректный ZIP ({exc})")
        else:
            candidates.append((upload.filename or "image", raw))
    if len(candidates) > MAX_FILES:
        raise HTTPException(400, f"За один раз допускается не более {MAX_FILES} изображений")
    for name, raw in candidates:
        try:
            asset_id = ingest(session_id, name, raw)
            if asset_id:
                accepted.append(asset_id)
        except Exception as exc:
            errors.append(f"{safe_name(name)}: {exc}")
    if not accepted:
        raise HTTPException(400, {"message": "Не найдено читаемых изображений", "errors": errors})
    return {"session_id": session_id, "accepted": len(accepted), "errors": errors}


@app.post("/api/sessions/{session_id}/assets")
async def add_session_assets(session_id: str, files: list[UploadFile] = File(...)):
    with connection() as db:
        if not db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone():
            raise HTTPException(404, "Session not found")
    accepted, errors, candidates = [], [], []
    for upload in files:
        raw = await upload.read(MAX_FILE_BYTES + 1)
        if (upload.filename or "").lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    for info in archive.infolist():
                        if info.is_dir() or Path(info.filename).suffix.lower() not in ALLOWED:
                            continue
                        if info.file_size > MAX_FILE_BYTES:
                            errors.append(f"{info.filename}: слишком большой файл")
                            continue
                        candidates.append((info.filename, archive.read(info)))
            except Exception as exc:
                errors.append(f"{upload.filename}: некорректный ZIP ({exc})")
        else:
            candidates.append((upload.filename or "image", raw))
    if len(candidates) > MAX_FILES:
        raise HTTPException(400, f"За один раз допускается не более {MAX_FILES} изображений")
    for name, raw in candidates:
        try:
            asset_id = ingest(session_id, name, raw)
            if asset_id:
                accepted.append(asset_id)
        except Exception as exc:
            errors.append(f"{safe_name(name)}: {exc}")
    if not accepted:
        raise HTTPException(400, {"message": "Не найдено читаемых изображений", "errors": errors})
    return {"session_id": session_id, "accepted": len(accepted), "errors": errors}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    with connection() as db:
        assets = [dict(r) for r in db.execute(
            "SELECT id,name,width,height FROM assets WHERE session_id=? ORDER BY created_at,id", (session_id,)
        )]
        queue_count = db.execute(
            "SELECT count(*) FROM retrain_queue q JOIN assets a ON a.id=q.asset_id WHERE a.session_id=? AND q.status='queued'",
            (session_id,),
        ).fetchone()[0]
    if not assets:
        raise HTTPException(404, "Session not found")
    return {"session_id": session_id, "assets": assets, "queue_count": queue_count}


@app.get("/api/assets/{asset_id}/image")
def asset_image(asset_id: str):
    with connection() as db:
        row = db.execute("SELECT preview_path FROM assets WHERE id=?", (asset_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Image not found")
    return FileResponse(row[0], media_type="image/png")


@app.get("/api/assets/{asset_id}/annotation")
def get_annotation(asset_id: str):
    with connection() as db:
        asset = db.execute("SELECT id,width,height FROM assets WHERE id=?", (asset_id,)).fetchone()
        row = db.execute("SELECT payload,updated_at FROM annotations WHERE asset_id=?", (asset_id,)).fetchone()
        queued = db.execute("SELECT status FROM retrain_queue WHERE asset_id=?", (asset_id,)).fetchone()
    if not asset:
        raise HTTPException(404, "Image not found")
    payload = json.loads(row[0]) if row else empty_annotation(asset[1], asset[2])
    payload = merge_annotation_defaults(payload, asset[1], asset[2])
    return {"payload": payload, "updated_at": row[1] if row else None, "queue_status": queued[0] if queued else None}


def annotation_rows(asset_id: str, name: str, payload: dict) -> list[dict]:
    common = {
        "asset_id": asset_id, "file_name": name, "class": payload.get("class", "UNKNOWN"),
        "side": payload.get("side", ""), "quality_flags": ",".join(payload.get("quality_flags", [])),
        "comment": payload.get("comment", ""),
    }
    rows = []
    geometry = payload.get("geometry") or {}
    spine = geometry.get("spine") or {}
    for item in spine.get("disc_lines") or []:
        points = item.get("points") or [[None, None], [None, None]]
        rows.append({**common, "object_type": "disc_line", "object_id": item.get("id", ""),
                     "x1": points[0][0], "y1": points[0][1], "x2": points[1][0], "y2": points[1][1]})
    for key, point in (spine.get("iliac_crests") or {}).items():
        if point:
            rows.append({**common, "object_type": f"iliac_{key}", "object_id": "",
                         "x1": point[0], "y1": point[1], "x2": "", "y2": ""})
    for item in spine.get("foreign_objects") or []:
        box = item.get("bbox") or ["", "", "", ""]
        rows.append({**common, "object_type": "foreign_object", "object_id": item.get("id", ""),
                     "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
    hip = geometry.get("hip") or {}
    for key, point in (hip.get("landmarks") or {}).items():
        if point:
            rows.append({**common, "object_type": key, "object_id": "",
                         "x1": point[0], "y1": point[1], "x2": "", "y2": ""})
    for key, strokes in (hip.get("lesser_trochanter_traces") or {}).items():
        for stroke in strokes or []:
            for point_index, point in enumerate(stroke.get("points") or []):
                rows.append({**common, "object_type": f"trace_{key}", "object_id": stroke.get("id", ""),
                             "point_index": point_index, "x1": point[0], "y1": point[1], "x2": "", "y2": ""})
    return rows or [{**common, "object_type": "annotation", "object_id": "", "x1": "", "y1": "", "x2": "", "y2": ""}]


@app.get("/api/assets/{asset_id}/annotation/export")
def export_annotation(asset_id: str, format: str = "csv"):
    format = format.lower()
    if format not in {"csv", "xlsx"}:
        raise HTTPException(400, "Поддерживаются только CSV и XLSX")
    with connection() as db:
        asset = db.execute("SELECT name,width,height FROM assets WHERE id=?", (asset_id,)).fetchone()
        annotation = db.execute("SELECT payload FROM annotations WHERE asset_id=?", (asset_id,)).fetchone()
    if not asset:
        raise HTTPException(404, "Image not found")
    payload = merge_annotation_defaults(json.loads(annotation[0]) if annotation else {}, asset[1], asset[2])
    rows = annotation_rows(asset_id, asset[0], payload)
    columns = ["asset_id", "file_name", "class", "side", "object_type", "object_id", "point_index",
               "x1", "y1", "x2", "y2", "quality_flags", "comment"]
    base_name = Path(asset[0]).stem or "annotation"
    if format == "csv":
        stream = io.StringIO(newline="")
        stream.write("\ufeff")
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        body = io.BytesIO(stream.getvalue().encode("utf-8"))
        media_type = "text/csv; charset=utf-8"
    else:
        from openpyxl import Workbook
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Разметка"
        sheet.append(columns)
        for row in rows:
            sheet.append([row.get(column, "") for column in columns])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column_cells in sheet.columns:
            width = min(42, max(10, max(len(str(cell.value or "")) for cell in column_cells) + 2))
            sheet.column_dimensions[column_cells[0].column_letter].width = width
        body = io.BytesIO()
        workbook.save(body)
        body.seek(0)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    download_name = "".join(
        char if char.isascii() and (char.isalnum() or char in "._-") else "_" for char in base_name
    ).strip("_") or "annotation"
    headers = {"Content-Disposition": f'attachment; filename="{download_name}_annotation.{format}"'}
    return StreamingResponse(body, media_type=media_type, headers=headers)


def empty_annotation(width: int, height: int) -> dict:
    return {
        "schema_version": 2,
        "class": "UNKNOWN",
        "side": "",
        "quality_flags": [],
        "comment": "",
        "curves": [],
        "geometry": {
            "spine": {
                "disc_lines": [],
                "iliac_crests": {"image_left": None, "image_right": None},
                "foreign_objects": [],
            },
            "hip": {
                "landmarks": {
                    "greater_trochanter": None,
                    "femoral_neck": None,
                    "ischial_bone": None,
                },
                "lesser_trochanter_traces": {"trochanter": [], "adjacent_bone": []},
            },
            "image_view": {"mode": "original", "threshold_8bit": 128},
        },
        "image_width": width,
        "image_height": height,
    }


def merge_annotation_defaults(payload: dict, width: int, height: int) -> dict:
    """Add v2 geometry to legacy prototype annotations without losing old curves."""
    merged = empty_annotation(width, height)
    if not isinstance(payload, dict):
        return merged
    for key in ("class", "side", "quality_flags", "comment", "curves"):
        if key in payload:
            merged[key] = payload[key]
    geometry = payload.get("geometry")
    if isinstance(geometry, dict):
        for region in ("spine", "hip"):
            if isinstance(geometry.get(region), dict):
                merged["geometry"][region].update(geometry[region])
        if isinstance(geometry.get("image_view"), dict):
            merged["geometry"]["image_view"].update(geometry["image_view"])
    return merged


def validate_annotation(payload: dict, width: int, height: int):
    if not isinstance(payload, dict):
        raise ValueError("Annotation must be an object")
    label = payload.get("class", "UNKNOWN")
    if label not in {"UNKNOWN", "SPINE", "LEG"}:
        raise ValueError("Unknown image class")
    curves = payload.get("curves", [])
    if not isinstance(curves, list) or len(curves) > 128:
        raise ValueError("Invalid curves")
    clean_curves = []
    for curve in curves:
        points = curve.get("points") if isinstance(curve, dict) else None
        if not isinstance(points, list) or len(points) != 4:
            raise ValueError("Each Bezier curve must contain four points")
        clean = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Invalid curve point")
            x, y = float(point[0]), float(point[1])
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError("Curve point is outside the image")
            clean.append([round(x, 2), round(y, 2)])
        clean_curves.append({"id": str(curve.get("id") or uuid.uuid4().hex)[:64], "points": clean})
    flags = payload.get("quality_flags", [])
    allowed_flags = {"cropping", "position", "rotation", "artifact"}
    if not isinstance(flags, list) or not set(flags).issubset(allowed_flags):
        raise ValueError("Invalid quality flags")
    def point(value):
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("Invalid annotation point")
        x, y = float(value[0]), float(value[1])
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError("Annotation point is outside the image")
        return [round(x, 2), round(y, 2)]

    def line(value):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("Annotation line must contain two points")
        return [point(value[0]), point(value[1])]

    def box(value):
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError("Annotation box must contain four coordinates")
        x1, y1 = point(value[:2])
        x2, y2 = point(value[2:])
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    geometry = payload.get("geometry") or {}
    spine = geometry.get("spine") or {}
    hip = geometry.get("hip") or {}
    disc_lines = spine.get("disc_lines") or []
    foreign_objects = spine.get("foreign_objects") or []
    if not isinstance(disc_lines, list) or len(disc_lines) > 32:
        raise ValueError("Invalid intervertebral lines")
    if not isinstance(foreign_objects, list) or len(foreign_objects) > 64:
        raise ValueError("Invalid foreign objects")
    iliac = spine.get("iliac_crests") or {}
    landmarks = hip.get("landmarks") or {}
    traces = hip.get("lesser_trochanter_traces") or {}
    clean_traces = {}
    for trace_name in ("trochanter", "adjacent_bone"):
        strokes = traces.get(trace_name) or []
        if not isinstance(strokes, list) or len(strokes) > 24:
            raise ValueError("Invalid hip traces")
        clean_traces[trace_name] = []
        for stroke in strokes:
            points = stroke.get("points") if isinstance(stroke, dict) else None
            if not isinstance(points, list) or not 2 <= len(points) <= 2048:
                raise ValueError("A hip trace must contain 2–2048 points")
            clean_traces[trace_name].append({
                "id": str(stroke.get("id") or uuid.uuid4().hex)[:64],
                "points": [point(item) for item in points],
            })
    view = geometry.get("image_view") or {}
    mode = view.get("mode", "original")
    if mode not in {"original", "threshold"}:
        mode = "original"
    threshold = int(view.get("threshold_8bit", 128))
    threshold = max(0, min(255, threshold))
    return {
        "schema_version": 2, "class": label,
        "side": payload.get("side", "") if payload.get("side", "") in {"", "LEFT", "RIGHT"} else "",
        "quality_flags": flags, "comment": str(payload.get("comment", ""))[:2000],
        "curves": clean_curves,
        "geometry": {
            "spine": {
                "disc_lines": [
                    {"id": str(item.get("id") or uuid.uuid4().hex)[:64], "points": line(item.get("points"))}
                    for item in disc_lines if isinstance(item, dict)
                ],
                "iliac_crests": {
                    "image_left": point(iliac.get("image_left")),
                    "image_right": point(iliac.get("image_right")),
                },
                "foreign_objects": [
                    {"id": str(item.get("id") or uuid.uuid4().hex)[:64],
                     "kind": str(item.get("kind") or "other")[:32], "bbox": box(item.get("bbox"))}
                    for item in foreign_objects if isinstance(item, dict)
                ],
            },
            "hip": {
                "landmarks": {
                    "greater_trochanter": point(landmarks.get("greater_trochanter")),
                    "femoral_neck": point(landmarks.get("femoral_neck")),
                    "ischial_bone": point(landmarks.get("ischial_bone")),
                },
                "lesser_trochanter_traces": clean_traces,
            },
            "image_view": {"mode": mode, "threshold_8bit": threshold},
        },
        "image_width": width, "image_height": height,
    }


@app.put("/api/assets/{asset_id}/annotation")
def put_annotation(asset_id: str, payload: dict):
    with connection() as db:
        asset = db.execute("SELECT width,height FROM assets WHERE id=?", (asset_id,)).fetchone()
        if not asset:
            raise HTTPException(404, "Image not found")
        try:
            clean = validate_annotation(payload, asset[0], asset[1])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        stamp = now()
        db.execute(
            "INSERT INTO annotations VALUES (?,?,?) ON CONFLICT(asset_id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at",
            (asset_id, json.dumps(clean, ensure_ascii=False), stamp),
        )
    return {"saved": True, "updated_at": stamp, "payload": clean}


@app.post("/api/assets/{asset_id}/queue")
def enqueue(asset_id: str):
    with connection() as db:
        annotated = db.execute("SELECT 1 FROM annotations WHERE asset_id=?", (asset_id,)).fetchone()
        if not annotated:
            raise HTTPException(409, "Сначала сохраните корректировку")
        already_queued = db.execute(
            "SELECT status FROM retrain_queue WHERE asset_id=? AND status='queued'", (asset_id,)
        ).fetchone() is not None
        db.execute(
            "INSERT INTO retrain_queue VALUES (?,?,?) ON CONFLICT(asset_id) DO UPDATE SET status='queued',queued_at=excluded.queued_at",
            (asset_id, "queued", now()),
        )
    return {"queued": True, "already_queued": already_queued}


def execute_job(job_id: str, command_template: str, manifest: Path, output: Path):
    command = command_template.format(manifest=str(manifest), output=str(output))
    with connection() as db:
        db.execute("UPDATE train_jobs SET status='running',command=?,started_at=? WHERE id=?", (command, now(), job_id))
    try:
        process = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=24 * 3600, check=False)
        status = "completed" if process.returncode == 0 else "failed"
        log = (process.stdout + "\n" + process.stderr)[-100_000:]
    except Exception as exc:
        status, log = "failed", str(exc)
    with connection() as db:
        db.execute("UPDATE train_jobs SET status=?,log=?,finished_at=? WHERE id=?", (status, log, now(), job_id))


@app.post("/api/training/start")
def start_training():
    with connection() as db:
        rows = db.execute("""
          SELECT a.id,a.name,a.source_path,n.payload,q.queued_at
          FROM retrain_queue q JOIN assets a ON a.id=q.asset_id
          JOIN annotations n ON n.asset_id=a.id WHERE q.status='queued' ORDER BY q.queued_at
        """).fetchall()
    if not rows:
        raise HTTPException(409, "Очередь дообучения пуста")
    job_id = uuid.uuid4().hex
    job_dir = RUNS / job_id
    job_dir.mkdir(parents=True)
    manifest = job_dir / "corrections.jsonl"
    output = job_dir / "model_output"
    with manifest.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({
                "asset_id": row[0], "name": row[1], "source_path": row[2],
                "annotation": json.loads(row[3]), "queued_at": row[4],
            }, ensure_ascii=False) + "\n")
    command = os.getenv("TRAIN_COMMAND", "").strip()
    status = "queued" if command else "awaiting_runner"
    with connection() as db:
        db.execute("INSERT INTO train_jobs VALUES (?,?,?,?,?,?,?,?,?)", (
            job_id, status, str(manifest), str(output), command, "", now(), None, None
        ))
        db.executemany("UPDATE retrain_queue SET status='assigned' WHERE asset_id=?", [(r[0],) for r in rows])
    if command:
        threading.Thread(target=execute_job, args=(job_id, command, manifest, output), daemon=True).start()
    return {"job_id": job_id, "status": status, "items": len(rows), "runner_configured": bool(command)}


@app.get("/api/training/jobs")
def training_jobs():
    with connection() as db:
        rows = [dict(r) for r in db.execute(
            "SELECT id,status,manifest_path,output_path,created_at,started_at,finished_at,log FROM train_jobs ORDER BY created_at DESC LIMIT 20"
        )]
    return {"jobs": rows}
