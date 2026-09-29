from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import csv
import hashlib
import hmac
import io
import json
import os
import secrets
import sqlite3
import stat
import threading
import time
import uuid
import zipfile

import httpx
import pydicom
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(os.getenv("USER_WEB_DATA", "/data")).resolve()
UPLOADS = ROOT / "uploads"
CORRECTIONS = ROOT / "retraining_queue"
DB_PATH = ROOT / "user_web.db"
DXA_API_URL = os.getenv("DXA_API_URL", "http://127.0.0.1:8765").rstrip("/")
ALLOWED = {".dcm", ".dicom"}
MAX_FILES = 1000
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
BLOCKED_ARCHIVE_SUFFIXES = {".bat", ".cmd", ".com", ".dll", ".exe", ".js", ".msi", ".ps1", ".sh", ".vbs"}
TABLE_COLUMNS = (
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "quality_class",
    "violation_type",
    "processing_status",
    "time_of_processing",
)
SPINE_TARGETS = ("spine_position", "spine_axis", "spine_artifact", "spine_scoliosis")
HIP_TARGETS = ("hip_position", "hip_roi", "hip_rotation")
SPINE_REVIEWED = ["spine", "artifact", "spine_crests", "scoliosis"]
HIP_REVIEWED = ["hip", "hip_mask", "hip_points"]
AUTH_COOKIE = "dxa_web_session"
AUTH_MAX_AGE = 30 * 24 * 60 * 60
PASSWORD_ITERATIONS = 310_000

for directory in (ROOT, UPLOADS, CORRECTIONS):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DXA Studio user web", version="1.0.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
prediction_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connection():
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=30000")
    return db


def ensure_columns(db: sqlite3.Connection, table: str, columns: dict[str, str]):
    existing = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    for name, declaration in columns.items():
        if name not in existing:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def valid_login(value: object) -> str:
    login = str(value or "").strip()
    if not 3 <= len(login) <= 64:
        raise HTTPException(422, "Логин должен содержать от 3 до 64 символов")
    return login


def valid_password(value: object) -> str:
    password = str(value or "")
    if not 8 <= len(password) <= 200:
        raise HTTPException(422, "Пароль должен содержать не менее 8 символов")
    return password


def password_record(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return salt.hex(), digest.hex()


def init_db():
    with connection() as db:
        db.executescript(
            """
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
            CREATE TABLE IF NOT EXISTS web_users (
              id INTEGER PRIMARY KEY CHECK(id=1), login TEXT NOT NULL UNIQUE COLLATE NOCASE,
              password_hash TEXT NOT NULL, password_salt TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS web_auth_sessions (
              token_hash TEXT PRIMARY KEY, kind TEXT NOT NULL, user_id INTEGER,
              created_at TEXT NOT NULL, expires_at INTEGER NOT NULL
            );
            """
        )
        ensure_columns(db, "sessions", {
            "predict_job_id": "TEXT", "status": "TEXT NOT NULL DEFAULT 'legacy'", "error": "TEXT",
        })
        ensure_columns(db, "assets", {
            "service_image_id": "TEXT", "service_path": "TEXT", "pixel_hash": "TEXT",
            "prediction_status": "TEXT NOT NULL DEFAULT 'legacy'", "prediction_error": "TEXT",
            "result_id": "TEXT", "table_json": "TEXT", "prediction_json": "TEXT",
            "geometry_json": "TEXT", "annotation_version": "INTEGER NOT NULL DEFAULT 0",
            "corrected": "INTEGER NOT NULL DEFAULT 0",
        })
        ensure_columns(db, "annotations", {"service_version": "INTEGER NOT NULL DEFAULT 0"})
        ensure_columns(db, "retrain_queue", {
            "training_job_id": "TEXT", "kind": "TEXT NOT NULL DEFAULT 'human'",
        })
        ensure_columns(db, "train_jobs", {
            "service_job_id": "TEXT", "result_json": "TEXT", "error": "TEXT",
        })
        db.execute(
            """
            INSERT OR IGNORE INTO retrain_queue(asset_id,status,queued_at,training_job_id,kind)
            SELECT id,'queued',?,NULL,'model' FROM assets
            WHERE corrected=0 AND result_id IS NOT NULL
              AND prediction_json IS NOT NULL AND geometry_json IS NOT NULL
            """,
            (now(),),
        )
        default_login = os.getenv("DXA_DEFAULT_LOGIN", "").strip()
        default_password = os.getenv("DXA_DEFAULT_PASSWORD", "")
        if default_login and default_password and not db.execute("SELECT 1 FROM web_users WHERE id=1").fetchone():
            login = valid_login(default_login)
            password = valid_password(default_password)
            salt, digest = password_record(password)
            db.execute(
                "INSERT INTO web_users(id,login,password_hash,password_salt,updated_at) VALUES (1,?,?,?,?)",
                (login, digest, salt, now()),
            )


init_db()


def account_exists() -> bool:
    with connection() as db:
        return db.execute("SELECT 1 FROM web_users WHERE id=1").fetchone() is not None


def auth_context(request: Request) -> dict | None:
    token = request.cookies.get(AUTH_COOKIE)
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with connection() as db:
        row = db.execute("SELECT * FROM web_auth_sessions WHERE token_hash=?", (token_hash,)).fetchone()
        if not row:
            return None
        if int(row["expires_at"]) <= int(time.time()):
            db.execute("DELETE FROM web_auth_sessions WHERE token_hash=?", (token_hash,))
            return None
        login = None
        if row["kind"] == "user":
            user = db.execute("SELECT login FROM web_users WHERE id=?", (row["user_id"],)).fetchone()
            if not user:
                db.execute("DELETE FROM web_auth_sessions WHERE token_hash=?", (token_hash,))
                return None
            login = user["login"]
    return {"kind": row["kind"], "user_id": row["user_id"], "login": login, "token_hash": token_hash}


def auth_payload(context: dict | None = None) -> dict:
    mode = context["kind"] if context else None
    return {
        "account_exists": account_exists(), "mode": mode,
        "authenticated": mode == "user", "guest": mode == "guest",
        "login": context.get("login") if context else None,
    }


def session_response(kind: str, user_id: int | None = None) -> JSONResponse:
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expires = int(time.time()) + AUTH_MAX_AGE
    with connection() as db:
        db.execute("DELETE FROM web_auth_sessions WHERE expires_at<=?", (int(time.time()),))
        db.execute(
            "INSERT INTO web_auth_sessions(token_hash,kind,user_id,created_at,expires_at) VALUES (?,?,?,?,?)",
            (token_hash, kind, user_id, now(), expires),
        )
        login = db.execute("SELECT login FROM web_users WHERE id=?", (user_id,)).fetchone()[0] if user_id else None
    context = {"kind": kind, "user_id": user_id, "login": login}
    response = JSONResponse(auth_payload(context))
    response.set_cookie(AUTH_COOKIE, token, max_age=AUTH_MAX_AGE, httponly=True, samesite="lax", path="/")
    return response


def safe_name(value: str) -> str:
    name = PurePosixPath(str(value).replace("\\", "/")).name
    cleaned = "".join(c for c in name if c.isalnum() or c in "._- ()[]")[:180]
    return cleaned or "image.dcm"


def unsafe_archive_member(info: zipfile.ZipInfo) -> str | None:
    normalized = str(info.filename).replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
        return "архив содержит небезопасный путь"
    if stat.S_ISLNK(info.external_attr >> 16):
        return "символические ссылки в архиве запрещены"
    if path.suffix.lower() in BLOCKED_ARCHIVE_SUFFIXES:
        return "исполняемые файлы в архиве запрещены"
    return None


def api_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        detail = body.get("detail", body) if isinstance(body, dict) else body
        return detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
    except Exception:
        return response.text or f"HTTP {response.status_code}"


def api_request(method: str, path: str, *, expected=(200,), **kwargs):
    try:
        with httpx.Client(base_url=DXA_API_URL, timeout=httpx.Timeout(120, connect=10)) as client:
            response = client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        raise HTTPException(503, f"API модели недоступен: {exc}") from exc
    if response.status_code not in expected:
        status = 409 if response.status_code in (409, 422) else response.status_code
        raise HTTPException(status, api_detail(response))
    return response


def api_json(method: str, path: str, *, expected=(200,), **kwargs):
    return api_request(method, path, expected=expected, **kwargs).json()


def get_asset(asset_id: str):
    with connection() as db:
        row = db.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Снимок не найден")
    return dict(row)


def empty_geometry(width: int, height: int) -> dict:
    return {
        "schema_version": 1,
        "coordinate_system": "original_dicom_pixels_top_left",
        "image_width": int(width), "image_height": int(height),
        "spine": {"disc_lines": [], "iliac_crests": {"image_left": None, "image_right": None}, "foreign_objects": []},
        "hip": {
            "landmarks": {"greater_trochanter": None, "femoral_neck": None, "ischial_bone": None},
            "lesser_trochanter": None,
            "lesser_trochanter_traces": {"trochanter": [], "adjacent_bone": []},
            "lesser_trochanter_pixels": [], "lesser_trochanter_mask_ready": False,
            "lesser_trochanter_partial": False, "roi_box": None,
        },
        "image_view": {"mode": "original", "threshold_8bit": 128},
        "complete": {"spine": False, "hip": False},
    }


async def upload_candidates(files: list[UploadFile]):
    candidates: list[tuple[str, bytes]] = []
    errors: list[str] = []
    for upload in files:
        raw = await upload.read(MAX_ARCHIVE_BYTES + 1)
        await upload.close()
        filename = upload.filename or "image.dcm"
        if len(raw) > MAX_ARCHIVE_BYTES:
            errors.append(f"{safe_name(filename)}: файл больше 512 МиБ")
            continue
        if filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    archive_candidates: list[tuple[str, bytes]] = []
                    expanded_bytes = 0
                    for info in archive.infolist():
                        unsafe_reason = unsafe_archive_member(info)
                        if unsafe_reason:
                            errors.append(f"{safe_name(filename)}: {unsafe_reason}")
                            archive_candidates = []
                            break
                        if info.is_dir() or Path(info.filename).suffix.lower() not in ALLOWED:
                            continue
                        if info.file_size > MAX_FILE_BYTES:
                            errors.append(f"{safe_name(info.filename)}: DICOM больше 128 МиБ")
                            continue
                        expanded_bytes += info.file_size
                        if expanded_bytes > MAX_ARCHIVE_BYTES:
                            errors.append(f"{safe_name(filename)}: распакованный архив больше 512 МиБ")
                            archive_candidates = []
                            break
                        archive_candidates.append((info.filename, archive.read(info)))
                    candidates.extend(archive_candidates)
            except (zipfile.BadZipFile, RuntimeError) as exc:
                errors.append(f"{safe_name(filename)}: некорректный ZIP ({exc})")
        elif Path(filename).suffix.lower() in ALLOWED:
            if len(raw) > MAX_FILE_BYTES:
                errors.append(f"{safe_name(filename)}: DICOM больше 128 МиБ")
            else:
                candidates.append((filename, raw))
        else:
            errors.append(f"{safe_name(filename)}: поддерживаются только DICOM и ZIP")
    if len(candidates) > MAX_FILES:
        raise HTTPException(400, f"За один раз допускается не более {MAX_FILES} DICOM")
    return candidates, errors


def create_prediction_session(candidates: list[tuple[str, bytes]], errors: list[str]):
    health = api_json("GET", "/v1/health")
    if health.get("status") != "ready":
        raise HTTPException(503, "API запущен без активной модели")
    session_id = uuid.uuid4().hex
    stamp = now()
    with connection() as db:
        db.execute("INSERT INTO sessions(id,name,created_at,status) VALUES (?,?,?,?)",
                   (session_id, f"Исследование {stamp}", stamp, "uploading"))
    accepted: list[str] = []
    seen_images: set[str] = set()
    session_dir = UPLOADS / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    for original_name, raw in candidates:
        name = safe_name(original_name)
        try:
            service = api_json("POST", "/v1/data/uploads",
                               files={"file": (name, raw, "application/dicom")})
            if service["image_id"] in seen_images:
                continue
            seen_images.add(service["image_id"])
            asset_id = uuid.uuid4().hex
            source = session_dir / f"{asset_id}_{name}"
            source.write_bytes(raw)
            with connection() as db:
                db.execute(
                    """
                    INSERT INTO assets(
                      id,session_id,name,source_path,preview_path,width,height,created_at,
                      service_image_id,service_path,pixel_hash,prediction_status,annotation_version
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (asset_id, session_id, name, str(source), "", int(service["width"]), int(service["height"]),
                     stamp, service["image_id"], service["path"], service.get("pixel_hash"), "queued",
                     int(service.get("annotation_version", 0))),
                )
            accepted.append(service["image_id"])
        except HTTPException as exc:
            errors.append(f"{name}: {exc.detail}")
    if not accepted:
        with connection() as db:
            db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        raise HTTPException(400, {"message": "Не найдено корректных DICOM", "errors": errors})
    prediction = api_json("POST", "/v1/model/predict?wait_seconds=0", expected=(200, 202), json={
        "image_ids": accepted, "paths": [], "mode": "single" if len(accepted) == 1 else "batch",
        "model_version": None, "request_id": f"web-predict-{session_id}",
    })
    job_id = prediction["job_id"]
    with connection() as db:
        db.execute("UPDATE sessions SET predict_job_id=?,status='predicting',error=NULL WHERE id=?",
                   (job_id, session_id))
    return {"session_id": session_id, "accepted": len(accepted), "errors": errors, "job_id": job_id}


def refresh_prediction(session_id: str):
    with prediction_lock, connection() as db:
        session = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not session:
            raise HTTPException(404, "Исследование не найдено")
        if session["status"] not in ("predicting", "queued", "running") or not session["predict_job_id"]:
            return
        job_id = session["predict_job_id"]
    try:
        job = api_json("GET", f"/v1/jobs/{job_id}")
    except HTTPException as exc:
        if exc.status_code == 503:
            return
        raise
    status = job["status"]
    if status in ("queued", "running"):
        with connection() as db:
            db.execute("UPDATE sessions SET status=? WHERE id=?", (status, session_id))
            db.execute("UPDATE assets SET prediction_status=? WHERE session_id=? AND result_id IS NULL",
                       (status, session_id))
        return job
    if status == "failed":
        error = job.get("error") or "Ошибка распознавания"
        with connection() as db:
            db.execute("UPDATE sessions SET status='failed',error=? WHERE id=?", (error, session_id))
            db.execute("UPDATE assets SET prediction_status='failed',prediction_error=? WHERE session_id=?",
                       (error, session_id))
        return job
    if status not in ("completed", "completed_with_errors"):
        return job
    result = job.get("result") or {}
    table_by_path = {row["path_to_study"]: row for row in result.get("table", [])}
    predictions = {}
    for result_id in result.get("result_ids", []):
        item = api_json("GET", f"/v1/results/{result_id}")
        geometry = api_json("GET", f"/v1/results/{result_id}/files/geometry.json")
        predictions[item["source"]] = (result_id, item, geometry)
    with connection() as db:
        assets = db.execute("SELECT * FROM assets WHERE session_id=?", (session_id,)).fetchall()
        for asset in assets:
            path = asset["service_path"]
            row = table_by_path.get(path)
            prediction = predictions.get(path)
            if prediction:
                result_id, item, geometry = prediction
                db.execute(
                    """
                    UPDATE assets SET prediction_status=?,prediction_error=NULL,result_id=?,
                      table_json=?,prediction_json=?,geometry_json=? WHERE id=?
                    """,
                    (row.get("processing_status", "Success") if row else "Success", result_id,
                     json.dumps(row, ensure_ascii=False) if row else None, json.dumps(item, ensure_ascii=False),
                     json.dumps(geometry, ensure_ascii=False), asset["id"]),
                )
                db.execute(
                    """
                    INSERT INTO retrain_queue(asset_id,status,queued_at,training_job_id,kind)
                    VALUES (?,?,?,NULL,'model') ON CONFLICT(asset_id) DO NOTHING
                    """,
                    (asset["id"], "queued", now()),
                )
            else:
                error = next((entry.get("error") for entry in result.get("errors", [])
                              if entry.get("path") == path), "Предсказание недоступно")
                db.execute("UPDATE assets SET prediction_status='Failure',prediction_error=?,table_json=? WHERE id=?",
                           (error, json.dumps(row, ensure_ascii=False) if row else None, asset["id"]))
        db.execute("UPDATE sessions SET status=?,error=NULL WHERE id=?", (status, session_id))
    return job


def asset_payload(asset: dict, saved_payload: dict | None = None):
    if saved_payload is not None:
        return saved_payload
    prediction = json.loads(asset["prediction_json"]) if asset.get("prediction_json") else {}
    geometry = json.loads(asset["geometry_json"]) if asset.get("geometry_json") else empty_geometry(asset["width"], asset["height"])
    region = prediction.get("region", "UNKNOWN")
    targets = prediction.get("quality_flags") or {}
    if region == "SPINE":
        image_class, side = "SPINE", ""
    elif region in ("LEG_LEFT", "LEG_RIGHT"):
        image_class, side = "LEG", region.removeprefix("LEG_")
    else:
        image_class, side = "UNKNOWN", ""
    return {
        "schema_version": 1, "class": image_class, "side": side,
        "quality_flags": [key for key, value in targets.items() if value == 1],
        "targets": targets, "comment": "", "geometry": geometry,
        "image_width": asset["width"], "image_height": asset["height"],
        "service_version": int(asset.get("annotation_version") or 0),
    }


def validated_training_payload(asset: dict, payload: dict):
    image_class = payload.get("class")
    side = payload.get("side", "")
    if image_class == "SPINE":
        region, target_names, reviewed = "SPINE", SPINE_TARGETS, SPINE_REVIEWED
    elif image_class == "LEG" and side in ("LEFT", "RIGHT"):
        region, target_names, reviewed = f"LEG_{side}", HIP_TARGETS, HIP_REVIEWED
    else:
        raise HTTPException(400, "Для бедра укажите сторону; для снимка выберите анатомическую область")
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        raise HTTPException(400, "Геометрическая разметка отсутствует")
    geometry["image_width"] = int(asset["width"])
    geometry["image_height"] = int(asset["height"])
    geometry["schema_version"] = 1
    geometry.setdefault("coordinate_system", "original_dicom_pixels_top_left")
    geometry.setdefault("complete", {"spine": False, "hip": False})
    geometry["complete"]["spine" if region == "SPINE" else "hip"] = True
    incoming = payload.get("targets") or {}
    checked = set(payload.get("quality_flags") or [])
    targets = {name: int(incoming.get(name, 1 if name in checked else 0) or 0) for name in target_names}
    return {
        "schema_version": 1, "class": image_class, "side": side if image_class == "LEG" else "",
        "quality_flags": [key for key, value in targets.items() if value == 1], "targets": targets,
        "comment": str(payload.get("comment", ""))[:2000], "geometry": geometry,
        "image_width": int(asset["width"]), "image_height": int(asset["height"]),
        "reviewed": reviewed, "region": region,
    }


def write_correction_snapshot(asset: dict, payload: dict, version: int):
    folder = CORRECTIONS / asset["id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "geometry.json").write_text(json.dumps(payload["geometry"], ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "annotation.json").write_text(
        json.dumps({**payload, "asset_id": asset["id"], "annotation_version": version, "saved_at": now()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    ds = pydicom.dcmread(asset["source_path"], force=True)
    block = ds.private_block(0x0011, "DXA_MANUAL_LABELER", create=True)
    label = "SPINE" if payload["region"] == "SPINE" else "LEG"
    ds.add_new(block.get_tag(0x01), "CS", label)
    ds.add_new(block.get_tag(0x02), "LT", payload.get("comment", ""))
    ds.add_new(block.get_tag(0x03), "LO", "DXA Studio")
    ds.add_new(block.get_tag(0x04), "DT", datetime.now().strftime("%Y%m%d%H%M%S.%f"))
    ds.add_new(block.get_tag(0x05), "CS", payload.get("side", "") if label == "LEG" else "")
    ds.add_new(block.get_tag(0x09), "UT", json.dumps(payload["geometry"], ensure_ascii=True, separators=(",", ":")))
    ds.add_new(block.get_tag(0x0A), "UT", json.dumps(payload["targets"], ensure_ascii=True, separators=(",", ":")))
    temp = folder / "image.tmp.dcm"
    ds.save_as(temp, enforce_file_format=True)
    temp.replace(folder / "image.dcm")
    return str(folder)


@app.get("/api/auth/status")
def get_auth_status(request: Request):
    return auth_payload(auth_context(request))


@app.post("/api/auth/guest")
def enter_as_guest():
    return session_response("guest")


@app.post("/api/auth/register")
def register_account(payload: dict):
    login = valid_login(payload.get("login"))
    password = valid_password(payload.get("password"))
    salt, digest = password_record(password)
    with connection() as db:
        if db.execute("SELECT 1 FROM web_users WHERE id=1").fetchone():
            raise HTTPException(409, "Локальный аккаунт уже создан")
        db.execute(
            "INSERT INTO web_users(id,login,password_hash,password_salt,updated_at) VALUES (1,?,?,?,?)",
            (login, digest, salt, now()),
        )
    return session_response("user", 1)


@app.post("/api/auth/login")
def login_account(payload: dict):
    login = str(payload.get("login") or "").strip()
    password = str(payload.get("password") or "")
    with connection() as db:
        user = db.execute("SELECT * FROM web_users WHERE id=1 AND login=? COLLATE NOCASE", (login,)).fetchone()
    if not user:
        raise HTTPException(401, "Неверный логин или пароль")
    _, digest = password_record(password, user["password_salt"])
    if not hmac.compare_digest(digest, user["password_hash"]):
        raise HTTPException(401, "Неверный логин или пароль")
    return session_response("user", int(user["id"]))


@app.post("/api/auth/logout")
def logout_account(request: Request):
    context = auth_context(request)
    if context:
        with connection() as db:
            db.execute("DELETE FROM web_auth_sessions WHERE token_hash=?", (context["token_hash"],))
    response = JSONResponse(auth_payload())
    response.delete_cookie(AUTH_COOKIE, path="/")
    return response


@app.put("/api/auth/account")
def update_account(request: Request, payload: dict):
    context = auth_context(request)
    if not context or context["kind"] != "user":
        raise HTTPException(401, "Изменять аккаунт может только авторизованный пользователь")
    with connection() as db:
        user = db.execute("SELECT * FROM web_users WHERE id=?", (context["user_id"],)).fetchone()
    current_password = str(payload.get("current_password") or "")
    _, current_digest = password_record(current_password, user["password_salt"])
    if not hmac.compare_digest(current_digest, user["password_hash"]):
        raise HTTPException(401, "Текущий пароль указан неверно")
    login = valid_login(payload.get("login") or user["login"])
    new_password = str(payload.get("new_password") or "")
    salt, digest = (password_record(valid_password(new_password)) if new_password
                    else (user["password_salt"], user["password_hash"]))
    with connection() as db:
        db.execute(
            "UPDATE web_users SET login=?,password_hash=?,password_salt=?,updated_at=? WHERE id=?",
            (login, digest, salt, now(), context["user_id"]),
        )
        db.execute("DELETE FROM web_auth_sessions WHERE kind='user'")
    return session_response("user", int(context["user_id"]))


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/system/status")
def system_status():
    return api_json("GET", "/v1/health")


@app.post("/api/sessions")
async def create_session(files: list[UploadFile] = File(...)):
    candidates, errors = await upload_candidates(files)
    return create_prediction_session(candidates, errors)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    prediction_job = refresh_prediction(session_id)
    with connection() as db:
        session = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not session:
            raise HTTPException(404, "Исследование не найдено")
        assets = []
        service_names = {}
        for row in db.execute("SELECT * FROM assets WHERE session_id=? ORDER BY created_at,id", (session_id,)):
            item = dict(row)
            item["table"] = json.loads(item["table_json"]) if item.get("table_json") else None
            if item.get("service_path"):
                service_names[item["service_path"]] = item["name"]
            for key in ("source_path", "preview_path", "service_path", "table_json", "prediction_json", "geometry_json"):
                item.pop(key, None)
            assets.append(item)
        queue_count = db.execute(
            """SELECT count(*) FROM retrain_queue q JOIN assets a ON a.id=q.asset_id
               WHERE a.session_id=? AND q.status='queued' AND q.kind='human'""",
            (session_id,),
        ).fetchone()[0]
    progress = None
    if prediction_job and session["status"] in ("predicting", "queued", "running"):
        result = prediction_job.get("result") or {}
        current = result.get("current_image") or {}
        index = int(current.get("image_index") or 0)
        paths = (prediction_job.get("payload") or {}).get("paths") or []
        current_path = paths[index - 1] if 0 < index <= len(paths) else None
        progress = {
            "index": index,
            "processed": int(result.get("processed") or 0),
            "total": int(result.get("total") or len(assets)),
            "current_file": service_names.get(current_path) if current_path else None,
        }
    return {"session_id": session_id, "status": session["status"], "error": session["error"],
            "assets": assets, "queue_count": queue_count, "progress": progress}


@app.get("/api/assets/{asset_id}/image")
def asset_image(asset_id: str):
    asset = get_asset(asset_id)
    endpoint = (f"/v1/results/{asset['result_id']}/files/overlay.png" if asset.get("result_id")
                else f"/v1/data/images/{asset['service_image_id']}/preview.png")
    response = api_request("GET", endpoint)
    return Response(response.content, media_type=response.headers.get("content-type", "image/png"),
                    headers={"Cache-Control": "no-store"})


@app.get("/api/assets/{asset_id}/preview")
def asset_preview(asset_id: str):
    asset = get_asset(asset_id)
    response = api_request("GET", f"/v1/data/images/{asset['service_image_id']}/preview.png")
    return Response(response.content, media_type=response.headers.get("content-type", "image/png"),
                    headers={"Cache-Control": "no-store"})


@app.get("/api/assets/{asset_id}/annotation")
def get_annotation(asset_id: str):
    asset = get_asset(asset_id)
    with connection() as db:
        row = db.execute("SELECT payload,updated_at,service_version FROM annotations WHERE asset_id=?", (asset_id,)).fetchone()
        queued = db.execute(
            "SELECT status FROM retrain_queue WHERE asset_id=? AND kind='human'", (asset_id,)
        ).fetchone()
    payload = asset_payload(asset, json.loads(row["payload"]) if row else None)
    return {"payload": payload, "updated_at": row["updated_at"] if row else None,
            "queue_status": queued["status"] if queued else None,
            "prediction_status": asset["prediction_status"]}


@app.put("/api/assets/{asset_id}/annotation")
def put_annotation(asset_id: str, payload: dict):
    asset = get_asset(asset_id)
    clean = validated_training_payload(asset, payload)
    service = api_json("PUT", f"/v1/data/images/{asset['service_image_id']}/annotation", json={
        "region": clean["region"], "geometry": clean["geometry"], "reviewed": clean["reviewed"],
        "targets": clean["targets"], "expected_version": int(asset.get("annotation_version") or 0),
        "mask_result_id": None,
    })
    version = int(service["version"])
    clean["service_version"] = version
    stamp = now()
    correction_path = write_correction_snapshot(asset, clean, version)
    with connection() as db:
        db.execute(
            """
            INSERT INTO annotations(asset_id,payload,updated_at,service_version) VALUES (?,?,?,?)
            ON CONFLICT(asset_id) DO UPDATE SET payload=excluded.payload,
              updated_at=excluded.updated_at,service_version=excluded.service_version
            """,
            (asset_id, json.dumps(clean, ensure_ascii=False), stamp, version),
        )
        db.execute("UPDATE assets SET annotation_version=?,corrected=1 WHERE id=?", (version, asset_id))
    return {"saved": True, "updated_at": stamp, "payload": clean, "correction_path": correction_path}


@app.post("/api/assets/{asset_id}/queue")
def enqueue(asset_id: str):
    asset = get_asset(asset_id)
    with connection() as db:
        annotated = db.execute("SELECT service_version FROM annotations WHERE asset_id=?", (asset_id,)).fetchone()
        if not annotated:
            raise HTTPException(409, "Сначала сохраните исправленную разметку")
        already = db.execute(
            "SELECT 1 FROM retrain_queue WHERE asset_id=? AND status='queued' AND kind='human'", (asset_id,)
        ).fetchone() is not None
        db.execute(
            """
            INSERT INTO retrain_queue(asset_id,status,queued_at,training_job_id,kind) VALUES (?,?,?,NULL,'human')
            ON CONFLICT(asset_id) DO UPDATE SET status='queued',queued_at=excluded.queued_at,
              training_job_id=NULL,kind='human'
            """,
            (asset_id, "queued", now()),
        )
        human_count = db.execute(
            "SELECT count(*) FROM retrain_queue WHERE status='queued' AND kind='human'"
        ).fetchone()[0]
    return {"queued": True, "already_queued": already, "human_count": human_count,
            "folder": str(CORRECTIONS / asset["id"])}


@app.get("/api/training/queue")
def training_queue():
    with connection() as db:
        counts = {row["kind"]: row["total"] for row in db.execute(
            "SELECT kind,count(*) AS total FROM retrain_queue WHERE status='queued' GROUP BY kind"
        )}
    return {"human": counts.get("human", 0), "model": counts.get("model", 0)}


@app.get("/api/assets/{asset_id}/annotation/export")
def export_annotation(asset_id: str, format: str = "csv"):
    asset = get_asset(asset_id)
    if format not in ("csv", "xlsx"):
        raise HTTPException(400, "Поддерживаются только CSV и XLSX")
    if not asset.get("table_json"):
        raise HTTPException(409, "Таблица предсказания ещё не готова")
    row = json.loads(asset["table_json"])
    ordered = {column: row.get(column, "") for column in TABLE_COLUMNS}
    base = Path(asset["name"]).stem or "prediction"
    if format == "csv":
        stream = io.StringIO(newline="")
        stream.write("\ufeff")
        writer = csv.DictWriter(stream, fieldnames=TABLE_COLUMNS)
        writer.writeheader(); writer.writerow(ordered)
        body = io.BytesIO(stream.getvalue().encode("utf-8"))
        media_type = "text/csv; charset=utf-8"
    else:
        from openpyxl import Workbook
        workbook = Workbook(); sheet = workbook.active; sheet.title = "Результат"
        sheet.append(TABLE_COLUMNS); sheet.append([ordered[column] for column in TABLE_COLUMNS])
        sheet.freeze_panes = "A2"; sheet.auto_filter.ref = sheet.dimensions
        for cells in sheet.columns:
            sheet.column_dimensions[cells[0].column_letter].width = min(48, max(12, max(len(str(cell.value or "")) for cell in cells) + 2))
        body = io.BytesIO(); workbook.save(body); body.seek(0)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ascii_name = "".join(c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in base)
    return StreamingResponse(body, media_type=media_type,
                             headers={"Content-Disposition": f'attachment; filename="{ascii_name or "prediction"}.{format}"'})


def prediction_sample(asset: dict):
    prediction = json.loads(asset["prediction_json"])
    region = prediction["region"]
    return {"path": asset["service_path"], "geometry": json.loads(asset["geometry_json"]),
            "region": region, "targets": prediction.get("quality_flags") or {},
            "reviewed": SPINE_REVIEWED if region == "SPINE" else HIP_REVIEWED}


@app.post("/api/training/start")
def start_training():
    with connection() as db:
        humans = [dict(row) for row in db.execute(
            """
            SELECT a.*,n.service_version,q.queued_at FROM retrain_queue q
            JOIN assets a ON a.id=q.asset_id JOIN annotations n ON n.asset_id=a.id
            WHERE q.status='queued' AND q.kind='human' ORDER BY q.queued_at
            """)]
    if not humans:
        raise HTTPException(409, "Чтобы дообучить модель, измените разметку хотя бы одного снимка и добавьте его в очередь.")
    human_hashes = {row["pixel_hash"] for row in humans}
    with connection() as db:
        candidates = [dict(row) for row in db.execute(
            """
            SELECT a.*,q.queued_at FROM retrain_queue q JOIN assets a ON a.id=q.asset_id
            WHERE q.status='queued' AND q.kind='model' AND a.corrected=0
              AND a.result_id IS NOT NULL AND a.prediction_json IS NOT NULL AND a.geometry_json IS NOT NULL
            ORDER BY q.queued_at,a.id
            """)]
    model_rows = []
    seen = set(human_hashes)
    for row in candidates:
        if not row["pixel_hash"] or row["pixel_hash"] in seen:
            continue
        seen.add(row["pixel_hash"]); model_rows.append(row)
    request_id = f"web-fit-{uuid.uuid4().hex}"
    payload = {
        "base_model_version": None,
        "human": {"images": [{"image_id": row["service_image_id"], "annotation_version": int(row["service_version"])} for row in humans]},
        "augmentation": {"n_pp": 5, "n_pn": 5, "n_nn": 5, "seed": 42},
        "learning_rate": 0.00002, "request_id": request_id,
    }
    if model_rows:
        payload["model"] = {"images": [prediction_sample(row) for row in model_rows]}
    result = api_json("POST", "/v1/model/partial_fit", expected=(202,), json=payload)
    job_id = result["job_id"]
    with connection() as db:
        db.execute(
            """
            INSERT INTO train_jobs(id,status,manifest_path,output_path,command,log,created_at,
              service_job_id,result_json,error) VALUES (?,?,?,?,?,?,?,?,?,NULL)
            """,
            (job_id, result.get("status", "queued"), str(CORRECTIONS), "", "/v1/model/partial_fit", "",
             now(), job_id, json.dumps(result, ensure_ascii=False)),
        )
        db.executemany("UPDATE retrain_queue SET status='assigned',training_job_id=? WHERE asset_id=?",
                       [(job_id, row["id"]) for row in humans + model_rows])
    return {"job_id": job_id, "status": "queued", "human": len(humans), "model": len(model_rows)}


@app.get("/api/training/jobs")
def training_jobs():
    with connection() as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM train_jobs ORDER BY created_at DESC LIMIT 20")]
    jobs = []
    for row in rows:
        try:
            service_id = row["service_job_id"] or row["id"]
            service = api_json("GET", f"/v1/jobs/{service_id}")
            logs = api_json("GET", f"/v1/jobs/{service_id}/logs?offset=0&limit=1000")
            status, result, error = service["status"], service.get("result"), service.get("error")
            with connection() as db:
                db.execute(
                    """
                    UPDATE train_jobs SET status=?,log=?,result_json=?,error=?,
                      finished_at=CASE WHEN ? IN ('completed','failed','interrupted','cancelled')
                      THEN COALESCE(finished_at,?) ELSE finished_at END WHERE id=?
                    """,
                    (status, json.dumps(logs, ensure_ascii=False), json.dumps(result, ensure_ascii=False) if result else None,
                     error, status, now(), row["id"]),
                )
                if status == "completed":
                    db.execute("UPDATE retrain_queue SET status='trained' WHERE training_job_id=?", (row["id"],))
                elif status in ("failed", "interrupted", "cancelled"):
                    db.execute("UPDATE retrain_queue SET status='queued',training_job_id=NULL WHERE training_job_id=?", (row["id"],))
            model = (result or {}).get("model") or {}
            jobs.append({"id": row["id"], "status": status, "created_at": row["created_at"], "error": error,
                         "stage": (result or {}).get("stage"), "model_status": model.get("status"),
                         "model_version": model.get("version"), "events": logs.get("events", [])})
        except HTTPException as exc:
            jobs.append({"id": row["id"], "status": row["status"], "created_at": row["created_at"],
                         "error": str(exc.detail), "events": []})
    return {"jobs": jobs}
