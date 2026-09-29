import os
from pathlib import Path
import stat
import tempfile
import zipfile


os.environ.setdefault("USER_WEB_DATA", str(Path(tempfile.gettempdir()) / "dxa-web-security-tests"))

from user_web_prototype.backend import unsafe_archive_member


def test_allows_regular_nested_dicom():
    assert unsafe_archive_member(zipfile.ZipInfo("study/scan.dcm")) is None


def test_rejects_parent_traversal():
    assert unsafe_archive_member(zipfile.ZipInfo("../scan.dcm")) == "архив содержит небезопасный путь"


def test_rejects_absolute_and_windows_drive_paths():
    assert unsafe_archive_member(zipfile.ZipInfo("/scan.dcm")) == "архив содержит небезопасный путь"
    assert unsafe_archive_member(zipfile.ZipInfo("C:/scan.dcm")) == "архив содержит небезопасный путь"


def test_rejects_executable_content():
    assert unsafe_archive_member(zipfile.ZipInfo("run.exe")) == "исполняемые файлы в архиве запрещены"


def test_rejects_symbolic_links():
    info = zipfile.ZipInfo("scan.dcm")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    assert unsafe_archive_member(info) == "символические ссылки в архиве запрещены"
