"""Local configuration for the HRI questionnaire application."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _local_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


DATA_DIR = _local_path(os.getenv("HRI_DATA_DIR", "data"))
EXPORT_DIR = _local_path(os.getenv("HRI_EXPORT_DIR", "exports"))
BACKUP_DIR = _local_path(os.getenv("HRI_BACKUP_DIR", "backups"))
DATABASE_PATH = _local_path(
    os.getenv("DATABASE_PATH", os.getenv("HRI_DATABASE_PATH", "data/hri_questionnaires.sqlite3"))
)

# Only a salted PBKDF2 hash is accepted. There is intentionally no default PIN.
EXPERIMENTER_PIN_HASH = os.getenv("EXPERIMENTER_PIN_HASH", "").strip()
PAYMENT_QUESTIONNAIRE_URL = os.getenv("PAYMENT_QUESTIONNAIRE_URL", "").strip()
BACKUP_RETENTION = max(20, int(os.getenv("HRI_BACKUP_RETENTION", "20")))


def ensure_directories() -> None:
    """Create all application-owned storage directories."""

    for path in (DATA_DIR, EXPORT_DIR, BACKUP_DIR):
        path.mkdir(parents=True, exist_ok=True)
