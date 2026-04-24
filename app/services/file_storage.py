from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


class UploadStorageService:
    def __init__(self, uploads_dir: Path) -> None:
        self.uploads_dir = uploads_dir
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, upload: UploadFile) -> tuple[Path, int]:
        suffix = Path(upload.filename or "").suffix.lower() or ".bin"
        destination = self.uploads_dir / f"{uuid4().hex}{suffix}"
        payload = await upload.read()
        destination.write_bytes(payload)
        await upload.close()
        return destination, len(payload)
