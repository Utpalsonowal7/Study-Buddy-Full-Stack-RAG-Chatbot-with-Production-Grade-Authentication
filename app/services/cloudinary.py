import time
from io import BytesIO

import cloudinary.uploader
import cloudinary.utils
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from app.config import RagSettings, require_credentials


class CloudinaryStorage:
    def __init__(self, settings: RagSettings):
        self.settings = settings

    def options(self) -> dict:
        require_credentials(self.settings.cloud_name, self.settings.cloud_key, self.settings.cloud_secret)
        return {"cloud_name": self.settings.cloud_name, "api_key": self.settings.cloud_key,
                "api_secret": self.settings.cloud_secret, "resource_type": "raw", "type": "authenticated"}

    async def upload(self, data: bytes, public_id: str) -> str:
        options = self.options()
        try:
            result = await run_in_threadpool(cloudinary.uploader.upload, BytesIO(data),
                                    public_id=public_id, overwrite=False, timeout=60, **options)
            url = result["secure_url"]
            if not isinstance(url, str) or not url.startswith("https://"):
                raise ValueError("Invalid Cloudinary upload URL")
            return url
        except Exception as exc:
            raise HTTPException(502, "Cloudinary upload failed. Check storage settings and account limits.") from exc

    async def delete(self, public_id: str) -> None:
        options = self.options()
        try:
            result = await run_in_threadpool(cloudinary.uploader.destroy, public_id, invalidate=True, timeout=60, **options)
            if result.get("result") not in {"ok", "not found"}:
                raise ValueError("Unexpected deletion result")
        except Exception as exc:
            raise HTTPException(502, "Cloudinary deletion failed. Please retry.") from exc

    def download_url(self, public_id: str) -> str:
        return cloudinary.utils.private_download_url(
            public_id, "", attachment=True, expires_at=int(time.time()) + 60, secure=True, **self.options()
        )
