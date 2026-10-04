import os
from dataclasses import dataclass

from fastapi import HTTPException


def require_credentials(*values: str) -> None:
    if any(not value or value.lower().startswith(("replace", "your_", "dummy", "placeholder")) for value in values):
        raise HTTPException(503, "Provider credentials are not configured. Update the server environment settings.")


@dataclass(frozen=True)
class Settings:
    api_key: str
    api_url: str
    embedding_model: str
    chat_model: str
    cloud_name: str
    cloud_key: str
    cloud_secret: str
    max_file_bytes: int = 10 * 1024 * 1024
    max_chunks_per_document: int = 200
    max_chunks_per_user: int = 2000
    top_k: int = 6
    min_score: float = 0.25

    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.getenv("RAG_API_KEY", ""),
            api_url=os.getenv("RAG_API_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
            chat_model=os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini"),
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", ""),
            cloud_key=os.getenv("CLOUDINARY_API_KEY", ""),
            cloud_secret=os.getenv("CLOUDINARY_API_SECRET", ""),
        )
