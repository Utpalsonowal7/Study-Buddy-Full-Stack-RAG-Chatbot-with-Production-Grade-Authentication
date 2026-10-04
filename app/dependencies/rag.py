from fastapi import Depends

from app.config import RagSettings
from app.services.ai import AIProvider
from app.services.cloudinary import CloudinaryStorage


def get_settings() -> RagSettings:
    return RagSettings.from_env()


def get_ai(settings: RagSettings = Depends(get_settings)) -> AIProvider:
    return AIProvider(settings)


def get_storage(settings: RagSettings = Depends(get_settings)) -> CloudinaryStorage:
    return CloudinaryStorage(settings)
