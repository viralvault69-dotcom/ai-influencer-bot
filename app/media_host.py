from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cloudinary
import cloudinary.uploader
import requests

from app.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HostedMedia:
    url: str
    public_id: str
    resource_type: str
    format: str
    bytes_size: int
    width: int | None = None
    height: int | None = None


def _redact_secrets(message: str) -> str:
    return re.sub(r"cloudinary://[^@\s]+@", "cloudinary://***:***@", message)


class MediaHost:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        if not self.config.cloudinary_url:
            raise EnvironmentError("CLOUDINARY_URL is required for hosting generated media.")
        if not self.config.cloudinary_url.startswith("cloudinary://"):
            raise ValueError("CLOUDINARY_URL must follow the format cloudinary://API_KEY:API_SECRET@CLOUD_NAME")
        cloudinary.config(cloudinary_url=self.config.cloudinary_url, secure=True)

    def upload_media(
        self,
        local_path: Path,
        media_type: str,
        slot_key: str,
    ) -> HostedMedia:
        if not local_path.exists() or local_path.stat().st_size == 0:
            raise FileNotFoundError(f"Media file for upload does not exist or is empty: {local_path}")

        resource_type = "video" if media_type.upper() == "REEL" else "image"
        try:
            logger.info(
                "Uploading %s (%s) to Cloudinary for slot=%s",
                local_path.name,
                resource_type,
                slot_key,
            )
            response = cloudinary.uploader.upload(
                str(local_path),
                resource_type=resource_type,
                folder="ai_influencer_solace_shasha",
                public_id=slot_key.replace(":", "-"),
                overwrite=True,
            )
        except Exception as exc:
            safe_err = _redact_secrets(str(exc))
            raise RuntimeError(f"Cloudinary upload failed: {safe_err}") from None

        secure_url = str(response.get("secure_url") or "")
        if not secure_url.startswith("https://"):
            raise RuntimeError(f"Cloudinary did not return a valid public HTTPS URL: {secure_url}")

        self.verify_public_url(secure_url, expected_resource_type=resource_type)

        return HostedMedia(
            url=secure_url,
            public_id=str(response.get("public_id") or ""),
            resource_type=resource_type,
            format=str(response.get("format") or ""),
            bytes_size=int(response.get("bytes") or local_path.stat().st_size),
            width=response.get("width"),
            height=response.get("height"),
        )

    @staticmethod
    def verify_public_url(url: str, expected_resource_type: str) -> None:
        if not url.startswith("https://"):
            raise ValueError(f"Media URL must use HTTPS: {url}")

        resp = requests.head(url, timeout=20, allow_redirects=True)
        if resp.status_code >= 400 or resp.status_code == 405:
            resp = requests.get(url, stream=True, timeout=20)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Hosted media URL {url} is not publicly accessible (HTTP {resp.status_code})."
            )

        content_type = resp.headers.get("Content-Type", "").lower()
        if expected_resource_type == "image" and not content_type.startswith("image/"):
            raise RuntimeError(
                f"Expected image/* Content-Type from {url}, got '{content_type}'."
            )
        if expected_resource_type == "video" and not (
            content_type.startswith("video/") or "octet-stream" in content_type
        ):
            raise RuntimeError(
                f"Expected video/* Content-Type from {url}, got '{content_type}'."
            )
