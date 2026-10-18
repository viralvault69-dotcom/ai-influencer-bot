from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from app.config import AppConfig

logger = logging.getLogger(__name__)


class VideoGenerator:
    def __init__(
        self,
        config: AppConfig,
        poll_interval_seconds: int = 15,
        max_wait_seconds: int = 600,
    ) -> None:
        self.config = config
        self.poll_interval_seconds = poll_interval_seconds
        self.max_wait_seconds = max_wait_seconds
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.config.generated_assets_dir.mkdir(parents=True, exist_ok=True)

    def _model_candidates(self) -> list[str]:
        models = [self.config.gemini_video_model]
        for fb in self.config.video_model_fallbacks:
            if fb not in models:
                models.append(fb)
        return models

    def generate_reel_video(
        self,
        video_prompt: str,
        slot_name: str,
        local_date: str,
        profile: dict[str, Any],
        reference_image_path: Path | None = None,
    ) -> Path:
        vis = profile.get("visual_identity", {})
        enriched_prompt = (
            "Vertical 9:16 photorealistic cinematic Instagram Reel video, natural editorial lighting, "
            "smooth gimbal motion, realistic human movement and biomechanics. "
            f"Subject: {profile.get('name', 'Shasha Laurent')}, {vis.get('facial_features')}, "
            f"{vis.get('hair')}, {vis.get('skin_tone')}, {vis.get('body_type')}. "
            f"Action: {video_prompt}"
        )

        ref_image_obj: types.Image | None = None
        if reference_image_path and reference_image_path.exists():
            ref_image_obj = types.Image(
                image_bytes=reference_image_path.read_bytes(),
                mime_type="image/png",
            )

        video_bytes = self._invoke_veo_generation(
            prompt=enriched_prompt,
            reference_image=ref_image_obj,
        )

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = self.config.generated_assets_dir / f"{local_date}_{slot_name}_{timestamp}.mp4"
        output_path.write_bytes(video_bytes)

        if not self.verify_video_file(output_path):
            raise RuntimeError(f"Generated video at {output_path} is invalid or unplayable.")

        return output_path

    def _invoke_veo_generation(
        self,
        prompt: str,
        reference_image: types.Image | None,
    ) -> bytes:
        last_error: Exception | None = None

        for model_id in self._model_candidates():
            attempts = [reference_image, None] if reference_image is not None else [None]
            for img_input in attempts:
                try:
                    logger.info(
                        "Starting Veo video generation on model=%s (with_reference_image=%s)",
                        model_id,
                        img_input is not None,
                    )
                    kwargs: dict[str, Any] = {
                        "model": model_id,
                        "prompt": prompt,
                        "config": types.GenerateVideosConfig(
                            aspect_ratio="9:16",
                            person_generation="allow_adult",
                        ),
                    }
                    if img_input is not None:
                        kwargs["image"] = img_input

                    operation = self.client.models.generate_videos(**kwargs)
                    completed_op = self._poll_operation(operation)
                    return self._extract_video_bytes(completed_op)
                except Exception as exc:
                    last_error = exc
                    err_text = str(exc)
                    logger.warning(
                        "Veo generation attempt failed on model=%s (with_ref=%s): %s",
                        model_id,
                        img_input is not None,
                        err_text,
                    )
                    if img_input is not None and any(
                        token in err_text.lower()
                        for token in ("image", "unsupported", "invalid argument", "400")
                    ):
                        continue
                    break

        raise RuntimeError(
            f"Veo video generation failed across configured Gemini video models. "
            f"Exact API/model/quota error: {last_error}"
        )

    def _poll_operation(self, operation: Any) -> Any:
        elapsed = 0
        while not getattr(operation, "done", False):
            if elapsed >= self.max_wait_seconds:
                raise TimeoutError(
                    f"Veo video operation timed out after {self.max_wait_seconds} seconds."
                )
            time.sleep(self.poll_interval_seconds)
            elapsed += self.poll_interval_seconds
            operation = self.client.operations.get(operation)

        if getattr(operation, "error", None):
            raise RuntimeError(f"Veo operation completed with error: {operation.error}")
        return operation

    def _extract_video_bytes(self, operation: Any) -> bytes:
        response = getattr(operation, "response", None) or getattr(operation, "result", None)
        if not response:
            raise RuntimeError("Veo operation completed but returned an empty response payload.")

        generated_videos = getattr(response, "generated_videos", None) or []
        if not generated_videos:
            raise RuntimeError("Veo response contained no generated_videos entries.")

        first_video_entry = generated_videos[0]
        video_ref = getattr(first_video_entry, "video", None)
        if video_ref is None:
            raise RuntimeError("Veo generated_videos[0] is missing the 'video' attribute.")

        inline_bytes = getattr(video_ref, "video_bytes", None)
        if isinstance(inline_bytes, bytes) and len(inline_bytes) > 0:
            return inline_bytes

        downloaded = self.client.files.download(file=video_ref)
        if isinstance(downloaded, bytes) and len(downloaded) > 0:
            return downloaded

        raise RuntimeError("Failed to download non-empty video bytes from Veo operation result.")

    @staticmethod
    def verify_video_file(path: Path) -> bool:
        if not path.exists() or not path.is_file() or path.stat().st_size < 256:
            return False
        try:
            header = path.read_bytes()[:64]
            if b"ftyp" not in header:
                return False
        except OSError:
            return False

        try:
            proc = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if proc.returncode == 0 and "video" not in proc.stdout.lower():
                return False
        except FileNotFoundError:
            pass
        except Exception:
            return False

        return True
