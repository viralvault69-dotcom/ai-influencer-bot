from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image

from app.config import AppConfig

logger = logging.getLogger(__name__)


class ImageGenerator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.config.generated_assets_dir.mkdir(parents=True, exist_ok=True)
        self.config.character_reference_path.parent.mkdir(parents=True, exist_ok=True)

    def _model_candidates(self) -> list[str]:
        models = [self.config.gemini_image_model]
        for fb in self.config.image_model_fallbacks:
            if fb not in models:
                models.append(fb)
        return models

    @staticmethod
    def build_character_sheet_prompt(profile: dict[str, Any]) -> str:
        vis = profile.get("visual_identity", {})
        return (
            "Photorealistic editorial portrait photograph, 4:5 vertical aspect ratio, shot on 50mm f/2.0 lens. "
            f"Subject: {profile.get('name', 'Shasha Laurent')}, an adult woman appearing {profile.get('apparent_age', 30)} years old. "
            f"Facial features: {vis.get('facial_features')}. "
            f"Hair: {vis.get('hair')}. "
            f"Skin tone: {vis.get('skin_tone')}. "
            f"Body type: {vis.get('body_type')}. "
            f"Makeup & styling: {vis.get('makeup_and_styling')}. "
            "Setting: Sunlit architectural penthouse living room with travertine stone walls and soft morning daylight. "
            "Natural skin texture, anatomically accurate proportions, sharp focus on eyes, luxury editorial photography."
        )

    def ensure_character_reference(self, profile: dict[str, Any]) -> Path:
        ref_path = self.config.character_reference_path
        if self.verify_image_file(ref_path):
            return ref_path

        logger.info("Generating initial character reference image at %s", ref_path)
        prompt = self.build_character_sheet_prompt(profile)
        image_bytes = self._invoke_image_generation(prompt=prompt, reference_image_path=None)
        self._save_and_verify_image(image_bytes, ref_path)
        return ref_path

    def generate_post_image(
        self,
        visual_prompt: str,
        slot_name: str,
        local_date: str,
        profile: dict[str, Any],
    ) -> Path:
        ref_path = self.ensure_character_reference(profile)
        vis = profile.get("visual_identity", {})
        full_prompt = (
            "Photorealistic 4:5 vertical Instagram portrait photography, editorial luxury lifestyle and fitness aesthetic. "
            "Maintain strict facial and physical identity consistency with the reference character: "
            f"{vis.get('facial_features')}; {vis.get('hair')}; {vis.get('skin_tone')}; {vis.get('body_type')}. "
            f"Scene & Action: {visual_prompt}. "
            "Ensure realistic human anatomy, natural hands with five fingers, authentic gym equipment geometry, "
            "and natural skin texture without plastic smoothing."
        )

        image_bytes = self._invoke_image_generation(
            prompt=full_prompt,
            reference_image_path=ref_path if ref_path.exists() else None,
        )

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = self.config.generated_assets_dir / f"{local_date}_{slot_name}_{timestamp}.png"
        self._save_and_verify_image(image_bytes, out_path)
        return out_path

    def _invoke_image_generation(
        self,
        prompt: str,
        reference_image_path: Path | None,
    ) -> bytes:
        last_error: Exception | None = None

        for model_id in self._model_candidates():
            try:
                if model_id.startswith("imagen-"):
                    logger.info("Requesting image via generate_images on model=%s", model_id)
                    result = self.client.models.generate_images(
                        model=model_id,
                        prompt=prompt,
                        config=types.GenerateImagesConfig(
                            number_of_images=1,
                            aspect_ratio="3:4",
                            person_generation="allow_adult",
                            output_mime_type="image/png",
                        ),
                    )
                    generated = getattr(result, "generated_images", None) or []
                    if not generated or not getattr(generated[0], "image", None):
                        raise RuntimeError(f"Model {model_id} returned no generated_images.")
                    raw_bytes = generated[0].image.image_bytes
                    if not raw_bytes:
                        raise RuntimeError(f"Model {model_id} returned empty image_bytes.")
                    return raw_bytes

                logger.info(
                    "Requesting image via generate_content on model=%s (with_ref=%s)",
                    model_id,
                    bool(reference_image_path and reference_image_path.exists()),
                )
                contents: list[Any] = []
                if reference_image_path and reference_image_path.exists():
                    ref_bytes = reference_image_path.read_bytes()
                    contents.append(
                        types.Part.from_bytes(data=ref_bytes, mime_type="image/png")
                    )
                    contents.append(
                        "Use the attached character reference image to keep the exact same woman's face, "
                        "bone structure, hair color, and complexion in the new scene described below:\n"
                        + prompt
                    )
                else:
                    contents.append(prompt)

                response = self.client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )
                extracted = self._extract_inline_image_bytes(response)
                if extracted:
                    return extracted
                raise RuntimeError(f"Model {model_id} response did not contain inline image bytes.")

            except Exception as exc:
                last_error = exc
                logger.warning("Image generation attempt with %s failed: %s", model_id, exc)
                continue

        raise RuntimeError(f"All configured Gemini image models failed. Last error: {last_error}")

    @staticmethod
    def _extract_inline_image_bytes(response: Any) -> bytes | None:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    data = inline_data.data
                    if isinstance(data, bytes) and len(data) > 0:
                        return data
        return None

    @staticmethod
    def _save_and_verify_image(image_bytes: bytes, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(output_path, format="PNG")

        if not ImageGenerator.verify_image_file(output_path):
            raise RuntimeError(f"Saved image at {output_path} failed post-write verification.")

    @staticmethod
    def verify_image_file(path: Path) -> bool:
        if not path.exists() or not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False
