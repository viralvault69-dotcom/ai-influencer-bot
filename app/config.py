from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IST_TZ = ZoneInfo("Asia/Kolkata")
UTC_TZ = ZoneInfo("UTC")


@dataclass(frozen=True)
class SlotSpec:
    name: str
    cron_utc: str
    generation_hour_ist: int
    publish_hour_ist: int
    media_type: str  # "IMAGE" or "REEL"
    preferred_pillars: tuple[str, ...]


SLOT_SPECS: dict[str, SlotSpec] = {
    "morning": SlotSpec(
        name="morning",
        cron_utc="30 3 * * *",
        generation_hour_ist=9,
        publish_hour_ist=10,
        media_type="IMAGE",
        preferred_pillars=("fitness_workout", "wellness_nutrition"),
    ),
    "afternoon": SlotSpec(
        name="afternoon",
        cron_utc="30 8 * * *",
        generation_hour_ist=14,
        publish_hour_ist=15,
        media_type="REEL",
        preferred_pillars=("fitness_workout", "luxury_lifestyle", "mindset_independence"),
    ),
    "evening": SlotSpec(
        name="evening",
        cron_utc="30 13 * * *",
        generation_hour_ist=19,
        publish_hour_ist=20,
        media_type="IMAGE",
        preferred_pillars=(
            "luxury_lifestyle",
            "wellness_nutrition",
            "mindset_independence",
            "fitness_workout",
        ),
    ),
}

CRON_TO_SLOT: dict[str, str] = {spec.cron_utc: name for name, spec in SLOT_SPECS.items()}


def _parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AppConfig:
    gemini_api_key: str
    gemini_text_model: str = "gemini-3.8-flash"
    gemini_image_model: str = "gemini-nano-banana-2.1"
    gemini_video_model: str = "veo-3.1-generate-preview"
    buffer_api_key: str = ""
    buffer_organization_id: str = "6ad3ea55ae4e83b03a8ee01d"
    buffer_channel_id: str = "6ad3eff6778b240e2f3fae0d"
    buffer_connected_account: str = "solace_shasha"
    cloudinary_url: str = ""
    publish_mode: str = "live"
    enable_image_generation: bool = True
    enable_video_generation: bool = True
    influencer_config_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "config" / "influencer.json"
    )
    state_file_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "state" / "posted_slots.json"
    )
    character_reference_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "assets" / "character_reference.png"
    )
    generated_assets_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "assets" / "generated"
    )

    text_model_fallbacks: tuple[str, ...] = (
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    )
    image_model_fallbacks: tuple[str, ...] = (
        "gemini-2.5-flash-image-preview",
        "imagen-3.0-generate-002",
    )
    video_model_fallbacks: tuple[str, ...] = (
        "veo-2.0-generate-001",
    )

    @classmethod
    def from_env(cls) -> "AppConfig":
        publish_mode = os.getenv("PUBLISH_MODE", "live").strip().lower()
        if publish_mode != "live":
            raise ValueError(
                f"Invalid PUBLISH_MODE='{publish_mode}'. This application operates strictly in 'live' mode."
            )

        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_text_model=os.getenv("GEMINI_TEXT_MODEL", "gemini-3.8-flash").strip(),
            gemini_image_model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-nano-banana-2.1").strip(),
            gemini_video_model=os.getenv("GEMINI_VIDEO_MODEL", "veo-3.1-generate-preview").strip(),
            buffer_api_key=os.getenv("BUFFER_API_KEY", "").strip(),
            buffer_organization_id=os.getenv(
                "BUFFER_ORGANIZATION_ID", "6ad3ea55ae4e83b03a8ee01d"
            ).strip(),
            buffer_channel_id=os.getenv(
                "BUFFER_CHANNEL_ID", "6ad3eff6778b240e2f3fae0d"
            ).strip(),
            cloudinary_url=os.getenv("CLOUDINARY_URL", "").strip(),
            publish_mode=publish_mode,
            enable_image_generation=_parse_bool(os.getenv("ENABLE_IMAGE_GENERATION"), True),
            enable_video_generation=_parse_bool(os.getenv("ENABLE_VIDEO_GENERATION"), True),
        )

    def load_influencer_profile(self) -> dict[str, Any]:
        if not self.influencer_config_path.exists():
            raise FileNotFoundError(
                f"Influencer configuration not found at {self.influencer_config_path}"
            )
        with self.influencer_config_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def validate_for_slot(self, slot_name: str) -> None:
        if slot_name not in SLOT_SPECS:
            raise ValueError(f"Unknown slot '{slot_name}'. Expected one of {list(SLOT_SPECS)}.")
        missing: list[str] = []
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")
        if not self.buffer_api_key:
            missing.append("BUFFER_API_KEY")
        if not self.buffer_organization_id:
            missing.append("BUFFER_ORGANIZATION_ID")
        if not self.buffer_channel_id:
            missing.append("BUFFER_CHANNEL_ID")
        if not self.cloudinary_url:
            missing.append("CLOUDINARY_URL")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables for live execution: {', '.join(missing)}"
            )

        spec = SLOT_SPECS[slot_name]
        if spec.media_type == "IMAGE" and not self.enable_image_generation:
            raise RuntimeError(
                f"Slot '{slot_name}' requires image generation, but ENABLE_IMAGE_GENERATION is false."
            )
        if spec.media_type == "REEL" and not self.enable_video_generation:
            raise RuntimeError(
                f"Slot '{slot_name}' requires video generation, but ENABLE_VIDEO_GENERATION is false."
            )


def resolve_slot_from_schedule(
    explicit_slot: str | None = None,
    cron_expression: str | None = None,
    now_utc: datetime | None = None,
) -> str:
    if explicit_slot:
        normalized = explicit_slot.strip().lower()
        if normalized in SLOT_SPECS:
            return normalized
        raise ValueError(f"Invalid explicit slot '{explicit_slot}'. Valid options: {list(SLOT_SPECS)}")

    if cron_expression:
        normalized_cron = " ".join(cron_expression.strip().split())
        if normalized_cron in CRON_TO_SLOT:
            return CRON_TO_SLOT[normalized_cron]
        raise ValueError(f"Unrecognized schedule cron '{cron_expression}'.")

    ref_utc = now_utc or datetime.now(tz=UTC_TZ)
    ref_ist = ref_utc.astimezone(IST_TZ)
    hour = ref_ist.hour
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    return "evening"


def compute_slot_publication_times(
    slot_name: str,
    reference_now_utc: datetime | None = None,
) -> tuple[str, datetime, str]:
    if slot_name not in SLOT_SPECS:
        raise ValueError(f"Unknown slot '{slot_name}'.")

    spec = SLOT_SPECS[slot_name]
    now_utc = (reference_now_utc or datetime.now(tz=UTC_TZ)).astimezone(UTC_TZ)
    now_ist = now_utc.astimezone(IST_TZ)

    target_date_ist = now_ist.date()
    target_dt_ist = datetime.combine(
        target_date_ist,
        time(hour=spec.publish_hour_ist, minute=0, second=0),
        tzinfo=IST_TZ,
    )

    if target_dt_ist <= now_ist + timedelta(minutes=5):
        target_date_ist = target_date_ist + timedelta(days=1)
        target_dt_ist = datetime.combine(
            target_date_ist,
            time(hour=spec.publish_hour_ist, minute=0, second=0),
            tzinfo=IST_TZ,
        )

    target_dt_utc = target_dt_ist.astimezone(UTC_TZ)
    due_at_iso = target_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    local_date_str = target_date_ist.isoformat()
    return local_date_str, target_dt_ist, due_at_iso
