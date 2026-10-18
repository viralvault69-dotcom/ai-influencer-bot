from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from app.buffer_client import BufferClient
from app.config import (
    SLOT_SPECS,
    AppConfig,
    compute_slot_publication_times,
    resolve_slot_from_schedule,
)
from app.content_generator import ContentGenerator, GeneratedPostContent
from app.image_generator import ImageGenerator
from app.media_host import MediaHost
from app.state import SlotRecord, StateManager, compute_content_hash
from app.video_generator import VideoGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_influencer_bot")


def run_slot(slot_name: str, config: AppConfig) -> SlotRecord:
    config.validate_for_slot(slot_name)
    spec = SLOT_SPECS[slot_name]
    profile = config.load_influencer_profile()
    state = StateManager(config.state_file_path)

    local_date, target_dt_ist, due_at_utc = compute_slot_publication_times(slot_name)
    slot_key = StateManager.build_slot_key(local_date, slot_name, due_at_utc)

    logger.info(
        "Starting slot='%s' | local_date=%s | publish_ist=%s | due_at_utc=%s | media=%s",
        slot_name,
        local_date,
        target_dt_ist.isoformat(),
        due_at_utc,
        spec.media_type,
    )

    if state.is_slot_already_scheduled(slot_key):
        existing = state.get_slot(slot_key)
        assert existing is not None
        logger.info(
            "Slot %s is already scheduled in Buffer (post_id=%s). Skipping duplicate execution.",
            slot_key,
            existing.buffer_post_id,
        )
        return existing

    record = state.get_slot(slot_key) or SlotRecord(
        slot_key=slot_key,
        local_date=local_date,
        slot_name=slot_name,
        due_at_utc=due_at_utc,
        media_type=spec.media_type,
        channel_id=config.buffer_channel_id,
    )

    buffer_client = BufferClient(config)
    media_host = MediaHost(config)
    image_gen = ImageGenerator(config)

    try:
        buffer_client.verify_account_and_channel()

        ref_path = image_gen.ensure_character_reference(profile)
        char_ref_state = state.get_character_reference()
        if not char_ref_state.get("public_url"):
            hosted_ref = media_host.upload_media(ref_path, "IMAGE", "character_reference_master")
            state.set_character_reference(str(ref_path), hosted_ref.url)

        if record.content_payload and record.caption:
            content = GeneratedPostContent.model_validate(record.content_payload)
            content_hash = record.content_hash or compute_content_hash(
                content.caption, content.visual_prompt, slot_name
            )
            logger.info("Resuming existing generated content for slot %s (hash=%s)", slot_key, content_hash)
        else:
            content_gen = ContentGenerator(config, state)
            content, content_hash = content_gen.generate_for_slot(slot_name, local_date)
            record.pillar = content.content_pillar
            record.topic = content.topic
            record.content_hash = content_hash
            record.caption = content.format_full_instagram_text()
            record.content_payload = content.model_dump()
            record.generation_status = "content_ready"
            state.upsert_slot(record)

        local_media_path = Path(record.local_media_path) if record.local_media_path else None
        media_valid = False
        if local_media_path and local_media_path.exists():
            media_valid = (
                VideoGenerator.verify_video_file(local_media_path)
                if spec.media_type == "REEL"
                else ImageGenerator.verify_image_file(local_media_path)
            )

        if not media_valid:
            if spec.media_type == "IMAGE":
                generated_path = image_gen.generate_post_image(
                    visual_prompt=content.visual_prompt,
                    slot_name=slot_name,
                    local_date=local_date,
                    profile=profile,
                )
            else:
                video_gen = VideoGenerator(config)
                video_prompt = content.video_generation_prompt or content.visual_prompt
                generated_path = video_gen.generate_reel_video(
                    video_prompt=video_prompt,
                    slot_name=slot_name,
                    local_date=local_date,
                    profile=profile,
                    reference_image_path=ref_path,
                )
            record.local_media_path = str(generated_path)
            record.generation_status = "media_ready"
            state.upsert_slot(record)
            local_media_path = generated_path

        assert local_media_path is not None

        if not record.asset_url or not record.asset_url.startswith("https://"):
            hosted = media_host.upload_media(
                local_path=local_media_path,
                media_type=spec.media_type,
                slot_key=slot_key,
            )
            record.asset_url = hosted.url
            record.generation_status = "hosted"
            state.upsert_slot(record)

        schedule_result = buffer_client.schedule_instagram_post(
            text=record.caption,
            media_url=record.asset_url,
            media_type=spec.media_type,
            due_at_iso=due_at_iso,
            alt_text=content.accessibility_description,
        )
        record.buffer_post_id = schedule_result.post_id
        record.channel_id = schedule_result.channel_id
        state.record_completed_post(record)

        logger.info(
            "Successfully scheduled %s on Buffer | post_id=%s | dueAt=%s | url=%s",
            spec.media_type,
            record.buffer_post_id,
            schedule_result.due_at,
            record.asset_url,
        )
        return record

    except Exception as exc:
        logger.error("Slot execution failed for %s: %s", slot_key, exc)
        state.record_failure(record, str(exc))
        raise


def run_verify(config: AppConfig) -> int:
    profile = config.load_influencer_profile()
    logger.info("Loaded influencer persona: %s (@%s)", profile.get("name"), profile.get("handle"))

    missing: list[str] = []
    for env_name, val in [
        ("GEMINI_API_KEY", config.gemini_api_key),
        ("BUFFER_API_KEY", config.buffer_api_key),
        ("CLOUDINARY_URL", config.cloudinary_url),
    ]:
        if not val:
            missing.append(env_name)

    if missing:
        logger.warning("Read-only check: Missing live credentials in environment: %s", ", ".join(missing))
        return 1

    from google import genai

    client = genai.Client(api_key=config.gemini_api_key)
    models = list(client.models.list())
    logger.info("Gemini API authenticated. Accessible models count: %d", len(models))

    buffer_client = BufferClient(config)
    channel_info = buffer_client.verify_account_and_channel()
    logger.info(
        "Buffer API authenticated. Verified channel id=%s name=%s service=%s",
        channel_info.get("id"),
        channel_info.get("name"),
        channel_info.get("service"),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Automated AI Influencer Bot (Gemini + Veo + Cloudinary + Buffer)"
    )
    parser.add_argument(
        "--slot",
        choices=["morning", "afternoon", "evening"],
        default=None,
        help="Explicit daily slot to execute.",
    )
    parser.add_argument(
        "--cron",
        default=os.getenv("GITHUB_EVENT_SCHEDULE", ""),
        help="UTC cron schedule expression passed by GitHub Actions.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run safe read-only credential and channel verification without generating or posting.",
    )
    parser.add_argument(
        "--init-character",
        action="store_true",
        help="Generate and save the master character reference image without creating a post.",
    )

    args = parser.parse_args(argv)
    config = AppConfig.from_env()

    if args.verify:
        return run_verify(config)

    if args.init_character:
        if not config.gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY is required to initialize character_reference.png")
        profile = config.load_influencer_profile()
        img_gen = ImageGenerator(config)
        path = img_gen.ensure_character_reference(profile)
        logger.info("Character reference ready at %s", path)
        return 0

    slot_name = resolve_slot_from_schedule(
        explicit_slot=args.slot,
        cron_expression=args.cron if args.cron else None,
    )
    run_slot(slot_name, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
