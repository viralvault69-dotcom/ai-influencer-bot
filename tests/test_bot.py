from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from PIL import Image

from app.buffer_client import BufferClient
from app.config import (
    PROJECT_ROOT,
    AppConfig,
    compute_slot_publication_times,
    resolve_slot_from_schedule,
)
from app.content_generator import ContentGenerator, GeneratedPostContent
from app.image_generator import ImageGenerator
from app.media_host import MediaHost
from app.state import SlotRecord, StateManager, compute_content_hash
from app.video_generator import VideoGenerator


def _make_png_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 80), color=(180, 140, 120))
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_mp4_bytes() -> bytes:
    header = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    return header + (b"\x00" * 512)


def test_schedule_to_slot_mapping() -> None:
    assert resolve_slot_from_schedule(cron_expression="30 3 * * *") == "morning"
    assert resolve_slot_from_schedule(cron_expression="30 8 * * *") == "afternoon"
    assert resolve_slot_from_schedule(cron_expression="30 13 * * *") == "evening"
    assert resolve_slot_from_schedule(explicit_slot="afternoon") == "afternoon"


def test_ist_to_utc_timestamp_conversion() -> None:
    ref_morning = datetime(2026, 10, 18, 3, 30, 0, tzinfo=timezone.utc)
    local_date, target_ist, due_utc = compute_slot_publication_times("morning", ref_morning)
    assert local_date == "2026-10-18"
    assert target_ist.hour == 10 and target_ist.minute == 0
    assert due_utc == "2026-10-18T04:30:00Z"

    ref_afternoon = datetime(2026, 10, 18, 8, 30, 0, tzinfo=timezone.utc)
    local_date_af, target_ist_af, due_utc_af = compute_slot_publication_times("afternoon", ref_afternoon)
    assert local_date_af == "2026-10-18"
    assert target_ist_af.hour == 15
    assert due_utc_af == "2026-10-18T09:30:00Z"

    ref_evening = datetime(2026, 10, 18, 13, 30, 0, tzinfo=timezone.utc)
    local_date_ev, target_ist_ev, due_utc_ev = compute_slot_publication_times("evening", ref_evening)
    assert local_date_ev == "2026-10-18"
    assert target_ist_ev.hour == 20
    assert due_utc_ev == "2026-10-18T14:30:00Z"


def test_content_structure_validation_and_gemini_response(tmp_path: Path) -> None:
    cfg = AppConfig(
        gemini_api_key="test-key",
        buffer_api_key="buf-key",
        cloudinary_url="cloudinary://123:456@demo",
        state_file_path=tmp_path / "state.json",
    )
    state = StateManager(cfg.state_file_path)

    sample_payload = {
        "topic": "Morning heavy tempo squats and nervous system composure",
        "content_pillar": "fitness_workout",
        "content_idea": "Show controlled barbell work in a sunlit private gym",
        "intended_audience_appeal": "Resonates with disciplined professionals who train early",
        "opening_hook": "Discipline feels quieter than motivation ever will.",
        "caption": (
            "Sixty minutes on the platform before my phone leaves airplane mode. "
            "Slow eccentrics, clean depth, and zero negotiation with the bar."
        ),
        "hashtags": ["LuxuryFitness", "#StrengthTraining", "WomenWhoLift", "PilatesAndIron", "QuietLuxury"],
        "call_to_action": "What is the one non-negotiable anchor in your morning?",
        "visual_prompt": (
            "Photorealistic 4:5 portrait of Shasha Laurent in an espresso athletic set "
            "racking a brushed steel barbell in a travertine penthouse gym at 9 AM."
        ),
        "accessibility_description": "Shasha Laurent standing by a squat rack in a sunlit travertine gym.",
        "ai_disclosure": "Virtual creator • AI-crafted visual & story",
    }

    with patch("app.content_generator.genai.Client") as mock_client_cls:
        mock_instance = mock_client_cls.return_value
        mock_instance.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps(sample_payload)
        )
        gen = ContentGenerator(cfg, state)
        content, chash = gen.generate_for_slot("morning", "2026-10-18")

    assert isinstance(content, GeneratedPostContent)
    assert content.hashtags[0] == "#LuxuryFitness"
    assert len(chash) == 16


def test_image_generator_with_reference_and_verification(tmp_path: Path) -> None:
    cfg = AppConfig(
        gemini_api_key="test-key",
        character_reference_path=tmp_path / "character_reference.png",
        generated_assets_dir=tmp_path / "generated",
    )
    png_bytes = _make_png_bytes()

    with patch("app.image_generator.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_content.return_value = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(inline_data=SimpleNamespace(data=png_bytes))]
                    )
                )
            ]
        )
        img_gen = ImageGenerator(cfg)
        profile = cfg.load_influencer_profile()
        out_path = img_gen.generate_post_image(
            visual_prompt="Sunlit reformer Pilates session in bone-white activewear",
            slot_name="morning",
            local_date="2026-10-18",
            profile=profile,
        )

    assert out_path.exists()
    assert ImageGenerator.verify_image_file(out_path)
    assert cfg.character_reference_path.exists()


def test_video_operation_polling_and_download(tmp_path: Path) -> None:
    cfg = AppConfig(
        gemini_api_key="test-key",
        generated_assets_dir=tmp_path / "generated",
    )
    mp4_bytes = _make_mp4_bytes()

    pending_op = SimpleNamespace(done=False, error=None, response=None)
    completed_op = SimpleNamespace(
        done=True,
        error=None,
        response=SimpleNamespace(
            generated_videos=[
                SimpleNamespace(video=SimpleNamespace(video_bytes=None, uri="files/veo-123"))
            ]
        ),
    )

    with patch("app.video_generator.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_videos.return_value = pending_op
        mock_client.operations.get.return_value = completed_op
        mock_client.files.download.return_value = mp4_bytes

        vid_gen = VideoGenerator(cfg, poll_interval_seconds=0, max_wait_seconds=5)
        profile = cfg.load_influencer_profile()
        with patch("app.video_generator.subprocess.run", side_effect=FileNotFoundError):
            video_path = vid_gen.generate_reel_video(
                video_prompt="Controlled kettlebell swings on a coastal terrace at golden hour",
                slot_name="afternoon",
                local_date="2026-10-18",
                profile=profile,
            )

    assert video_path.exists()
    assert video_path.stat().st_size > 256


def test_media_hosting_and_url_validation(tmp_path: Path) -> None:
    cfg = AppConfig(
        gemini_api_key="test-key",
        cloudinary_url="cloudinary://12345:secret@mycloud",
    )
    img_file = tmp_path / "sample.png"
    img_file.write_bytes(_make_png_bytes())

    with (
        patch("app.media_host.cloudinary.uploader.upload") as mock_upload,
        patch("app.media_host.requests.head") as mock_head,
    ):
        mock_upload.return_value = {
            "secure_url": "https://res.cloudinary.com/mycloud/image/upload/v1/sample.png",
            "public_id": "ai_influencer_solace_shasha/sample",
            "format": "png",
            "bytes": len(_make_png_bytes()),
            "width": 64,
            "height": 80,
        }
        mock_head.return_value = SimpleNamespace(
            status_code=200,
            headers={"Content-Type": "image/png"},
        )
        host = MediaHost(cfg)
        hosted = host.upload_media(img_file, "IMAGE", "2026-10-18_morning")

    assert hosted.url.startswith("https://")
    assert hosted.resource_type == "image"


def test_buffer_success_and_graphql_error_handling() -> None:
    cfg = AppConfig(
        gemini_api_key="test-key",
        buffer_api_key="buf-live-token",
    )
    mock_session = MagicMock()

    reconcile_response = MagicMock(
        status_code=200,
        json=lambda: {"data": {"posts": {"edges": []}}},
    )
    create_success_response = MagicMock(
        status_code=200,
        json=lambda: {
            "data": {
                "createPost": {
                    "__typename": "PostActionSuccess",
                    "post": {
                        "id": "buf_post_999",
                        "status": "scheduled",
                        "dueAt": "2099-10-18T04:30:00Z",
                        "text": "Test caption",
                    },
                }
            }
        },
    )
    mock_session.post.side_effect = [reconcile_response, create_success_response]

    client = BufferClient(cfg, session=mock_session)
    res = client.schedule_instagram_post(
        text="Test caption",
        media_url="https://res.cloudinary.com/demo/image/upload/sample.png",
        media_type="IMAGE",
        due_at_iso="2099-10-18T04:30:00Z",
        alt_text="Alt description",
    )
    assert res.post_id == "buf_post_999"
    assert res.status == "scheduled"


def test_duplicate_prevention_and_persistent_state_recovery(tmp_path: Path) -> None:
    state_path = tmp_path / "posted_slots.json"
    state = StateManager(state_path)
    slot_key = StateManager.build_slot_key("2026-10-18", "morning", "2026-10-18T04:30:00Z")

    rec = SlotRecord(
        slot_key=slot_key,
        local_date="2026-10-18",
        slot_name="morning",
        due_at_utc="2026-10-18T04:30:00Z",
        media_type="IMAGE",
        pillar="fitness_workout",
        topic="Tempo squats",
        content_hash=compute_content_hash("caption", "prompt", "morning"),
        asset_url="https://res.cloudinary.com/demo/image/upload/sample.png",
        buffer_post_id="buf_123",
    )
    state.record_completed_post(rec)

    reloaded = StateManager(state_path)
    assert reloaded.is_slot_already_scheduled(slot_key) is True
    assert reloaded.is_content_hash_duplicate(rec.content_hash) is True


def test_github_actions_workflow_syntax() -> None:
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "auto-post.yml"
    assert workflow_path.exists()
    parsed: dict[str, Any] = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    triggers = parsed.get("on") or parsed.get(True)
    assert triggers is not None
    schedules = triggers.get("schedule", [])
    cron_list = [item["cron"] for item in schedules]
    assert "30 3 * * *" in cron_list
    assert "30 8 * * *" in cron_list
    assert "30 13 * * *" in cron_list
