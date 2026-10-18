from __future__ import annotations

import json
import logging
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

from app.config import SLOT_SPECS, AppConfig
from app.state import StateManager, compute_content_hash

logger = logging.getLogger(__name__)


class GeneratedPostContent(BaseModel):
    topic: str = Field(..., min_length=5, description="Specific topic of the post")
    content_pillar: str = Field(
        ...,
        description="One of: fitness_workout, wellness_nutrition, luxury_lifestyle, mindset_independence",
    )
    content_idea: str = Field(..., min_length=10)
    intended_audience_appeal: str = Field(..., min_length=10)
    opening_hook: str = Field(..., min_length=8)
    caption: str = Field(..., min_length=40)
    hashtags: list[str] = Field(..., min_length=5, max_length=18)
    call_to_action: str = Field(..., min_length=5)
    visual_prompt: str = Field(..., min_length=30)
    accessibility_description: str = Field(..., min_length=15)
    ai_disclosure: str = Field(
        default="Virtual creator • AI-crafted visual & story",
        min_length=5,
    )
    reel_scene_description: str | None = Field(default=None)
    reel_on_screen_text: str | None = Field(default=None)
    reel_voiceover: str | None = Field(default=None)
    video_generation_prompt: str | None = Field(default=None)

    @field_validator("hashtags")
    @classmethod
    def normalize_hashtags(cls, v: list[str]) -> list[str]:
        cleaned: list[str] = []
        for tag in v:
            t = tag.strip().replace(" ", "")
            if not t:
                continue
            if not t.startswith("#"):
                t = f"#{t}"
            if t not in cleaned:
                cleaned.append(t)
        return cleaned[:15]

    def format_full_instagram_text(self) -> str:
        hashtags_line = " ".join(self.hashtags)
        parts = [
            self.opening_hook.strip(),
            "",
            self.caption.strip(),
            "",
            self.call_to_action.strip(),
            "",
            f"({self.ai_disclosure.strip()})",
            "",
            hashtags_line,
        ]
        return "\n".join(parts).strip()


class ContentGenerator:
    def __init__(self, config: AppConfig, state: StateManager) -> None:
        self.config = config
        self.state = state
        self.client = genai.Client(api_key=config.gemini_api_key)

    def _model_candidates(self) -> list[str]:
        models = [self.config.gemini_text_model]
        for fb in self.config.text_model_fallbacks:
            if fb not in models:
                models.append(fb)
        return models

    def generate_for_slot(
        self,
        slot_name: str,
        local_date: str,
    ) -> tuple[GeneratedPostContent, str]:
        spec = SLOT_SPECS[slot_name]
        profile = self.config.load_influencer_profile()
        pillars_cfg = profile.get("content_pillars", {})
        target_ratios = {k: float(v.get("target_ratio", 0.25)) for k, v in pillars_cfg.items()}

        chosen_pillar = self.state.select_next_pillar(
            target_ratios=target_ratios,
            preferred_pillars=spec.preferred_pillars,
        )
        recent_history = self.state.get_recent_history(limit=12)
        recent_topics = [
            f"- [{h.get('local_date')} {h.get('slot_name')} | {h.get('pillar')}]: {h.get('topic')}"
            for h in recent_history
        ]
        recent_summary = "\n".join(recent_topics) if recent_topics else "None yet."

        visual_id = profile.get("visual_identity", {})
        pillar_detail = pillars_cfg.get(chosen_pillar, {})

        system_instruction = (
            f"You are the creative director and voice of {profile.get('name', 'Shasha Laurent')} "
            f"(@{profile.get('handle', 'solace_shasha')}), a fictional {profile.get('apparent_age', 30)}-year-old "
            "divorced, ultra-wealthy, financially independent fitness and luxury lifestyle creator.\n"
            f"Biography & Voice: {json.dumps(profile.get('biography', {}))}\n"
            f"Visual Identity (MUST be preserved in visual_prompt and video_generation_prompt): "
            f"{json.dumps(visual_id)}\n\n"
            "STRICT RULES:\n"
            "1. Write in natural, confident, editorial first-person English. Avoid generic AI clichés "
            "('unlock your potential', 'tapestry', 'beacon', 'dive into', 'game-changer').\n"
            "2. Do NOT invent real brand sponsorships, medical degrees, clinical claims, or verifiable real-world awards.\n"
            "3. For health and fitness topics, focus on discipline, form, strength, mobility, whole-food nourishment, "
            "and recovery—never promise medical cures.\n"
            "4. Ensure visual_prompt describes a photorealistic 4:5 portrait photograph with specific camera lens, "
            "lighting, plausible exercise biomechanics/equipment, and exact facial/hair consistency.\n"
            "5. If media_type is REEL, populate reel_scene_description, reel_on_screen_text, reel_voiceover, "
            "and video_generation_prompt (optimized for vertical 9:16 Veo cinematic movement, 5-8 seconds continuous shot)."
        )

        user_prompt = (
            f"Generate a completely original Instagram {spec.media_type} post concept for:\n"
            f"- Date (Asia/Kolkata): {local_date}\n"
            f"- Slot: {slot_name} (Publication time: {spec.publish_hour_ist}:00 IST)\n"
            f"- Media Type: {spec.media_type}\n"
            f"- Assigned Content Pillar: {chosen_pillar} ({pillar_detail.get('description', '')})\n"
            f"- Suggested Settings: {json.dumps(pillar_detail.get('example_settings', []))}\n\n"
            f"Recent posts to avoid repeating:\n{recent_summary}\n\n"
            "Return valid JSON matching the required schema."
        )

        last_error: Exception | None = None
        for attempt in range(1, 4):
            for model_id in self._model_candidates():
                try:
                    logger.info(
                        "Generating structured post content with model=%s (attempt %d)",
                        model_id,
                        attempt,
                    )
                    response = self.client.models.generate_content(
                        model=model_id,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_mime_type="application/json",
                            response_schema=GeneratedPostContent,
                            temperature=0.85,
                        ),
                    )
                    if not response.text:
                        raise RuntimeError(f"Empty text response from model {model_id}")

                    raw_data: dict[str, Any] = json.loads(response.text)
                    raw_data["content_pillar"] = chosen_pillar
                    if spec.media_type == "REEL":
                        if not raw_data.get("video_generation_prompt"):
                            raw_data["video_generation_prompt"] = (
                                f"Vertical 9:16 photorealistic cinematic video of {visual_id.get('facial_features')}, "
                                f"{visual_id.get('hair')}, {visual_id.get('body_type')}. "
                                f"{raw_data.get('visual_prompt', '')}. Smooth gimbal movement, natural lighting."
                            )
                        if not raw_data.get("reel_scene_description"):
                            raw_data["reel_scene_description"] = raw_data["visual_prompt"]
                        if not raw_data.get("reel_on_screen_text"):
                            raw_data["reel_on_screen_text"] = raw_data["opening_hook"]

                    content = GeneratedPostContent.model_validate(raw_data)
                    content_hash = compute_content_hash(
                        content.caption, content.visual_prompt, slot_name
                    )
                    if self.state.is_content_hash_duplicate(content_hash) and attempt < 3:
                        logger.warning("Generated content hash %s collided with history; retrying.", content_hash)
                        user_prompt += f"\nEnsure a completely different angle from hash {content_hash}."
                        continue

                    return content, content_hash
                except Exception as exc:
                    last_error = exc
                    err_str = str(exc).lower()
                    if any(k in err_str for k in ("not found", "404", "not supported", "invalid argument")):
                        logger.warning(
                            "Model %s unavailable or incompatible (%s); trying next Gemini model fallback.",
                            model_id,
                            exc,
                        )
                        continue
                    logger.error("Gemini text generation error on %s: %s", model_id, exc)
                    break

        raise RuntimeError(f"Failed to generate post content via Gemini API: {last_error}")
