from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_content_hash(caption: str, visual_prompt: str, slot_name: str) -> str:
    normalized = f"{slot_name.strip().lower()}|{caption.strip()}|{visual_prompt.strip()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass
class SlotRecord:
    slot_key: str
    local_date: str
    slot_name: str
    due_at_utc: str
    media_type: str
    pillar: str = ""
    topic: str = ""
    generation_status: str = "pending"
    content_hash: str = ""
    caption: str = ""
    local_media_path: str = ""
    asset_url: str = ""
    buffer_post_id: str = ""
    channel_id: str = ""
    scheduling_status: str = "unscheduled"
    error_details: str = ""
    last_attempt_at: str = ""
    completed_at: str = ""
    content_payload: dict[str, Any] = field(default_factory=dict)


class StateManager:
    MAX_HISTORY_ITEMS = 60

    def __init__(self, state_file_path: Path) -> None:
        self.state_file_path = state_file_path
        self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _default_state(self) -> dict[str, Any]:
        return {
            "character_reference": {
                "local_path": "assets/character_reference.png",
                "public_url": "",
                "generated_at": None,
            },
            "slots": {},
            "recent_history": [],
        }

    def _load(self) -> dict[str, Any]:
        if not self.state_file_path.exists():
            data = self._default_state()
            self._write(data)
            return data
        try:
            with self.state_file_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                return self._default_state()
            loaded.setdefault("character_reference", self._default_state()["character_reference"])
            loaded.setdefault("slots", {})
            loaded.setdefault("recent_history", [])
            return loaded
        except (json.JSONDecodeError, OSError):
            return self._default_state()

    def _write(self, data: dict[str, Any]) -> None:
        temp_path = self.state_file_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        temp_path.replace(self.state_file_path)

    def save(self) -> None:
        self._write(self._data)

    @staticmethod
    def build_slot_key(local_date: str, slot_name: str, due_at_utc: str) -> str:
        return f"{local_date}_{slot_name}_{due_at_utc}"

    def get_slot(self, slot_key: str) -> SlotRecord | None:
        raw = self._data["slots"].get(slot_key)
        if not raw:
            return None
        return SlotRecord(**raw)

    def upsert_slot(self, record: SlotRecord) -> None:
        record.last_attempt_at = utc_now_iso()
        self._data["slots"][record.slot_key] = asdict(record)
        self.save()

    def is_slot_already_scheduled(self, slot_key: str) -> bool:
        record = self.get_slot(slot_key)
        if not record:
            return False
        return (
            record.scheduling_status == "scheduled"
            and bool(record.buffer_post_id)
            and bool(record.asset_url)
        )

    def is_content_hash_duplicate(self, content_hash: str) -> bool:
        for item in self._data.get("recent_history", []):
            if item.get("content_hash") == content_hash:
                return True
        for slot_data in self._data.get("slots", {}).values():
            if (
                slot_data.get("content_hash") == content_hash
                and slot_data.get("scheduling_status") == "scheduled"
            ):
                return True
        return False

    def record_completed_post(self, record: SlotRecord) -> None:
        record.generation_status = "completed"
        record.scheduling_status = "scheduled"
        record.completed_at = utc_now_iso()
        record.error_details = ""
        self._data["slots"][record.slot_key] = asdict(record)

        history_entry = {
            "slot_key": record.slot_key,
            "local_date": record.local_date,
            "slot_name": record.slot_name,
            "due_at_utc": record.due_at_utc,
            "pillar": record.pillar,
            "topic": record.topic,
            "content_hash": record.content_hash,
            "buffer_post_id": record.buffer_post_id,
            "asset_url": record.asset_url,
            "completed_at": record.completed_at,
        }
        history = [
            h for h in self._data.get("recent_history", []) if h.get("slot_key") != record.slot_key
        ]
        history.append(history_entry)
        self._data["recent_history"] = history[-self.MAX_HISTORY_ITEMS :]
        self.save()

    def record_failure(self, record: SlotRecord, error_message: str) -> None:
        record.error_details = error_message[:1000]
        if record.scheduling_status != "scheduled":
            record.scheduling_status = "failed"
        if record.generation_status != "completed":
            record.generation_status = "failed"
        self.upsert_slot(record)

    def get_recent_history(self, limit: int = 15) -> list[dict[str, Any]]:
        return list(self._data.get("recent_history", [])[-limit:])

    def get_character_reference(self) -> dict[str, Any]:
        return dict(self._data.get("character_reference", {}))

    def set_character_reference(self, local_path: str, public_url: str = "") -> None:
        self._data["character_reference"] = {
            "local_path": local_path,
            "public_url": public_url,
            "generated_at": utc_now_iso(),
        }
        self.save()

    def select_next_pillar(
        self,
        target_ratios: dict[str, float],
        preferred_pillars: tuple[str, ...],
    ) -> str:
        history = self.get_recent_history(limit=20)
        last_pillar = history[-1].get("pillar") if history else None

        counts: dict[str, int] = {k: 0 for k in target_ratios}
        for item in history:
            p = item.get("pillar")
            if p in counts:
                counts[p] += 1
        total = sum(counts.values())

        candidates = list(preferred_pillars) if preferred_pillars else list(target_ratios.keys())
        non_repeating = [p for p in candidates if p != last_pillar] or candidates

        best_pillar = non_repeating[0]
        largest_deficit = -999.0
        for pillar in non_repeating:
            target = target_ratios.get(pillar, 0.25)
            actual = (counts.get(pillar, 0) / total) if total > 0 else 0.0
            deficit = target - actual
            if deficit > largest_deficit:
                largest_deficit = deficit
                best_pillar = pillar

        return best_pillar
