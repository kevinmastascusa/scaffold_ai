"""
File-backed conversation memory with token budgeting and robust writes.

Design goals:
- Atomic writes (temp file + replace) to avoid partial/corrupted JSON.
- UTF-8 encoding, safe reads with fallback to empty list on errors.
- Token budgeting by approximate word count (chars/4 heuristic).
- Simple deduplication of consecutive duplicate messages.
- Portable, dependency-free (no external file-lock library).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List


class MemoryStore:
    def __init__(
        self,
        base_dir: Path | str = Path("conversations"),
        max_messages: int = 100,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # Upper bound on stored messages per session file
        self.max_messages = max(1, int(max_messages))

    def _path(self, session_id: str) -> Path:
        safe_id = (session_id or "").strip() or "default"
        return self.base_dir / f"{safe_id}.json"

    def _atomic_write_json(self, path: Path, obj: Any) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        # Write to temp file first
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        # Atomically replace
        os.replace(tmp_path, path)

    def load(self, session_id: str) -> List[Dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except Exception:
            # Corrupt file: rename and start fresh
            try:
                path.rename(path.with_suffix(path.suffix + ".corrupt"))
            except Exception:
                pass
            return []

    def save(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        # Trim to max_messages
        trimmed = messages[-self.max_messages :]
        # Deduplicate consecutive duplicates by content+type
        deduped: List[Dict[str, Any]] = []
        for msg in trimmed:
            if (
                deduped
                and deduped[-1].get("type") == msg.get("type")
                and (deduped[-1].get("content") or "").strip()
                == (msg.get("content") or "").strip()
            ):
                continue
            deduped.append(msg)
        self._atomic_write_json(self._path(session_id), deduped)

    def append(self, session_id: str, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = self.load(session_id)
        messages.append(message)
        self.save(session_id, messages)
        return messages

    def clear(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.exists():
            try:
                path.unlink()
            except Exception:
                # As a fallback, truncate the file
                try:
                    self._atomic_write_json(path, [])
                except Exception:
                    pass

    def get_context_text(
        self,
        session_id: str,
        token_budget: int = 600,
    ) -> str:
        """Return a compact text context from the last messages.

        token_budget uses a rough 1 token ≈ 4 chars heuristic.
        Messages are included newest→oldest until budget is met, then reversed.
        """
        messages = self.load(session_id)
        if not messages:
            return ""

        budget_chars = max(0, token_budget) * 4
        parts: List[str] = []
        used = 0
        for msg in reversed(messages):
            role = msg.get("type") or msg.get("role") or "user"
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            # Format line
            line = f"User: {content}" if role == "user" else f"Assistant: {content}"
            # Pre-check budget
            if used + len(line) + 1 > budget_chars:
                break
            parts.append(line)
            used += len(line) + 1

        parts.reverse()
        return "\n".join(parts)


