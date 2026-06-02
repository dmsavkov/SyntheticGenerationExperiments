"""Auto-generated row ids for playground synthetics."""

from __future__ import annotations

import uuid


def new_row_id(prefix: str = "synth") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"
