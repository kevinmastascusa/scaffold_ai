# -*- coding: utf-8 -*-
"""
Quality MetaPrompt utilities.

Provides a deterministic, sectioned meta-prompt for
"blueprint-quality" outputs. Activated when SC_QUALITY_MODE=true.
Adds dynamic, domain-aware requirements so metrics adapt to the
user request and context instead of being hard-coded.
"""

import os
from typing import Optional


def _device_note() -> str:
    use_gpu = str(os.getenv("SC_INCLUDE_CUDA", "")).lower() in (
        "1", "true", "yes"
    )
    if use_gpu:
        return (
            "- GPU available: allow heavier workflows (Torch/ONNX), "
            "but list CPU fallbacks.\n"
        )
    return "- Assume CPU-only environment; avoid heavy compute suggestions.\n"


def build_blueprint_header() -> str:
    return (
        "You are Scaffold AI, an expert curriculum assistant. Produce a "
        "concise, high-signal plan with the exact sections below. Avoid "
        "meta-commentary.\n\n"
        "Sections (use these headings exactly):\n"
        "- Learning outcomes\n"
        "- Six sustainability themes (each: metrics • mini‑lab • tools)\n"
        "- 8‑week module map\n"
        "- Labs\n"
        "- Capstone\n"
        "- Grading rubric\n"
        "- Ready-to-assign prompts\n"
        "- Tool set\n\n"
        "Requirements:\n"
    )


def _dynamic_metrics(user_query: str, context: Optional[str]) -> str:
    """Return general, domain-agnostic metric guidance (no conditionals)."""
    return (
        "- Use domain‑appropriate quantitative metrics with clear units; "
        "prioritize reproducible calculations.\n"
    )


def build_blueprint_prompt(
    user_query: str,
    context: Optional[str] = None,
) -> str:
    """Compose meta-prompt with dynamic metrics, device note, and context."""
    header = build_blueprint_header()
    metrics = _dynamic_metrics(user_query, context)
    device = _device_note()
    ctx_block = f"\nContext:\n{context}\n" if context else "\n"
    requirements = (
        "- Prefer bullets and short sentences; no filler.\n"
    )
    return (
        f"{header}{metrics}{requirements}{device}{ctx_block}User request:\n"
        f"{user_query}\n\nAnswer:"
    )
