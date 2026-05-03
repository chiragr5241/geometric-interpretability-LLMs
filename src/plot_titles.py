from __future__ import annotations

from html import escape
from typing import Any


def format_hmm_process(process_name: str, process_params: dict[str, Any]) -> str:
    """Format HMM metadata for Plotly title subtitles."""
    if process_params:
        params = ", ".join(f"{key}={value}" for key, value in process_params.items())
    else:
        params = "no params"
    return f"Process: {escape(str(process_name))} | params: {escape(params)}"


def with_hmm_subtitle(
    title: str,
    process_name: str,
    process_params: dict[str, Any],
    details: str | None = None,
) -> str:
    subtitle_parts = [format_hmm_process(process_name, process_params)]
    if details:
        subtitle_parts.append(details)
    return f"{title}<br><sup>{' | '.join(subtitle_parts)}</sup>"
