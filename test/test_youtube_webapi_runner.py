"""
Quick tester that summarizes a YouTube video using Gemini Web2API (OpenAI-compatible endpoint).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from youtube_summary import (
    DEFAULT_YOUTUBE_PROMPT,
    DEFAULT_YOUTUBE_PROMPT_PATH,
    format_youtube_prompt,
    load_prompt_file,
    read_config,
    resolve_gemini_model,
    resolve_gemini_extended_thinking,
    get_gemini_client,
    _extract_summary,
)


def load_prompt_text(prompt_arg: Optional[str]) -> str:
    if prompt_arg:
        return prompt_arg

    cfg = read_config()
    yt_cfg = cfg.get("youtube_summary", {}) or {}
    prompt_file = yt_cfg.get("prompt_file", DEFAULT_YOUTUBE_PROMPT_PATH)
    return load_prompt_file(prompt_file, DEFAULT_YOUTUBE_PROMPT)


def resolve_model(model_arg: Optional[str]) -> Optional[str]:
    if model_arg:
        name = str(model_arg).strip()
        if not name or name.lower() in ("unspecified", "default", "none"):
            return None
        return name

    cfg = read_config()
    return resolve_gemini_model(cfg)


async def summarize(
    url: str,
    prompt: str,
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    extended_thinking: bool = False,
) -> str:
    cfg = read_config()
    if base_url:
        cfg.setdefault("gemini", {})["base_url"] = base_url
    if api_key:
        cfg.setdefault("gemini", {})["api_key"] = api_key

    client, resolved_base = get_gemini_client(cfg)
    target_model = model or "gemini-2.5-flash"
    if extended_thinking and not target_model.endswith("-thinking"):
        target_model = f"{target_model}-thinking"

    print(f"[*] Calling Gemini Web2API at {resolved_base} with model={target_model}...")
    resp = await client.chat.completions.create(
        model=target_model,
        messages=[{"role": "user", "content": prompt}],
    )
    summary = _extract_summary(resp)
    if not summary:
        raise RuntimeError("Empty response received from Web2API")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a YouTube video with Gemini Web2API (OpenAI-compatible)."
    )
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--prompt", help="Override prompt text")
    parser.add_argument("--model", help="Gemini model name (default: from config.yaml)")
    parser.add_argument("--extended-thinking", action="store_true", help="Enable extended thinking mode")
    parser.add_argument("--base-url", help="Base URL of Gemini Web2API service (e.g. http://127.0.0.1:8083/v1)")
    parser.add_argument("--api-key", help="API key for Web2API")
    parser.add_argument("--channel", default="local-channel", help="Channel name placeholder for the prompt")
    parser.add_argument("--author", default="local-user", help="Author name placeholder for the prompt")
    args = parser.parse_args()

    prompt_template = load_prompt_text(args.prompt)
    prompt = format_youtube_prompt(prompt_template, args.url, args.author, args.channel)
    prompt = f"@Youtube {prompt} {args.url}"
    model = resolve_model(args.model)

    cfg = read_config()
    extended_thinking = args.extended_thinking or resolve_gemini_extended_thinking(cfg)

    try:
        summary = asyncio.run(summarize(args.url, prompt, model, args.base_url, args.api_key, extended_thinking))
    except Exception as exc:
        raise SystemExit(f"Gemini request failed: {exc}") from exc

    print(summary)


if __name__ == "__main__":
    main()
