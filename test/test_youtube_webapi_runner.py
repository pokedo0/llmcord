"""
Quick tester that summarizes a YouTube video using the community `gemini_webapi`
client (Gemini webapp cookies required).
"""
from __future__ import annotations

import argparse
import asyncio
import enum
import os
from typing import Optional

# python3.10 lacks enum.StrEnum which gemini_webapi expects
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        pass
    enum.StrEnum = StrEnum  # type: ignore[attr-defined]

SECURE_1PSID_DEFAULT = "g.a0004AgGYoeiTV0zOWYDC2g9a9-Ro9xiM54pBwLUb8ngW5Nt6G1qO3Tb_mGyFduE4tjrYwVM1QACgYKAU8SARUSFQHGX2Mirck9f1tbmAvq4PWOkGxZIxoVAUF8yKpEuli1L8vD_zKijkweNov30076"
SECURE_1PSIDTS_DEFAULT = "sidts-CjcBwQ9iI5peWWY8ayBRRuELq7S3o1_2ue3xfLl7pntk7a9rYQF2cXUtYnhIpZpu4Kio8O9oXNqEEAA"

from gemini_webapi import GeminiClient, GeminiError  # type: ignore  # imported after StrEnum shim
from gemini_webapi.constants import Model

from youtube_summary import (
    DEFAULT_YOUTUBE_PROMPT,
    DEFAULT_YOUTUBE_PROMPT_PATH,
    format_youtube_prompt,
    load_prompt_file,
    read_config,
)


def load_prompt_text(prompt_arg: Optional[str]) -> str:
    if prompt_arg:
        return prompt_arg

    cfg = read_config()
    yt_cfg = cfg.get("youtube_summary", {}) or {}
    prompt_file = yt_cfg.get("prompt_file", DEFAULT_YOUTUBE_PROMPT_PATH)
    return load_prompt_file(prompt_file, DEFAULT_YOUTUBE_PROMPT)


def resolve_model(model_arg: Optional[str]) -> Model:
    cfg = read_config()
    yt_cfg = cfg.get("youtube_summary", {}) or {}
    name = model_arg or yt_cfg.get("model") or Model.G_2_5_FLASH.model_name
    return Model.from_name(name)


def resolve_cookies(args: argparse.Namespace) -> tuple[str, Optional[str]]:
    secure_1psid = (
        args.secure_1psid
        or os.getenv("__Secure-1PSID")
        or os.getenv("SECURE_1PSID")
        or SECURE_1PSID_DEFAULT
    )
    secure_1psidts = (
        args.secure_1psidts
        or os.getenv("__Secure-1PSIDTS")
        or os.getenv("SECURE_1PSIDTS")
        or SECURE_1PSIDTS_DEFAULT
    )

    if not secure_1psid:
        raise SystemExit("Missing __Secure-1PSID cookie. Provide --secure-1psid or set env __Secure-1PSID.")

    return secure_1psid, secure_1psidts


async def summarize(url: str, prompt: str, model: Model, cookies: tuple[str, Optional[str]], proxy: Optional[str]) -> str:
    client = GeminiClient(secure_1psid=cookies[0], secure_1psidts=cookies[1], proxy=proxy)
    await client.init()
    try:
        output = await client.generate_content(prompt=prompt, model=model)
        return output.text
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a YouTube video with gemini_webapi (uses youtube_summary prompt from config.yaml by default)."
    )
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--prompt", help="Override prompt text")
    parser.add_argument("--model", help="Gemini model name (default: youtube_summary.model or gemini-2.5-flash)")
    parser.add_argument("--secure-1psid", dest="secure_1psid", help="__Secure-1PSID cookie value")
    parser.add_argument("--secure-1psidts", dest="secure_1psidts", help="__Secure-1PSIDTS cookie value (optional)")
    parser.add_argument("--channel", default="local-channel", help="Channel name placeholder for the prompt")
    parser.add_argument("--author", default="local-user", help="Author name placeholder for the prompt")
    parser.add_argument("--proxy", help="Optional proxy URL for gemini_webapi client")
    args = parser.parse_args()

    prompt_template = load_prompt_text(args.prompt)
    prompt = format_youtube_prompt(prompt_template, args.url, args.author, args.channel)
    # 强制启用 YouTube 扩展，且在末尾附带 URL。
    prompt = f"@Youtube {prompt} {args.url}"
    model = resolve_model(args.model)
    cookies = resolve_cookies(args)

    try:
        summary = asyncio.run(summarize(args.url, prompt, model, cookies, args.proxy))
    except GeminiError as exc:  # pragma: no cover - manual runner
        raise SystemExit(f"Gemini request failed: {exc}") from exc

    print(summary)


if __name__ == "__main__":
    main()
