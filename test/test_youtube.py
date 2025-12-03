"""
Quick manual tester for Gemini YouTube video understanding.

Run: python test_youtube.py --url https://youtu.be/<id>
"""
import argparse
import os
from typing import Any, Optional

from google import genai
from youtube_summary import (
    DEFAULT_YOUTUBE_MODEL,
    DEFAULT_YOUTUBE_PROMPT,
    DEFAULT_YOUTUBE_PROMPT_PATH,
    load_prompt_file,
    read_config,
)


def _extract_text(resp: Any) -> Optional[str]:
    if getattr(resp, "text", None):
        return resp.text.strip()
    if hasattr(resp, "candidates"):
        candidates = resp.candidates or []
        if candidates and getattr(candidates[0], "content", None):
            parts = getattr(candidates[0].content, "parts", None) or []
            texts = [getattr(p, "text", None) for p in parts if getattr(p, "text", None)]
            if texts:
                return "\n".join(texts).strip()
    if hasattr(resp, "to_dict"):
        data = resp.to_dict()
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
        texts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("text")]
        if texts:
            return "\n".join(texts).strip()
    return None


def summarize_youtube(
    url: str,
    api_key: str,
    model: str = DEFAULT_YOUTUBE_MODEL,
    prompt: str = DEFAULT_YOUTUBE_PROMPT,
) -> str:
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"file_data": {"file_uri": url}},
                    {"text": prompt},
                ],
            }
        ],
    )
    if text := _extract_text(resp):
        return text
    raise RuntimeError("Gemini returned no text response")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Gemini YouTube video understanding and summarization.")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--api-key", help="Gemini API key (or set providers.google.api_key / GOOGLE_API_KEY)")
    parser.add_argument("--model", help="Gemini model ID (defaults to config youtube_summary.model or models/gemini-2.5-flash)")
    parser.add_argument("--prompt", help="Custom prompt text (defaults to prompt_file or built-in)")
    args = parser.parse_args()

    cfg = read_config()
    yt_cfg = cfg.get("youtube_summary", {}) or {}

    api_key = (
        args.api_key
        or os.getenv("GOOGLE_API_KEY")
        or cfg.get("providers", {}).get("google", {}).get("api_key")
    )
    if not api_key:
        raise SystemExit("Missing API key. Provide --api-key, set providers.google.api_key, or env GOOGLE_API_KEY.")

    # 默认模型保持 2.5 flash，不从配置改写，除非显式传参
    model = args.model or DEFAULT_YOUTUBE_MODEL
    prompt_file = yt_cfg.get("prompt_file", DEFAULT_YOUTUBE_PROMPT_PATH)
    prompt = args.prompt or load_prompt_file(prompt_file, DEFAULT_YOUTUBE_PROMPT)

    try:
        summary = summarize_youtube(args.url, api_key, model, prompt)
    except Exception as exc:  # pragma: no cover - manual tester
        raise SystemExit(f"Failed to summarize: {exc}") from exc

    print(summary)


if __name__ == "__main__":
    main()
