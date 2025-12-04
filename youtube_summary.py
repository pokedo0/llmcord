import asyncio
from pathlib import Path
import logging
import os
import re
from typing import Any, Optional
import enum
import discord

# python 3.10 兼容 gemini_webapi 的 StrEnum 依赖
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        pass
    enum.StrEnum = StrEnum  # type: ignore[attr-defined]

from gemini_webapi import GeminiClient, GeminiError  # type: ignore
from gemini_webapi.constants import Model
import yaml
from csv_summary_img import DATA_START, DATA_END, extract_csv_payload, generate_table_image_file

DEFAULT_YOUTUBE_MODEL = "gemini-2.5-flash"
DEFAULT_YOUTUBE_PROMPT_PATH = "youtube-summary-prompt.txt"
DEFAULT_YOUTUBE_PROMPT = (
    "Summarize the YouTube video at {url}. Give the core topic, 5 concise bullet points "
    "with timestamps if available, and a short takeaway tailored to this channel."
)
YOUTUBE_URL_RE = re.compile(
    r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]{11}(?:\S+)?|youtu\.be/[\w-]{11}(?:\S+)?))",
    re.IGNORECASE,
)
EMBED_DESC_LIMIT = 4096
EMBED_TOTAL_PER_MESSAGE = 6000
EMBED_MAX_PER_MESSAGE = 10

youtube_watch_channels: dict[int, Optional[str]] = {}
UNWANTED_SNIPPET = "http://googleusercontent.com/youtube_content/0"


def read_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_prompt_file(path: str, fallback: str = "") -> str:
    try:
        content = Path(path).read_text(encoding="utf-8").strip()
        if content:
            return content
    except FileNotFoundError:
        logging.warning("Prompt file not found: %s", path)
    except OSError:
        logging.exception("Error reading prompt file: %s", path)
    return fallback


def write_prompt_file(path: str, content: str) -> bool:
    try:
        Path(path).write_text(content.strip() + "\n", encoding="utf-8")
        logging.info("Wrote YouTube prompt file: %s", path)
        return True
    except Exception:
        logging.exception("Failed to write YouTube prompt file: %s", path)
        return False


def load_youtube_prompt(youtube_config: dict[str, Any]) -> str:
    prompt_path = youtube_config.get("prompt_file", DEFAULT_YOUTUBE_PROMPT_PATH)
    return load_prompt_file(prompt_path, DEFAULT_YOUTUBE_PROMPT)


def extract_first_youtube_url(text: str) -> Optional[str]:
    if match := YOUTUBE_URL_RE.search(text or ""):
        return match.group(1)
    return None


def format_youtube_prompt(template: str, url: str, author_name: str, channel_name: str) -> str:
    prompt = template
    for key, value in {
        "{url}": url,
        "{author}": author_name,
        "{channel}": channel_name,
    }.items():
        prompt = prompt.replace(key, value)
    return prompt


def resolve_gemini_model(youtube_config: dict[str, Any]) -> Model:
    name = youtube_config.get("model") or DEFAULT_YOUTUBE_MODEL
    try:
        return Model.from_name(name)
    except Exception:
        logging.warning("Invalid Gemini model '%s', falling back to %s", name, DEFAULT_YOUTUBE_MODEL)
        return Model.from_name(DEFAULT_YOUTUBE_MODEL)


def get_gemini_cookies(config: dict[str, Any], youtube_config: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    secure_1psid = youtube_config.get("secure_1psid") or os.getenv("__Secure-1PSID") or os.getenv("SECURE_1PSID")
    secure_1psidts = youtube_config.get("secure_1psidts") or os.getenv("__Secure-1PSIDTS") or os.getenv("SECURE_1PSIDTS")
    if not secure_1psid:
        return None, None
    return secure_1psid, secure_1psidts


def persist_youtube_cookies(
    secure_1psid: Optional[str],
    secure_1psidts: Optional[str],
    clear_1psid: bool = False,
    clear_1psidts: bool = False,
) -> bool:
    try:
        cfg = read_config()
        yt_cfg = cfg.setdefault("youtube_summary", {}) or {}

        current_psid = yt_cfg.get("secure_1psid", "") or ""
        current_psidts = yt_cfg.get("secure_1psidts", "") or ""

        new_psid = "" if clear_1psid else (secure_1psid.strip() if secure_1psid is not None else current_psid)
        new_psidts = "" if clear_1psidts else (secure_1psidts.strip() if secure_1psidts is not None else current_psidts)

        yt_cfg["secure_1psid"] = new_psid
        yt_cfg["secure_1psidts"] = new_psidts

        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        logging.info("Updated youtube_summary cookies in config.yaml")
        return True
    except Exception:
        logging.exception("Failed to update youtube_summary cookies")
        return False


def split_text_for_embeds(
    text: str,
    max_embed_len: int = EMBED_DESC_LIMIT,
    max_total_per_message: int = EMBED_TOTAL_PER_MESSAGE,
    max_embeds_per_message: int = EMBED_MAX_PER_MESSAGE,
) -> list[list[str]]:
    """Split text into embed descriptions while packing paragraphs together to reduce embed count."""
    if not text:
        return []

    embed_chunks: list[str] = []
    current_desc = ""

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) > max_embed_len:
            if current_desc:
                embed_chunks.append(current_desc)
                current_desc = ""
            for start in range(0, len(paragraph), max_embed_len):
                embed_chunks.append(paragraph[start : start + max_embed_len])
            continue

        if not current_desc:
            current_desc = paragraph
        elif len(current_desc) + 2 + len(paragraph) <= max_embed_len:
            current_desc += "\n\n" + paragraph
        else:
            embed_chunks.append(current_desc)
            current_desc = paragraph

    if current_desc:
        embed_chunks.append(current_desc)

    if not embed_chunks:
        for start in range(0, len(text), max_embed_len):
            embed_chunks.append(text[start : start + max_embed_len])

    messages: list[list[str]] = []
    current_embeds: list[str] = []
    current_total = 0

    for chunk in embed_chunks:
        starts_new_message = (
            not current_embeds
            or len(current_embeds) >= max_embeds_per_message
            or current_total + len(chunk) > max_total_per_message
        )

        if starts_new_message and current_embeds:
            messages.append(current_embeds)
            current_embeds = []
            current_total = 0

        current_embeds.append(chunk)
        current_total += len(chunk)

    if current_embeds:
        messages.append(current_embeds)

    return messages


def _extract_summary(resp: Any) -> Optional[str]:
    if getattr(resp, "text", None):
        return resp.text.strip() or None
    return None


def _split_summary_and_csv(summary: str) -> tuple[str, Optional[str]]:
    """Return (clean_text, csv_payload_if_any). Removes placeholder block from text."""
    if not summary:
        return "", None

    start_idx = summary.find(DATA_START)
    end_idx = summary.find(DATA_END, start_idx + len(DATA_START)) if start_idx != -1 else -1
    if start_idx != -1 and end_idx != -1:
        csv_block = summary[start_idx : end_idx + len(DATA_END)]
        csv_payload = extract_csv_payload(csv_block)
        cleaned = (summary[:start_idx] + summary[end_idx + len(DATA_END) :]).strip()
        return cleaned, csv_payload

    return summary, None


async def summarize_youtube_video(
    url: str,
    prompt_text: str,
    youtube_config: dict[str, Any],
    cookies: tuple[Optional[str], Optional[str]],
    proxy: Optional[str] = None,
) -> Optional[str]:
    secure_1psid, secure_1psidts = cookies
    if not secure_1psid:
        logging.warning("Missing __Secure-1PSID cookie for Gemini webapi.")
        return None

    model = resolve_gemini_model(youtube_config)
    client = GeminiClient(secure_1psid=secure_1psid, secure_1psidts=secure_1psidts, proxy=proxy)

    try:
        await client.init()
        resp = await client.generate_content(prompt=prompt_text, model=model)
    except GeminiError:
        logging.exception("Gemini webapi request failed")
        return None
    except Exception:
        logging.exception("Unexpected error calling Gemini webapi")
        return None
    finally:
        try:
            await client.close()
        except Exception:
            logging.exception("Failed to close Gemini client cleanly")

    return _extract_summary(resp)


def persist_watch_channel(channel_id: int, enabled: bool) -> None:
    try:
        cfg = read_config()
        youtube_cfg = cfg.setdefault("youtube_summary", {})
        watch_channels = set(youtube_cfg.get("watch_channels") or [])

        if enabled:
            watch_channels.add(channel_id)
        else:
            watch_channels.discard(channel_id)

        youtube_cfg["watch_channels"] = sorted(watch_channels)

        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        logging.info("Persisted youtube_summary.watch_channels=%s", youtube_cfg["watch_channels"])
    except Exception:
        logging.exception("Failed to persist YouTube watch channels to config.yaml")


async def maybe_handle_youtube_summary(
    new_msg,  # discord.Message (kept generic to avoid import cycle)
    config: dict[str, Any],
    is_good_user: bool,
    is_good_channel: bool,
    is_dm: bool,
) -> None:
    youtube_config = config.get("youtube_summary", {})

    watch_channels = set(youtube_config.get("watch_channels", []) or [])
    if new_msg.channel.id in watch_channels and new_msg.channel.id not in youtube_watch_channels:
        youtube_watch_channels[new_msg.channel.id] = None
        logging.info("Auto-enabled YouTube summaries for channel %s from config", new_msg.channel.id)

    if (
        not youtube_config.get("enabled", True)
        or is_dm
        or not is_good_user
        or not is_good_channel
        or new_msg.channel.id not in youtube_watch_channels
    ):
        logging.debug(
            "YouTube summary skipped (enabled=%s, is_dm=%s, good_user=%s, good_channel=%s, watching=%s)",
            youtube_config.get("enabled", True),
            is_dm,
            is_good_user,
            is_good_channel,
            new_msg.channel.id in youtube_watch_channels,
        )
        return

    if not (url := extract_first_youtube_url(new_msg.content)):
        logging.debug("Message in watched channel had no YouTube URL (channel_id=%s, content=%s)", new_msg.channel.id, new_msg.content)
        return

    cookies = get_gemini_cookies(config, youtube_config)
    if not cookies[0]:
        await new_msg.reply("Gemini cookies (__Secure-1PSID/TS) are missing for YouTube summaries.", mention_author=False)
        return

    prompt_template = youtube_watch_channels.get(new_msg.channel.id) or load_youtube_prompt(youtube_config)
    prompt_text = format_youtube_prompt(
        prompt_template, url, author_name=str(new_msg.author), channel_name=getattr(new_msg.channel, "name", "this channel")
    )
    prompt_text = f"@Youtube {prompt_text} {url}"

    logging.info("Summarizing YouTube video for channel %s (user ID: %s): %s", new_msg.channel.id, new_msg.author.id, url)

    async with new_msg.channel.typing():
        summary = await summarize_youtube_video(
            url, prompt_text, youtube_config, cookies, proxy=youtube_config.get("proxy")
        )

    if not summary:
        logging.warning("YouTube summary failed or returned empty text (channel_id=%s, user_id=%s)", new_msg.channel.id, new_msg.author.id)
        await new_msg.reply("Couldn't summarize that video right now.", mention_author=False)
        return

    logging.info("YouTube summary raw length=%s chars. Raw content: %s", len(summary), summary)

    summary = summary.replace(UNWANTED_SNIPPET, "")

    clean_text, csv_payload = _split_summary_and_csv(summary)

    if clean_text:
        logging.info("YouTube summary success (channel_id=%s, user_id=%s)", new_msg.channel.id, new_msg.author.id)
        embed_batches = split_text_for_embeds(clean_text)
        first_embed_with_url = True

        for batch in embed_batches:
            embeds: list[discord.Embed] = []
            for desc in batch:
                embed_kwargs: dict[str, Any] = {"description": desc, "color": discord.Color.blurple()}
                if first_embed_with_url:
                    embed_kwargs["url"] = url
                    first_embed_with_url = False
                embeds.append(discord.Embed(**embed_kwargs))

            await new_msg.reply(embeds=embeds, mention_author=False)

    if csv_payload:
        try:
            png_path = await asyncio.to_thread(generate_table_image_file, csv_payload)
            file = discord.File(png_path, filename=os.path.basename(png_path))
            await new_msg.reply(content="", file=file, mention_author=False)
        finally:
            try:
                os.remove(png_path)
            except Exception:
                pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quickly test Gemini YouTube summarization locally.")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--prompt-file", help="Prompt file path", default=DEFAULT_YOUTUBE_PROMPT_PATH)
    parser.add_argument("--model", help="Gemini model name", default=DEFAULT_YOUTUBE_MODEL)
    parser.add_argument("--proxy", help="Optional proxy URL for gemini_webapi client")
    args = parser.parse_args()

    cfg = read_config()
    yt_cfg = cfg.get("youtube_summary", {}) or {}
    yt_cfg = {"model": args.model, "prompt_file": args.prompt_file, "proxy": args.proxy} | yt_cfg

    cookies = get_gemini_cookies(cfg, yt_cfg)
    if not cookies[0]:
        raise SystemExit("Missing __Secure-1PSID cookie. Please set it in config.yaml or environment variables.")

    prompt = load_prompt_file(args.prompt_file, DEFAULT_YOUTUBE_PROMPT)
    prompt = format_youtube_prompt(prompt, args.url, author_name="local-user", channel_name="local-channel")
    prompt = f"@Youtube {prompt} {args.url}"

    async def _run() -> None:
        summary = await summarize_youtube_video(args.url, prompt, yt_cfg, cookies, proxy=yt_cfg.get("proxy"))
        if summary:
            print(summary)
        else:
            print("No summary returned.")

    asyncio.run(_run())
