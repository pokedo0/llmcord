import asyncio
from pathlib import Path
import logging
import os
import tempfile
from typing import Any, Optional

import discord
import yaml

from gemini_webapi import GeminiClient, GeminiError
from gemini_webapi.constants import Model

from table_summary_img import DATA_START, DATA_END, extract_table_payload, generate_table_image_file
from youtube_summary import (
    split_text_for_embeds,
    _split_summary_and_table,
    read_config,
    load_prompt_file,
    get_gemini_cookies,
    resolve_gemini_model,
)

# ────────────────────────────
#  Constants
# ────────────────────────────

DEFAULT_RESEARCH_PROMPT_PATH = "config/research-prompt.txt"
DEFAULT_RESEARCH_PROMPT = (
    "详细总结附件内容。"
)

# Supported attachment MIME-type prefixes for research report monitoring
SUPPORTED_MIME_PREFIXES = (
    "application/pdf",
    "audio/",
    "image/",
    "text/",
    "application/msword",
    "application/vnd.openxmlformats-officedocument",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/epub",
    "application/json",
    "application/xml",
)

# In-memory mapping: channel_id -> optional prompt override (None = use default)
research_watch_channels: dict[int, Optional[str]] = {}

CONFIG_FILE = "config/config.yaml"


# ────────────────────────────
#  Config helpers
# ────────────────────────────

def load_research_prompt(research_config: dict[str, Any]) -> str:
    """Load the research report prompt from the configured file."""
    prompt_path = research_config.get("prompt_file", DEFAULT_RESEARCH_PROMPT_PATH)
    return load_prompt_file(prompt_path, DEFAULT_RESEARCH_PROMPT)


def resolve_research_model(research_config: dict[str, Any]) -> Model:
    """Resolve the Gemini model for research reports. Falls back to unspecified."""
    name = research_config.get("model") or "unspecified"
    try:
        return Model.from_name(name)
    except Exception:
        logging.warning("Invalid research Gemini model '%s', falling back to unspecified", name)
        return Model.from_name("unspecified")


def persist_research_channel(channel_id: int, enabled: bool) -> None:
    """Persist a channel in / out of research_report.watch_channels in config.yaml."""
    try:
        cfg = read_config()
        research_cfg = cfg.setdefault("research_report", {})
        watch_channels = set(research_cfg.get("watch_channels") or [])

        if enabled:
            watch_channels.add(channel_id)
        else:
            watch_channels.discard(channel_id)

        research_cfg["watch_channels"] = sorted(watch_channels)

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        logging.info("Persisted research_report.watch_channels=%s", research_cfg["watch_channels"])
    except Exception:
        logging.exception("Failed to persist research report watch channels to config.yaml")


# ────────────────────────────
#  Attachment detection
# ────────────────────────────

def _is_supported_attachment(att: discord.Attachment) -> bool:
    """Check if an attachment's MIME type is one we can process."""
    ct = att.content_type or ""
    return any(ct.startswith(prefix) for prefix in SUPPORTED_MIME_PREFIXES)


def _get_supported_attachments(msg: discord.Message) -> list[discord.Attachment]:
    """Return all attachments on a message that the research module can handle."""
    return [att for att in msg.attachments if _is_supported_attachment(att)]


# ────────────────────────────
#  Gemini interaction (gemini_webapi with cookies + file upload)
# ────────────────────────────

async def _summarize_with_gemini(
    prompt_text: str,
    attachment_paths: list[str],
    config: dict[str, Any],
) -> Optional[str]:
    """Upload files + prompt to Gemini via gemini_webapi and return the response text."""

    research_config = config.get("research_report", {})

    # Use shared Gemini cookies (same as youtube_summary)
    cookies = get_gemini_cookies(config, research_config)
    secure_1psid, secure_1psidts = cookies

    if not secure_1psid:
        logging.warning("Missing __Secure-1PSID cookie for research report Gemini webapi.")
        return None

    model = resolve_research_model(research_config)
    proxy = research_config.get("proxy")

    client = GeminiClient(secure_1psid=secure_1psid, secure_1psidts=secure_1psidts, proxy=proxy)

    try:
        await client.init(auto_refresh=True)

        logging.info(
            "Sending research prompt to Gemini (model=%s) with %d file(s): %s",
            model, len(attachment_paths), attachment_paths,
        )

        # gemini_webapi generate_content supports `files: list[str | Path]`
        resp = await client.generate_content(
            prompt=prompt_text,
            files=attachment_paths,
            model=model,
        )

        return resp.text.strip() if getattr(resp, "text", None) else None

    except GeminiError:
        logging.exception("Gemini webapi request failed for research report")
        return None
    except Exception:
        logging.exception("Unexpected error calling Gemini webapi for research report")
        return None
    finally:
        try:
            await client.close()
        except Exception:
            logging.exception("Failed to close Gemini client cleanly")


# ────────────────────────────
#  Core handler
# ────────────────────────────

async def maybe_handle_research_report(
    new_msg: discord.Message,
    config: dict[str, Any],
    is_good_user: bool,
    is_good_channel: bool,
    is_dm: bool,
) -> None:
    """
    Called from on_message. If the channel is being watched, and the message
    contains supported file attachments, download → summarise → post.
    """
    research_config = config.get("research_report", {})

    # Auto-enable channels persisted in config
    watch_channels = set(research_config.get("watch_channels", []) or [])
    if new_msg.channel.id in watch_channels and new_msg.channel.id not in research_watch_channels:
        research_watch_channels[new_msg.channel.id] = None
        logging.info("Auto-enabled research report monitoring for channel %s from config", new_msg.channel.id)

    # Gate checks
    if (
        not research_config.get("enabled", True)
        or is_dm
        or not is_good_user
        or not is_good_channel
        or new_msg.channel.id not in research_watch_channels
    ):
        return

    supported = _get_supported_attachments(new_msg)
    if not supported:
        return

    logging.info(
        "Research report triggered (channel=%s, user=%s, attachments=%d: %s)",
        new_msg.channel.id,
        new_msg.author.id,
        len(supported),
        ", ".join(att.filename for att in supported),
    )

    # Build prompt
    prompt_template = research_watch_channels.get(new_msg.channel.id) or load_research_prompt(research_config)

    # If the user added text alongside the file(s), prepend it
    user_text = new_msg.content.strip()
    if user_text:
        prompt_text = f"{user_text}\n\n{prompt_template}"
    else:
        prompt_text = prompt_template

    # Download attachments to temp dir
    tmp_dir = tempfile.mkdtemp(prefix="research_")
    downloaded_paths: list[str] = []

    try:
        for att in supported:
            dest = os.path.join(tmp_dir, att.filename)
            await att.save(dest)
            downloaded_paths.append(dest)
            logging.info("Downloaded attachment: %s (%s)", att.filename, att.content_type)

        # Summarise
        async with new_msg.channel.typing():
            summary = await _summarize_with_gemini(prompt_text, downloaded_paths, config)

    finally:
        # Clean up downloaded files
        for p in downloaded_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    if not summary:
        logging.warning("Research report summary failed or returned empty (channel=%s)", new_msg.channel.id)
        await new_msg.reply("研报解析失败，请稍后重试。", mention_author=False)
        return

    logging.info("Research report summary length=%d chars", len(summary))

    # Split text & table (same pattern as youtube_summary)
    clean_text, table_payload = _split_summary_and_table(summary)

    # Send text
    if clean_text:
        embed_batches = split_text_for_embeds(clean_text)
        for batch in embed_batches:
            embeds = [discord.Embed(description=desc, color=discord.Color.dark_teal()) for desc in batch]
            await new_msg.reply(embeds=embeds, mention_author=False)

    # Send table image
    if table_payload:
        png_path = None
        caption_parts = [att.filename for att in supported[:3]]
        caption = f"研报总结: {', '.join(caption_parts)}"
        try:
            png_path = await asyncio.to_thread(generate_table_image_file, table_payload, caption)
            if png_path:
                file = discord.File(png_path, filename=os.path.basename(png_path))
                await new_msg.reply(content="", file=file, mention_author=False)
        except Exception:
            logging.exception("Failed to render research table image")
        finally:
            try:
                if png_path:
                    os.remove(png_path)
            except Exception:
                pass
