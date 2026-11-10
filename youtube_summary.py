import asyncio
from pathlib import Path
import logging
import os
import re
from typing import Any, Optional
import discord

from openai import AsyncOpenAI
import yaml
from table_summary_img import DATA_START, DATA_END, extract_table_payload, generate_table_image_file

DEFAULT_YOUTUBE_MODEL = "unspecified"
DEFAULT_YOUTUBE_PROMPT_PATH = "config/youtube-summary-prompt.txt"
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
RETRY_MARKER_RE = re.compile(r"\\?<{1,2}RETRY\\?>{1,2}", re.IGNORECASE)

# Minimum length for a valid summary (shorter responses are likely failures)
MIN_SUMMARY_LENGTH = 50

def _is_failed_summary(text: str) -> bool:
    """Check if the summary text is too short (likely a failure response)."""
    if not text:
        return True
    text_stripped = text.strip()
    if len(text_stripped) < MIN_SUMMARY_LENGTH:
        logging.info("Summary too short (%d chars < %d), treating as failure", len(text_stripped), MIN_SUMMARY_LENGTH)
        return True
    return False

def read_config(filename: str = "config/config.yaml") -> dict[str, Any]:
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


def resolve_gemini_model(config: dict[str, Any], module_config: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Resolve unified Gemini model from gemini.model (or module fallback). Returns None for default/unspecified model."""
    gemini_cfg = config.get("gemini", {})
    name = gemini_cfg.get("model")
    if not name and module_config:
        name = module_config.get("model")
    name = str(name or DEFAULT_YOUTUBE_MODEL).strip()
    if not name or name.lower() in ("unspecified", "default", "none"):
        return None
    return name


def resolve_gemini_reasoning_effort(config: dict[str, Any], module_config: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Resolve unified reasoning_effort ('low', 'medium', 'high') from gemini or module config."""
    gemini_cfg = config.get("gemini", {})
    module_cfg = module_config or {}

    effort = gemini_cfg.get("reasoning_effort") or module_cfg.get("reasoning_effort")
    if effort:
        effort_str = str(effort).strip().lower()
        if effort_str in ("low", "medium", "high"):
            return effort_str

    # Backward compatibility: extended_thinking: bool
    if "extended_thinking" in gemini_cfg:
        return "high" if bool(gemini_cfg["extended_thinking"]) else "low"
    if "extended_thinking" in module_cfg:
        return "high" if bool(module_cfg["extended_thinking"]) else "low"

    return None


def resolve_gemini_extended_thinking(config: dict[str, Any], module_config: Optional[dict[str, Any]] = None) -> bool:
    """Legacy helper for backward compatibility: returns True if reasoning_effort is high."""
    effort = resolve_gemini_reasoning_effort(config, module_config)
    return effort == "high"


def get_gemini_client(config: dict[str, Any], module_config: Optional[dict[str, Any]] = None) -> tuple[AsyncOpenAI, str]:
    """Return an AsyncOpenAI client configured for AIStudio2API / Gemini Gateway and the base_url."""
    gemini_cfg = config.get("gemini", {})
    module_cfg = module_config or {}
    base_url = os.getenv("GEMINI_BASE_URL") or gemini_cfg.get("base_url") or module_cfg.get("base_url") or "http://127.0.0.1:2048/v1"
    api_key = os.getenv("GEMINI_API_KEY") or gemini_cfg.get("api_key") or module_cfg.get("api_key") or "none"
    timeout = float(gemini_cfg.get("timeout") or module_cfg.get("timeout") or 120.0)
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    return client, base_url


def get_gemini_cookies(config: dict[str, Any], youtube_config: Optional[dict[str, Any]] = None) -> tuple[Optional[str], Optional[str]]:
    """Legacy helper for backward compatibility: returns (base_url, api_key)."""
    gemini_cfg = config.get("gemini", {})
    module_cfg = youtube_config or {}
    base_url = os.getenv("GEMINI_BASE_URL") or gemini_cfg.get("base_url") or module_cfg.get("base_url") or "http://127.0.0.1:2048/v1"
    api_key = os.getenv("GEMINI_API_KEY") or gemini_cfg.get("api_key") or module_cfg.get("api_key") or ""
    return base_url, api_key


def persist_youtube_cookies(
    secure_1psid: Optional[str],
    secure_1psidts: Optional[str],
    clear_1psid: bool = False,
    clear_1psidts: bool = False,
) -> bool:
    try:
        cfg = read_config()
        # Update shared 'gemini' block
        gemini_cfg = cfg.setdefault("gemini", {}) or {}
        
        # Also check for legacy fields to potentially clear/migrate them? 
        # For now, we just write to the new location.
        # If users have old config, reading logic handles fallback.
        # But if they use /ysession, we enforce new structure.

        current_psid = gemini_cfg.get("secure_1psid", "") or ""
        current_psidts = gemini_cfg.get("secure_1psidts", "") or ""

        new_psid = "" if clear_1psid else (secure_1psid.strip() if secure_1psid is not None else current_psid)
        new_psidts = "" if clear_1psidts else (secure_1psidts.strip() if secure_1psidts is not None else current_psidts)

        gemini_cfg["secure_1psid"] = new_psid
        gemini_cfg["secure_1psidts"] = new_psidts
        cfg["gemini"] = gemini_cfg # Put it back to ensure it's set

        with open("config/config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        logging.info("Updated gemini cookies in config.yaml")
        return True
    except Exception:
        logging.exception("Failed to update gemini cookies")
        return False


def persist_gemini_model(
    model_name: Optional[str] = None,
    extended_thinking: Optional[bool] = None,
    reasoning_effort: Optional[str] = None,
) -> bool:
    """Persist unified model and/or reasoning_effort under gemini section in config.yaml."""
    try:
        cfg = read_config()
        gemini_cfg = cfg.setdefault("gemini", {}) or {}

        if model_name is not None:
            gemini_cfg["model"] = model_name.strip()
        if reasoning_effort is not None:
            gemini_cfg["reasoning_effort"] = reasoning_effort.strip().lower()
            gemini_cfg.pop("extended_thinking", None)
        elif extended_thinking is not None:
            gemini_cfg["reasoning_effort"] = "high" if extended_thinking else "low"
            gemini_cfg.pop("extended_thinking", None)

        cfg["gemini"] = gemini_cfg

        with open("config/config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        logging.info(
            "Updated unified gemini settings in config.yaml: model=%s, reasoning_effort=%s",
            model_name,
            gemini_cfg.get("reasoning_effort"),
        )
        return True
    except Exception:
        logging.exception("Failed to update gemini settings in config.yaml")
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
    if isinstance(resp, str):
        return resp.strip() or None
    if hasattr(resp, "choices") and resp.choices:
        msg = resp.choices[0].message
        return (msg.content or "").strip() or None
    if getattr(resp, "text", None):
        return resp.text.strip() or None
    return None


def _split_summary_and_table(summary: str) -> tuple[str, Optional[str]]:
    """Return (clean_text, table_payload_if_any). Removes placeholder block from text."""
    if not summary:
        return "", None

    start_idx = summary.find(DATA_START)
    end_idx = summary.find(DATA_END, start_idx + len(DATA_START)) if start_idx != -1 else -1
    if start_idx != -1 and end_idx != -1:
        table_block = summary[start_idx : end_idx + len(DATA_END)]
        table_payload = extract_table_payload(table_block)
        cleaned = (summary[:start_idx] + summary[end_idx + len(DATA_END) :]).strip()
        return cleaned, table_payload

    return summary, None


async def summarize_youtube_video(
    url: str,
    prompt_text: str,
    youtube_config: dict[str, Any],
    cookies: Optional[tuple[Optional[str], Optional[str]]] = None,
    proxy: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    cfg = config or read_config()
    raw_model = resolve_gemini_model(cfg, youtube_config)
    model = raw_model or "gemini-3.8-flash"
    reasoning_effort = resolve_gemini_reasoning_effort(cfg, youtube_config)

    client, base_url = get_gemini_client(cfg, youtube_config)
    logging.info(
        "Sending YouTube summary request to AIStudio2API (model=%s, reasoning_effort=%s, base_url=%s, url=%s)",
        model,
        reasoning_effort,
        base_url,
        url,
    )

    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        resp = await client.chat.completions.create(**kwargs)
        summary = _extract_summary(resp)
        if summary:
            logging.info("Successfully received YouTube summary (%d chars) for %s", len(summary), url)
        else:
            logging.warning("Received empty YouTube summary from AIStudio2API for %s", url)
        return summary
    except Exception:
        logging.exception("Error calling AIStudio2API for YouTube summary: %s", url)
        return None



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

        with open("config/config.yaml", "w", encoding="utf-8") as f:
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
    retry_count: int = 0,
) -> None:
    youtube_config = config.get("youtube_summary", {})

    watch_channels = set(youtube_config.get("watch_channels", []) or [])
    if new_msg.channel.id in watch_channels and new_msg.channel.id not in youtube_watch_channels:
        youtube_watch_channels[new_msg.channel.id] = None
        logging.info("Auto-enabled YouTube summaries for channel %s from config.watch_channels", new_msg.channel.id)

    is_watching = (new_msg.channel.id in youtube_watch_channels) and (youtube_watch_channels.get(new_msg.channel.id) is not False)

    if (
        not youtube_config.get("enabled", True)
        or is_dm
        or not is_good_user
        or not is_good_channel
        or not is_watching
    ):
        logging.debug(
            "YouTube summary skipped (enabled=%s, is_dm=%s, good_user=%s, good_channel=%s, watching=%s)",
            youtube_config.get("enabled", True),
            is_dm,
            is_good_user,
            is_good_channel,
            is_watching,
        )
        return

    if not (url := extract_first_youtube_url(new_msg.content)):
        logging.debug("Message in watched channel had no YouTube URL (channel_id=%s, content=%s)", new_msg.channel.id, new_msg.content)
        return

    cookies = get_gemini_cookies(config, youtube_config)
    if not cookies[0]:
        await new_msg.reply("AIStudio2API base_url is missing for YouTube summaries.", mention_author=False)
        return

    channel_prompt = youtube_watch_channels.get(new_msg.channel.id)
    prompt_template = channel_prompt if isinstance(channel_prompt, str) and channel_prompt else load_youtube_prompt(youtube_config)
    prompt_text = format_youtube_prompt(
        prompt_template, url, author_name=str(new_msg.author), channel_name=getattr(new_msg.channel, "name", "this channel")
    )
    prompt_text = f"@Youtube {prompt_text} {url}"

    logging.info("Summarizing YouTube video for channel %s (user ID: %s): %s", new_msg.channel.id, new_msg.author.id, url)

    post_delay = youtube_config.get("post_delay_seconds", 0)

    async with new_msg.channel.typing():
        if post_delay > 0:
            logging.info("Waiting %s seconds before summarizing (post_delay_seconds)", post_delay)
            await asyncio.sleep(post_delay)

        summary = await summarize_youtube_video(
            url, prompt_text, youtube_config, cookies, proxy=youtube_config.get("proxy"), config=config
        )

    if not summary:
        logging.warning("YouTube summary failed or returned empty text (channel_id=%s, user_id=%s)", new_msg.channel.id, new_msg.author.id)
        await new_msg.reply("Couldn't summarize that video right now.", mention_author=False)
        return

    logging.info("YouTube summary raw length=%s chars. Raw content: %s", len(summary), summary)

    summary = summary.replace(UNWANTED_SNIPPET, "")


    # 1. 优先处理重试逻辑 (显式 RETRY 标记 或 检测到失败响应)
    has_retry_marker = RETRY_MARKER_RE.search(summary)
    is_failed = _is_failed_summary(summary)
    
    if has_retry_marker or is_failed:
        # 提取并发送警告信息
        if has_retry_marker:
            clean_summary = RETRY_MARKER_RE.sub("", summary).strip()
            clean_summary = clean_summary.replace("\\<", "<").replace("\\>", ">")
        else:
            # 对于检测到的失败响应，生成一个友好的提示信息
            clean_summary = f"⚠️ 视频总结失败（AI 返回了无效响应），将在稍后自动重试。\n\n原始响应: {summary[:200]}{'...' if len(summary) > 200 else ''}"
        
        await new_msg.reply(content=clean_summary, mention_author=False)

        # 检查重试次数
        if retry_count < 1:  # 限制最大自动重试次数为 1 次
            youtube_config = config.get("youtube_summary", {})
            delay = youtube_config.get("retry_delay_seconds", 600)
            
            logging.info("Scheduling retry in %s seconds for channel %s (marker=%s, failed=%s)", 
                        delay, new_msg.channel.id, has_retry_marker, is_failed)
            
            async def _retry_task():
                await asyncio.sleep(delay)
                # 重新加载配置以获取最新的 Cookie（如果已更新）
                try:
                    new_cfg = read_config()
                except Exception:
                    new_cfg = config
                
                logging.info("Executing retry for channel %s", new_msg.channel.id)
                await maybe_handle_youtube_summary(new_msg, new_cfg, is_good_user, is_good_channel, is_dm, retry_count + 1)

            asyncio.create_task(_retry_task())
        return

    # 2. 处理正常总结逻辑
    clean_text, table_payload = _split_summary_and_table(summary)

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

    if table_payload:
        caption = next((embed.title for embed in new_msg.embeds if getattr(embed, "title", None)), None)
        caption = f"总结: {caption}" if caption else None
        try:
            png_path = await asyncio.to_thread(generate_table_image_file, table_payload, caption)
            if png_path is None:
                return
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
            if RETRY_MARKER_RE.search(summary):
                print("\n--- [RETRY DETECTED] ---")
                print(RETRY_MARKER_RE.sub("", summary).strip())
                print("------------------------\n")
            else:
                print(summary)
        else:
            print("No summary returned.")

    asyncio.run(_run())
