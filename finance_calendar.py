import asyncio
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import discord
import yaml
from gemini_webapi import GeminiClient, GeminiError
from gemini_webapi.constants import Model

# 复用 table_summary_img 的常量和函数
from table_summary_img import DATA_START, DATA_END, extract_table_payload, generate_table_image_file
# 复用 youtube_summary 的部分逻辑（如切分文本），如果无法直接导入则复制一份简化版
from youtube_summary import split_text_for_embeds, _split_summary_and_table

# 常量定义
PROMPT_FILE = "config/calendar-prompt.txt"
TABLE_CAPTION = "下两周财经日历总结"
CONFIG_FILE = "config/config.yaml"

# 全局变量记录已发送日期，避免重复发送 (简单内存记录，重启失效但配合整点判断也够用了)
last_sent_date_str = None

def read_config(filename: str = CONFIG_FILE) -> dict[str, Any]:
    try:
        with open(filename, encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        return {}

def save_calendar_channel(channel_id: int, remove: bool = False) -> bool:
    """保存或移除订阅了日历的频道ID到配置文件的 finance_calendar.channels"""
    try:
        cfg = read_config()
        fc_cfg = cfg.setdefault("finance_calendar", {})
        channels = set(fc_cfg.get("channels") or [])
        
        if remove:
            channels.discard(channel_id)
        else:
            channels.add(channel_id)
            
        fc_cfg["channels"] = sorted(list(channels))
        
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        return True
    except Exception:
        logging.exception("Failed to update finance_calendar channels in config")
        return False

def get_calendar_config() -> dict[str, Any]:
    cfg = read_config()
    return cfg.get("finance_calendar", {})

def get_gemini_cookies_from_config() -> tuple[Optional[str], Optional[str]]:
    """
    优先读取 gemini 下的 cookies (shared)，
    如果没有，尝试读取 legacy 的 finance_calendar / youtube_summary 配置
    """
    cfg = read_config()
    
    # 1. 尝试 Shared gemini block (New Standard)
    gemini_cfg = cfg.get("gemini", {})
    p1 = gemini_cfg.get("secure_1psid")
    p2 = gemini_cfg.get("secure_1psidts")
    if p1: return p1, p2
    
    # 4. 环境变量
    return os.getenv("SECURE_1PSID"), os.getenv("SECURE_1PSIDTS")

def load_prompt() -> str:
    try:
        return Path(PROMPT_FILE).read_text(encoding="utf-8").strip()
    except Exception:
        logging.error(f"Failed to read prompt file: {PROMPT_FILE}")
        return "Please summarize the financial calendar for the next two weeks."

async def generate_calendar_summary() -> Optional[str]:
    """调用 Gemini 生成内容"""
    cookies = get_gemini_cookies_from_config()
    secure_1psid, secure_1psidts = cookies
    
    if not secure_1psid:
        logging.warning("Missing Gemini cookies for finance calendar.")
        return None

    prompt_text = load_prompt()
    
    # 可以在这里替换一些变量，比如当前时间
    now_str = datetime.now().strftime("%Y-%m-%d")
    prompt_text = f"Current Date: {now_str}\n\n{prompt_text}"

    client = GeminiClient(secure_1psid=secure_1psid, secure_1psidts=secure_1psidts)
    
    try:
        await client.init(auto_refresh=True)
        # 使用默认模型或配置模型，这里暂定不指定，让库使用默认
        resp = await client.generate_content(prompt=prompt_text)
        return resp.text
    except Exception:
        logging.exception("Error generating calendar content with Gemini")
        return None
    finally:
        try:
            await client.close()
        except:
            pass

async def send_calendar_to_channel(channel: discord.abc.Messageable, summary: str):
    """处理并发送内容到指定频道"""
    # 1. 分离表格
    clean_text, table_payload = _split_summary_and_table(summary)
    
    # 2. 发送文字总结
    if clean_text:
        embed_batches = split_text_for_embeds(clean_text)
        for batch in embed_batches:
            embeds = []
            for desc in batch:
                embeds.append(discord.Embed(description=desc, color=discord.Color.gold()))
            if embeds:
                await channel.send(embeds=embeds)
    
    # 3. 发送表格图片
    if table_payload:
        try:
            png_path = await asyncio.to_thread(
                generate_table_image_file, 
                raw_table=f"{DATA_START}{table_payload}{DATA_END}", 
                caption=TABLE_CAPTION
            )
            if png_path:
                file = discord.File(png_path, filename="calendar_summary.png")
                await channel.send(file=file)
                try:
                    os.remove(png_path)
                except:
                    pass
        except Exception:
            logging.exception("Failed to render calendar table")
            await channel.send("表格生成失败，请查看日志。")

async def run_calendar_task(bot: discord.Client, force_channel_id: Optional[int] = None):
    """
    运行任务。
    如果指定 force_channel_id，则只向该频道发送（用于测试）。
    否则，读取配置向所有订阅频道发送。
    """
    logging.info("Starting run_calendar_task...")
    summary = await generate_calendar_summary()
    
    if not summary:
        logging.error("Failed to get summary from Gemini.")
        if force_channel_id:
            channel = bot.get_channel(force_channel_id)
            if channel:
                await channel.send("获取财经日历失败 (Gemini Error)。")
        return

    # 目标频道列表
    target_channel_ids = []
    if force_channel_id:
        target_channel_ids.append(force_channel_id)
    else:
        cfg = get_calendar_config()
        target_channel_ids = cfg.get("channels") or []

    if not target_channel_ids:
        logging.info("No channels subscribed to finance calendar.")
        return

    for cid in target_channel_ids:
        channel = bot.get_channel(cid)
        if not channel:
            # 可能是 fetch_channel 需要 await，但在 bot cache 里应该有 get_channel
            try:
                channel = await bot.fetch_channel(cid)
            except:
                logging.warning(f"Could not fetch channel {cid}")
                continue
        
        if channel:
            logging.info(f"Sending calendar to channel {cid}")
            await send_calendar_to_channel(channel, summary)

async def check_and_run_schedule(bot: discord.Client):
    """
    定时检查函数，需在主 loop 中运行。
    规则：每周日 (weekday=6) 8:00 执行。
    """
    global last_sent_date_str
    now = datetime.now()
    
    # 周日=6, 8点
    if now.weekday() == 5 and now.hour == 8:
        # 简单防重：如果今天已经发过了，就不发
        today_str = now.strftime("%Y-%m-%d")
        if last_sent_date_str == today_str:
            return
            
        logging.info("Triggering scheduled finance calendar task.")
        await run_calendar_task(bot)
        last_sent_date_str = today_str
