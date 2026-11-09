import argparse
import asyncio
import os
import sys
import yaml

# 获取项目根目录，以便于正确加载 config 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research_report import (
    resolve_research_model,
    load_research_prompt,
    _summarize_with_gemini,
)
from youtube_summary import resolve_gemini_extended_thinking, get_gemini_client

def read_config(filepath: str = "config/config.yaml") -> dict:
    full_path = os.path.join(PROJECT_ROOT, filepath)
    if not os.path.exists(full_path):
        print(f"配置文件不存在: {full_path}")
        return {}
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

async def main():
    parser = argparse.ArgumentParser(description="Test Gemini Research Report File Upload using Web2API.")
    parser.add_argument("--file", default="test.pdf", help="File to upload (defaults to test.pdf in the current dir)")
    args = parser.parse_args()

    # 1. 动态读取配置
    config = read_config()
    research_cfg = config.get("research_report", {})
    
    if not research_cfg.get("enabled", True):
        print("警告: research_report 在配置中已被禁用 (enabled: false)")

    # 模型获取
    model = resolve_research_model(config, research_cfg) or "gemini-2.5-flash"
    extended_thinking = resolve_gemini_extended_thinking(config, research_cfg)
    
    # 获取 Prompt
    prompt = load_research_prompt(research_cfg)

    # 目标文件路径
    test_file_path = os.path.join(os.path.dirname(__file__), args.file)
    if not os.path.exists(test_file_path):
        print(f"错误: 找不到测试文件 {test_file_path}")
        sys.exit(1)

    _, base_url = get_gemini_client(config, research_cfg)
    print("=========================================")
    print(f"测试文件: {test_file_path}")
    print(f"Web2API 地址: {base_url}")
    print(f"使用模型: {model}")
    print(f"扩展思考: {extended_thinking}")
    print("=========================================\n")
    print("正在通过 Gemini Web2API 发送请求...")

    try:
        text = await _summarize_with_gemini(prompt, [test_file_path], config)
        if text:
            print("\n---------- 成功返回 ----------\n")
            print(text)
        else:
            print("\n!!! 模型返回了空内容 !!!")
    except Exception as e:
        print(f"\n请求失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
