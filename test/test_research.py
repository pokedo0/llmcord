import argparse
import asyncio
import os
import sys
import yaml

# 获取项目根目录，以便于正确加载 config 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from gemini_webapi import GeminiClient
from youtube_summary import get_gemini_cookies
from research_report import resolve_research_model, load_research_prompt

def read_config(filepath: str = "config/config.yaml") -> dict:
    full_path = os.path.join(PROJECT_ROOT, filepath)
    if not os.path.exists(full_path):
        print(f"配置文件不存在: {full_path}")
        return {}
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

async def main():
    parser = argparse.ArgumentParser(description="Test Gemini Research Report File Upload using Web API.")
    parser.add_argument("--file", default="test.pdf", help="File to upload (defaults to test.pdf in the current dir)")
    args = parser.parse_args()

    # 1. 动态读取配置
    config = read_config()
    research_cfg = config.get("research_report", {})
    
    if not research_cfg.get("enabled", True):
        print("警告: research_report 在配置中已被禁用 (enabled: false)")

    # 获取 cookies
    secure_1psid, secure_1psidts = get_gemini_cookies(config, research_cfg)
    if not secure_1psid:
        print("错误: 找不到 __Secure-1PSID cookie。请在 config.yaml 设置 gemini.secure_1psid")
        sys.exit(1)

    # 模型获取
    model = resolve_research_model(research_cfg)
    proxy = research_cfg.get("proxy")
    
    # 获取 Prompt
    prompt = load_research_prompt(research_cfg)

    # 目标文件路径
    test_file_path = os.path.join(os.path.dirname(__file__), args.file)
    if not os.path.exists(test_file_path):
        print(f"错误: 找不到测试文件 {test_file_path}")
        sys.exit(1)

    print("=========================================")
    print(f"测试文件: {test_file_path}")
    print(f"使用模型: {model}")
    print("=========================================\n")
    print("正在初始化 Gemini WebAPI 客户端并发送请求...")

    client = GeminiClient(secure_1psid=secure_1psid, secure_1psidts=secure_1psidts, proxy=proxy)
    try:
        await client.init(auto_refresh=True)

        print(f"[*] 正在通过 WebAPI 上传文件并生成总结 ...")
        resp = await client.generate_content(
            prompt=prompt,
            files=[test_file_path],
            model=model,
        )
        
        text = resp.text.strip() if getattr(resp, "text", None) else None
        if text:
            print("\n---------- 成功返回 ----------\n")
            print(text)
        else:
            print("\n!!! 模型返回了空内容 !!!")

    except Exception as e:
        print(f"\n请求失败: {e}")
    finally:
        try:
            await client.close()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())

