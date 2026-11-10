"""
Integration test for Option 2: VPS Remote Direct Connection via GEMINI_BASE_URL.
This test spins up a mock OpenAI-compatible Web2API server on localhost, sets GEMINI_BASE_URL,
and tests that youtube_summary, finance_calendar, research_report, and get_gemini_client
all seamlessly connect, authenticate, and process requests properly.
"""
import asyncio
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from youtube_summary import get_gemini_client, summarize_youtube_video, read_config, maybe_handle_youtube_summary, youtube_watch_channels
from finance_calendar import generate_calendar_summary
from research_report import _summarize_with_gemini

MOCK_PORT = 18083
MOCK_API_KEY = "test_secret_key_123"

class MockWeb2APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence normal HTTP access logs
        pass

    def do_POST(self):
        auth_header = self.headers.get("Authorization", "")
        if f"Bearer {MOCK_API_KEY}" not in auth_header:
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "Unauthorized"}')
            return

        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            messages = data.get("messages", [])
            model = data.get("model", "")

            # Check message format
            user_msg = messages[0]["content"] if messages else ""
            resp_content = f"[Mock Summary Output] Processed model={model}, received input length={len(str(user_msg))}"

            response_payload = {
                "id": "chatcmpl-mock-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": resp_content,
                            "reasoning_content": "Thought process simulated."
                        },
                        "finish_reason": "stop"
                    }
                ]
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            auth_header = self.headers.get("Authorization", "")
            if f"Bearer {MOCK_API_KEY}" not in auth_header:
                self.send_response(401)
                self.end_headers()
                return

            models_payload = {
                "object": "list",
                "data": [
                    {"id": "gemini-2.5-flash", "object": "model"},
                    {"id": "gemini-2.5-flash-thinking", "object": "model"},
                    {"id": "gemini-2.5-pro", "object": "model"},
                ]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(models_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


async def run_tests():
    print("=" * 60)
    print("[START] 开始方案二测试：GEMINI_BASE_URL 远程环境变量直连")
    print("=" * 60)

    # 1. 启动本地 Mock Web2API 服务
    server = HTTPServer(("127.0.0.1", MOCK_PORT), MockWeb2APIHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[*] Mock Web2API 监听于 http://127.0.0.1:{MOCK_PORT}/v1")

    # 2. 设置环境变量（模拟 run_local.bat 中设置直连参数）
    test_base_url = f"http://127.0.0.1:{MOCK_PORT}/v1"
    os.environ["GEMINI_BASE_URL"] = test_base_url
    os.environ["GEMINI_API_KEY"] = MOCK_API_KEY

    cfg = read_config()

    # 测试 1: 客户端解析与动态环境变量注入
    print("\n[Test 1] 校验 get_gemini_client 环境变量优先级...")
    client, base_url = get_gemini_client(cfg)
    assert base_url == test_base_url, f"base_url 应为环境变量值 {test_base_url}，实际为 {base_url}"
    print(f"[PASS]: 成功识别 GEMINI_BASE_URL = {base_url}")

    # 测试 2: /v1/models 模型拉取
    print("\n[Test 2] 校验 /v1/models 接口获取模型列表...")
    models_resp = await client.models.list()
    model_ids = [m.id for m in models_resp.data]
    assert "gemini-2.5-flash" in model_ids, "未找到 gemini-2.5-flash"
    print(f"[PASS]: 成功获取远程模型列表: {model_ids}")

    # 测试 3: YouTube 总结调用
    print("\n[Test 3] 测试 YouTube 视频总结模块直连...")
    yt_summary = await summarize_youtube_video(
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        prompt_text="@Youtube 总结此视频 https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        youtube_config={"model": "gemini-2.5-flash"},
        config=cfg,
    )
    assert yt_summary is not None, "YouTube summary returned None"
    assert "[Mock Summary Output]" in yt_summary
    print(f"[PASS]: YouTube 视频总结成功返回: {yt_summary[:60]}...")

    # 测试 4: 财经日历调用
    print("\n[Test 4] 测试 财经日历生成模块直连...")
    cal_summary = await generate_calendar_summary()
    assert cal_summary is not None, "Calendar summary returned None"
    assert "[Mock Summary Output]" in cal_summary
    print(f"[PASS]: 财经日历成功返回: {cal_summary[:60]}...")

    # 测试 5: 研报文本附件直连
    print("\n[Test 5] 测试 研报分析模块直连 (纯文本附件)...")
    tmp_file = os.path.join(PROJECT_ROOT, "test", "temp_test_doc.txt")
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write("这是临时测试文档内容用于研报分析测试。")
    try:
        report_summary = await _summarize_with_gemini(
            prompt_text="总结附件",
            attachment_paths=[tmp_file],
            config=cfg,
        )
        assert report_summary is not None, "Research report summary returned None"
        print(f"[PASS]: 研报分析成功返回: {report_summary[:60]}...")
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    # 测试 6: 校验默认缺省 base_url 为 2048 以及 YouTube 默认开启解析机制
    print("\n[Test 6] 校验缺省 base_url 与 YouTube auto_watch_all 机制...")
    os.environ.pop("GEMINI_BASE_URL", None)
    os.environ.pop("GEMINI_API_KEY", None)
    default_client, default_base = get_gemini_client({})
    assert default_base == "http://127.0.0.1:2048/v1", f"缺省 base_url 应为 http://127.0.0.1:2048/v1，实际为 {default_base}"
    print(f"[PASS]: 缺省 base_url 正确回退到: {default_base}")

    class DummyChannel:
        id = 998877
        name = "test-channel"
        def typing(self):
            class DummyTyping:
                async def __aenter__(self): pass
                async def __aexit__(self, exc_type, exc, tb): pass
            return DummyTyping()

    class DummyAuthor:
        id = 112233
        name = "testuser"
        def __str__(self): return "testuser#0001"

    class DummyMsg:
        channel = DummyChannel()
        author = DummyAuthor()
        content = "check this out https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        embeds = []
        def __init__(self):
            self.replies = []
        async def reply(self, *args, **kwargs):
            self.replies.append((args, kwargs))

    os.environ["GEMINI_BASE_URL"] = test_base_url
    os.environ["GEMINI_API_KEY"] = MOCK_API_KEY
    dummy_msg = DummyMsg()
    test_cfg = {"youtube_summary": {"enabled": True, "auto_watch_all": True, "watch_channels": []}}
    await maybe_handle_youtube_summary(dummy_msg, test_cfg, is_good_user=True, is_good_channel=True, is_dm=False)
    assert len(dummy_msg.replies) > 0, "Default auto_watch_all 未能自动处理 YouTube 链接"
    print(f"[PASS]: YouTube 默认解析机制 (auto_watch_all=True) 自动触发成功 (收到 {len(dummy_msg.replies)} 条回复)")
    os.environ.pop("GEMINI_BASE_URL", None)
    os.environ.pop("GEMINI_API_KEY", None)

    # 4. 关闭 Mock 服务
    server.shutdown()
    print("\n" + "=" * 60)
    print("[SUCCESS] 全部集成测试通过！AIStudio2API 适配与默认 YouTube 解析均正常！")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_tests())
