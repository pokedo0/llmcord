# llmcord (Fork 版)

这是一个基于原版 `llmcord` 的定制 Fork，核心目标不是通用聊天，而是围绕 **YouTube 视频总结**、**财经日历推送**、**财报研读** 三个场景做自动化。

## 核心功能

- YouTube 总结：在指定频道监听 YouTube 链接，自动生成中文结构化总结（支持自动重试与表格渲染）。
- 财经日历：按订阅频道定时推送财经日历内容，并可手动触发测试。
- 财报研读：监听附件文件（PDF/文档等）并调用 Gemini 输出研读结果。
- 专用 Prompt：三类任务分别使用独立 prompt 文件，便于按业务持续迭代。

## 项目结构

```text
.
├─ llmcord.py                  # 主程序（Discord Bot）
├─ youtube_summary.py          # YouTube 总结逻辑
├─ finance_calendar.py         # 财经日历逻辑
├─ research_report.py          # 财报研读逻辑
├─ config/
│  ├─ config.yaml              # 运行配置（本地实际使用）
│  ├─ config-example.yaml      # 配置模板
│  ├─ youtube-summary-prompt.txt
│  ├─ calendar-prompt.txt
│  └─ research-prompt.txt
└─ gemini_cookies/             # gemini-webapi 本地 cookie/cache 目录
```

## 环境要求

- Python 3.10+
- 可用的 Discord Bot（`bot_token`、`client_id`）
- Gemini 网页可用账号（用于 `gemini-webapi`）

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

1. 复制配置模板并填写：

```bash
cp config/config-example.yaml config/config.yaml
```

Windows PowerShell：

```powershell
Copy-Item config/config-example.yaml config/config.yaml
```

2. 编辑 `config/config.yaml`，至少填写：

- `bot_token`
- `client_id`
- `permissions.users.admin_ids`
- `gemini.secure_1psid`
- `gemini.secure_1psidts`（建议）

3. 启动：

```bash
python llmcord.py
```

Windows 可直接使用：

```bat
run_local.bat
```

## Docker 运行（可选）

项目提供 `docker-compose.yaml`，会挂载：

- `./config -> /app/config`
- `./gemini_cookies -> /tmp/gemini_webapi`

启动：

```bash
docker compose up -d
```

## Discord 指令（Slash Commands）

以下命令默认仅管理员可用（受 `permissions.users.admin_ids` 控制）：

- `/ytwatch enabled:true|false`：开启/关闭当前频道的 YouTube 自动总结。
- `/ytprompt prompt:...`：更新 YouTube 总结 prompt 文件内容。
- `/ysession ...`：更新 Gemini cookie（写入 `config.yaml`）。
- `/calendar enabled:true|false`：订阅/取消财经日历定时推送。
- `/testcal`：立即测试一次财经日历推送。
- `/setresearch enabled:true|false`：开启/关闭当前频道财报研读监听。
- `/model model:...`：切换聊天模型。

## Prompt 说明

- YouTube 总结：`config/youtube-summary-prompt.txt`
- 财经日历：`config/calendar-prompt.txt`
- 财报研读：`config/research-prompt.txt`

建议把 Prompt 当成业务策略层：

- 需求变化优先改 Prompt，不要先改代码。
- Prompt 每次改动建议记录版本，便于回滚。

## 重点注意事项（必须看）

1. YouTube 工具首次可用前置条件

- 需要先在浏览器登录 Gemini 网页版（`gemini.google.com`），并同意“生成图像/相关使用协议”。
- 未完成这一步时，`@Youtube` 工具调用可能失败，表现为无法读取视频内容或反复重试。

2. 手动更新 `config.yaml` 的 Gemini Cookie 后，务必清理本地缓存

- 如果你手动改了 `gemini.secure_1psid` / `gemini.secure_1psidts`，建议清空 `gemini_cookies/` 目录内容。
- 原因：运行时可能优先读取目录中的旧缓存，导致新 cookie 不生效或账号错位。

## 常见问题

### 1) YouTube 总结返回“请稍后重试/<<RETRY>>”

通常是以下原因之一：

- 视频刚发布，字幕/画面后台数据尚未就绪。
- 视频是直播、会员专享、受限内容。
- 当前 Gemini 会话未完成必要条款确认。
- cookie 与当前登录账号不一致，或命中旧缓存。

排查顺序建议：

1. 检查 Gemini 网页是否可正常使用 YouTube 能力。
2. 核对 `config.yaml` 的 `gemini` cookie 是否最新。
3. 清空 `gemini_cookies/` 后重启。
4. 调整 `youtube_summary.post_delay_seconds` 与 `retry_delay_seconds`。

### 2) 财经日历没有按时发送

- 检查是否已通过 `/calendar enabled:true` 订阅频道。
- 检查机器人是否在线、频道权限是否允许发消息。
- 查看日志中 `finance_calendar` 相关错误。

### 3) 财报研读没有触发

- 检查频道是否已 `/setresearch enabled:true`。
- 检查附件格式是否受支持。
- 检查 Gemini cookie 是否可用。

## 配置建议

- `youtube_summary.post_delay_seconds`：建议 `30~120`，新视频更稳。
- `youtube_summary.retry_delay_seconds`：建议 `600`（10 分钟）起步。
- `youtube_summary.watch_channels` / `research_report.watch_channels` / `finance_calendar.channels`：可预置频道，重启后自动恢复监听。

## 免责声明

本项目是个人 Fork 的业务化改造版本，与原项目功能侧重点不同。请根据你的使用场景自行评估合规、稳定性与风险。
