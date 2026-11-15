# Windows：想用时开启，结束时关闭

这是**一个前台终端**的按需方式，不安装 Bot 服务、不设开机启动，也不需要打开 Codex。脚本读取当前 Tailscale 设备的 `*.ts.net` 名称，启动前台 Funnel，并把 `WEB_PUBLIC_ORIGIN`、`TELEGRAM_WEBHOOK_URL` 和 Mini App 开关只注入本次 Python 子进程；**不会改写 `.env`**。应用就绪后，可在 Telegram 私聊 Bot 并使用 `/workbench`。按 Ctrl+C 时，脚本等待应用关闭，核对并清除本次 Webhook（不丢弃待处理 Update），然后关闭本次 Funnel。

首次使用前仍需确认：Tailscale Windows 客户端已登录并允许 Funnel、Bot Token 与模型配置有效、专用 PostgreSQL 可用，以及 `web/dist` 已构建。公网模型地址必须直接配置 HTTPS，不能依赖 HTTP 跳转传递 API Key；本机或私有 IP 的测试 vLLM 可使用 HTTP。地址不用手填。加 `-UseLocalProductionCredential` 时，脚本会从以前的 Windows DPAPI 文件同时加载**专用数据库凭据和稳定应用密钥**，不打印它们，也不写回 `.env`；这要求由原 Windows 用户运行。不能读取凭据时，请先恢复凭据，不要改用隔离测试库。

在项目目录的 PowerShell 先运行只读检查：

```powershell
& .\ops\start_personal_assistant.ps1 -CheckOnly -UseLocalProductionCredential
```

检查只查询 Tailscale 状态、确认没有现存 Serve/Funnel 映射，并校验配置；不会启动 Bot、开放公网入口或写入数据库。若数据库凭据已通过安全的进程环境提供，省略 `-UseLocalProductionCredential`。如果 PowerShell 阻止脚本，仅在当前终端使用 `Set-ExecutionPolicy -Scope Process Bypass`，不要修改系统级策略。

正式使用：

```powershell
& .\ops\start_personal_assistant.ps1 -UseLocalProductionCredential
```

阅读提示并输入 `START` 后，脚本先确认 Bot 没有现存 Webhook，再启动前台 Funnel 和本机 `127.0.0.1:8000` 应用。启动可能恢复发送以前排队的 Telegram 消息；本启动方式**默认开启 Todo 提醒与每日简报**。启动时会检查已有到期提醒及当天的简报发送窗口，可能立即向数据库中已有的用户发送消息。每日简报逐用户读取自己的 Todo；目前只有配置的 Google Calendar owner 能读取该账号的日历，其他账号的简报会标明日历未连接。当前配置使用 `Asia/Shanghai` 的 `DAILY_BRIEF_HOUR=8`；如需调整时间，请在启动前检查 `.env`。看到 `Ready` 后，向自己的 Bot 发送一条安全测试消息，再试 `/workbench`。Google Calendar/Gmail 仍取决于原有 owner 与 OAuth 授权；不要用真实 Google 写操作作为第一条测试。

结束时在同一窗口按 **Ctrl+C 一次**，等待“Webhook/Funnel 清理已验证”的结果；关闭窗口或断电不能保证清理。异常退出后，可在确认端口 8000 已关闭的前提下运行 `& .\ops\start_personal_assistant.ps1 -CleanupOnly`，输入 `CLEANUP`；它只清除与**当前设备 HTTPS 地址完全匹配**的 Webhook 和指向本机 8000 的 Funnel，不会清除别的部署。也可以只读查看 `tailscale funnel status`。若状态仍异常，不要重复盲启。

停止服务不会删除 PostgreSQL 中的 Todo、会话或待发送消息。Tailscale Funnel 是公开 HTTPS 传输，不代替应用登录鉴权。脚本不会在无法识别的现有 Tailscale 映射上覆盖配置。更完整的外部服务与 Android 验收边界见仓库根目录的 `TELEGRAM_MINIAPP_G4_DEPLOYMENT_RUNBOOK.md`。
