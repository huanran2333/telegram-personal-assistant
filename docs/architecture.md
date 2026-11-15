# 系统架构与可靠性边界

## 入站消息和 Agent

Telegram 将消息或按钮点击作为 Update 发送给 FastAPI Webhook。后端先用 Bot 与 Telegram Update 的身份保存 Receipt，重复投递命中同一记录，再返回 HTTP 成功。HTTP 成功只证明接纳；真正的业务处理由后台 Worker 领取。Worker 使用同一用户会话的租约和 fence，防止过期处理者继续提交结果。失败的 Receipt 可按状态和重试策略再次领取，不能从“重试”推断整轮业务未曾发生。

Worker 按 Update 类型处理普通文字与 Telegram 审批按钮。普通文字进入统一 `create_agent` 图；按钮关联原会话和待审操作，恢复对应中断。LangGraph 的 PostgreSQL Checkpoint 保存会话图状态，包含需要继续处理的消息和中断。一个工具的结果可能已持久化为 pending write，而主 Checkpoint 尚未推进；这不代表工具未执行，也不代表整轮完成。

## 工具写入和远端边界

本地 Todo 写入与 ActionExecution 业务执行记录在同一 PostgreSQL 事务中提交，用于重试时识别已经发生的写入。写入还检查可信用户身份、目标归属及适用的版本或时间依据。Web 直接写入通过 WebMutation 记录提交标识；版本检查处理并发覆盖。这两种入口的重复提交标识不同，但共享同一业务数据与版本约束。

Google Calendar 是远端系统，无法与本地 PostgreSQL 合并为一个事务。独立事件写入使用稳定的操作身份及远端证据；创建后的不确定结果需要按稳定事件标识核对，修改和删除需要核对目标及版本。超时不等于远端必然失败。Todo 的后台日历镜像是一条独立业务路径，不应把本地 Todo 成功说成镜像也成功。

## 审批、回复和通知

敏感的 Telegram Agent 操作会暂停并产生待审提案。Telegram 原生按钮批准或拒绝后，系统核对用户、原操作、目标版本和运行结果，再决定是否继续；批准不会越过资源权限。Web 工作台现在不运行 Agent，也不通过另一套 Web Worker 恢复审批。

最终回答或审批卡片先写入 Outbox，再由出站 Sender 调用 Telegram Bot API。Sender 记录发送确认和分段进度；网络结果不确定时可能需要核对或重试，因此不能承诺远端仅投递一次。Receipt 已处理、业务已提交、Agent 已形成回答、Outbox 已发送是不同事实。

用户不再发消息时，Scheduler 仍按间隔扫描到期提醒与每日简报任务。通知先成为持久的待执行记录，再写入 Outbox，复用出站发送。Agent 无需持续运行到提醒时刻。调度间隔、网络和处理时间影响实际送达时间。

## 部署边界

Web 工作台由 `/workbench` 提供静态构建产物，并通过同源后端 API 操作业务。启用前必须配置 HTTPS 来源、用户绑定与会话保护。代码中的正式 Telegram 启动预检和离线数据库初始化器均对 PostgreSQL 身份有严格要求；仓库内没有经过验证的通用 Docker 一键部署。参见[部署说明](deployment.md)。