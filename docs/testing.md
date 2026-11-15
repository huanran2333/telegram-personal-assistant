# 公开测试入口与历史测试分类

## 可复现的当前检查

`ops/check_public_copy.ps1` 只执行当前业务的选定离线 Mock 测试，以及 Web 单元测试与构建。它强制使用不可用的数据库地址、空 Bot Token、关闭 Google 和 Scheduler，不连接真实外部资源。独立 PostgreSQL 与 Web 的真实持久化验收按[部署说明](deployment.md)使用全新 staging 数据目录；不会运行在默认 pytest 中。

在本副本 Python 3.12 环境，最新发布检查为 **176 项后端通过**（含 5 项 staging 安全测试）、**18 项前端通过**、前端构建通过。PostgreSQL 18.6（未安装 pgvector 扩展）staging 初始化、Web HTTPS 登录及 Todo 创建、去重、读取、删除已另行完成实际隔离冒烟；它们不是 Telegram 端到端测试。

隔离 Agent 主链可在已初始化的 staging 数据库上运行：先执行 `python ops/local_staging.py start --port 56520`，再执行 `python ops/smoke_official_agent.py --port 56520`，最后执行 `python ops/local_staging.py stop --port 56520`（均使用本副本 `.venv` 的 Python）。它使用固定响应的 Fake Chat Model，实际经过生产 `create_agent` 图、ToolNode、Todo 数据库写入、执行记录和 Checkpoint；合成轨迹保存在 Git 忽略的 `.release-local/`。**它不调用模型 API，也不能证明真实模型会自行选中工具。**

## 默认完整 pytest 为何没有全绿

退役代码清理后的默认完整离线运行：**709 passed、52 failed、3 skipped、342 deselected**。52 项失败集中在四个历史 Benchmark 测试文件：它们直接读取未公开的 U/H3 原始 Trace，或在 Trace 缺失后得到空输出、基础设施错误并触发后续断言。公开副本不包含这些私人原始轨迹；不能把这些失败记为 Agent 业务失败，也不能宣称历史评分回归已经通过。

当前业务的公开离线入口 `ops/check_public_copy.ps1` 通过 **176 项后端测试、18 项前端测试及前端构建**。要验证历史评测逻辑，需在私有环境提供经授权的原始 Trace。PostgreSQL 持久化、真实模型、Telegram 投递与 Google 集成分别需要隔离资源验收；本入口不会连接这些服务。
