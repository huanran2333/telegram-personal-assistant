# 统一 Agent 全链路故障恢复审查（2026-11-04）

## 0. 范围、版本与判断

本报告依据当前**未提交工作树**，分支 `feature/agent-capability-routing`，HEAD `8c59d364b46243fcfe66fcc23b93b24cd2550ed9`。`official_context.py`、`official_graph.py`、`official_permissions.py`、`official_runtime.py`、Telegram、Web 等均有未提交变更；`official_model_failover.py` 尚未跟踪。因此 HEAD 不是本报告所审的运行版本。本轮只读源码、既有测试和实验，不运行模型、数据库或 Bot；下述“通过”均指**先前保存的运行记录**，不是本轮重测。本轮只新增本报告。

**总体判断：**已经具备分层的局部恢复：模型请求有限切换，Calendar/Gmail 只读有限重试与带原因的错误 ToolMessage，HITL，Todo 同事务执行证明，远端 Calendar 对账，Saver/Receipt/WebRequest 重放及 Outbox/提醒后台处理。OA11 和双审批事故所暴露的精确窗口已有修复和对应证据。主要未闭合的是**整轮 Agent 反复无进展或永久失败时，如何有限终止、保留已提交结果、向用户交代实际状态**。这与“再加一次工具重试”不是一回事。

## 1. 真实生产路径及责任分层

```text
Telegram POST /webhook --去重接纳--> Receipt/Worker --同线程锁--> official_runtime.run_agent
WebRequest Worker ------稳定 turn_id/lease--------------------^       |
                                                                 PostgreSQL Saver
                                                                         |
                                                     create_agent: model ↔ ToolNode
                                                       |              |          |
                                            模型有限切换     只读重试/错误消息   审批 interrupt
                                                                         |
                                             Todo+ActionExecution / Calendar 远端证据
                                                                         |
                                回复意图持久化 → Outbox → Telegram；提醒另由 Scheduler 生成意图
```

[`build_graph`](../../app/agent/official_runtime.py) 在持久模式使用 `AsyncPostgresSaver`，构造 [`build_official_agent`](../../app/agent/official_graph.py) 的官方 `create_agent`，注册 `OfficialToolAuthorization`、`OfficialContextMiddleware`、可选 `OfficialModelFailover`、官方 `HumanInTheLoopMiddleware`、官方 `ToolErrorMiddleware`、项目 `AllGuardedPreflight`（`official_graph.py:135–168`）。当前运行入口 [`run_agent`](../../app/agent/official_runtime.py)（155–263）用版本化线程、可信 owner、Receipt 或 WebRequest 的稳定消息身份、`active_request` 和 `active_turn_id` 调图；新请求不能占用待审批/未完结的旧节点；同入口重放且 `snapshot.next` 时才 `ainvoke(None)` 续跑。Telegram [`webhook.py:633,714,932–942`](../../app/telegram/webhook.py) 与 Web [`runs.py:233,402`](../../app/web/runs.py) 接入它；旧 `graph.py` 只作为条件性历史读取/迁移，不是新请求的执行图。

**框架与项目分工：**`create_agent` 的模型—工具循环、ToolNode 参数绑定/同批调度、HITL interrupt/resume、Saver 的 checkpoint/pending writes 是框架能力。业务授权、同目标候选来源核验、ActionExecution、Receipt/Lease/Fence、WebRequest、Calendar 对账、Outbox 和回执证明均是项目实现。Checkpoint 记录图进度，不能替代 Todo/远端业务执行证明；错误 ToolMessage 仅让模型知道失败，并不自动完成用户目标。

## 2. 已有机制及证据等级

| 机制 | 当前行为及适用故障 | 证据与等级 | 边界 |
|---|---|---|---|
| 模型请求切换 | 本机 `.env` 形态核验显示一个主模型和一个备用已配置（未输出地址/凭据）；主模型可重试错误最多 2 次，间隔 0.25 s，随后备用各 1 次；连接/超时/429/5xx 才切换，401/403 直接抛。主 12 s、备用 45 s 是配置默认超时；实际运行时可覆盖。 | [`official_model_failover.py:36–115`](../../app/agent/official_model_failover.py)、[`config.py:26–35`](../../app/config.py)、[`tests/test_official_model_failover.py`](../../tests/test_official_model_failover.py)；实现＋隔离测试。 | 只重放**模型请求**，不重跑已完成 ToolNode；不是整轮截止时间。当前切换钩子未读取模型 429 的 `Retry-After`。未配置备用时 `nodes.py:100–103` 不设置 `max_retries=0`，SDK 默认重试次数须按实际运行配置核对，不能写成恒定的两次。 |
| 只读工具重试 | 四个 Calendar、三个 Gmail 读取经 `_read_with_retry`：仅传输/超时/429/5xx（或已分类的 provider unavailable），最多 2 次、总 35 s、单次 20 s；在预算内遵守 `Retry-After`；403/参数错误不自动重试。 | [`official_read_tools.py:34–103,110–210`](../../app/agent/official_read_tools.py)、[`tests/test_official_read_retry.py`](../../tests/test_official_read_retry.py)、Calendar/Gmail 错误边界测试；实现＋测试。 | 模型拿到错误 ToolMessage 后**自己再发新 Tool Call**不计入这两次；多个层次可能放大总等待。无跨请求全局重试预算。 |
| 只读失败隔离 | 已识别 `CalendarError`、`GmailError` 及选定 Tavily 只读异常由官方 `ToolErrorMiddleware` 转为原 Call ID、`status=error` 的安全分类原因；其他异常及危险写入不按“读失败”吞掉。 | [`official_graph.py:53–125,159–162`](../../app/agent/official_graph.py)、[OA11 原始/修复后 Trace](../../experiments/official_agent_eval_20261020/OA11_FIX_RESULTS_20261021.md)；实现＋真实模型/隔离 PG。 | Calendar/Gmail **读取目标仍失败**。`CalendarAdapter._call` 的宽泛 `except Exception → CalendarError`（[`calendar.py:179–195`](../../app/agent/calendar.py)）可能把底层程序错误也归类为读取错误；未见相应生产故障，属待验证风险。 |
| 参数与权限 | ToolNode 在工具执行前处理参数绑定/Schema 错误；项目按可信 Runtime/State 过滤工具并二次检查身份、owner、时间来源和提醒目标。 | [`official_permissions.py:20–69,140–193`](../../app/agent/official_permissions.py)、[`official_todo_tools.py:11–83`](../../app/agent/official_todo_tools.py)；真实 Qwen 的 `preferred_day_parts` 字符串错误及 ToolMessage 见[消融原始轨迹](../../experiments/cross_tool_qwen_ablation_20261104/RESULTS_AND_CONCLUSION.md)。 | 模型可以修正参数，但没有保证停止反复相同错误。普通 Todo 创建已接受“免审批、无逐标题语义授权”产品取舍，不能说服务端能拦截一切模型意图错误。 |
| 危险操作审批/对账 | 七类提案在模型后预检并绑定 action_id、owner、目标/版本；官方 HITL 中断，批准才执行；拒绝不执行。远端 Calendar 创建/改删对 `UNCERTAIN` 先按稳定 ID/ETag 查证，最多 2 次受证据约束的提供方尝试，无法确认留 `PARKED_REVIEW`。 | [`official_guarded_tools.py:24–133`](../../app/agent/official_guarded_tools.py)、[`calendar_create.py:25,211–258`](../../app/agent/calendar_create.py)、[`calendar_modify.py:199–263`](../../app/agent/calendar_modify.py)；隔离回归和历史真实审批。 | ToolMessage `status=error` 也可能表示 **UNCONFIRMED**，绝不能当作“没执行、可换 ID 再写”。真实远端 Exactly-Once 未证明。 |
| 本地 Todo 重放 | Telegram Receipt ID＋原 Tool Call ID＋操作名形成 action_id；WebRequest ID＋Call ID＋操作名另形成 action_id。参数摘要、owner、入口 lease/fence 匹配后，Todo 与 ActionExecution 结果同一事务提交；同身份恢复取回原结果。 | [`durable.py:694–804`](../../app/telegram/durable.py)、[`test_pg_official_oa11.py:325–454`](../../tests/integration/postgres/test_pg_official_oa11.py)；提交后、ToolMessage 前故障注入已覆盖 WebRequest 与 Telegram Receipt。早期 OA11 报告称 Receipt 图级测试尚未做，**后来**的[收尾记录](../../experiments/official_agent_eval_20261020/TODO_NEGATION_AND_TELEGRAM_RECEIPT_RECOVERY_20261021.md)及现有测试已补齐，不能沿用旧结论。 | 无持久 Receipt/WebRequest 的直接测试调用只走普通业务事务；不等同生产入口重放保证。不同参数或 owner fail closed。 |
| Saver/双审批回执 | A 的 ToolMessage 可作为 pending write 持久化而主 checkpoint ID 不前进；回执现按 action_id 唯一绑定的持久工具结果证明单项完成，不再仅凭主 ID；重复 callback 不再重新 resume 已决定项。 | [`webhook.py:103–199,217–305`](../../app/telegram/webhook.py)、[真实 Telegram 事故与复测](../../experiments/official_agent_migration_20261020/FINAL_LIVE_ACCEPTANCE_20261020.md)；真实联调＋隔离 PG。 | 此事故**已修复**；尚无大规模并发、远端消息 Exactly-Once 证明。 |
| 入站与回复 | Telegram `(bot_identity, update_id)` 去重；Receipt `RECEIVED→PROCESSING→PROCESSED/FAILED/DEAD`，同线程 lease/fence；失败 30 s 后再领取，普通异常最多 5 次；恢复扫描约 5 s。业务结果证明与回复意图固定后，Outbox 分块独立发送。 | [`durable.py:175–207,243–293,345–393,416–472`](../../app/telegram/durable.py)、[`durable_runtime.py:114–237`](../../app/telegram/durable_runtime.py)、相关集成测试；实现＋隔离测试。 | `PROCESSED` 指该入站 Update 的持久处理/回复意图完成，**不等于远端回复送达**。`DEAD` 本身没有看到面向用户的新失败回复生成路径；已有运行监控可看见 DEAD。租约到期重领不经 `fail_receipt`，因此不能只从 `max_attempts=5` 推断所有反复崩溃情形必在 5 次终止。 |
| Outbox 发送 | 已有 `SENT` 及消息 ID 不再重发；远端调用异常或发送租约过期标 `UNCERTAIN`，待间隔后重新尝试，最多 5 次，超限 `EXHAUSTED`。审批卡片过期/失效可终止发送。 | [`durable.py:475–691`](../../app/telegram/durable.py)；实现＋隔离测试。 | Telegram 可能已经收到第一次请求而本地没收到确认，重新发送有**重复消息风险**；它保证可追溯/尽力投递，不能保证 Telegram 端恰好一次。 |
| WebRequest | 用户＋操作＋idempotency_key 与请求摘要去重，lease/fence，稳定 `web-request:<id>` 恢复；权限/版本等明确拒绝记 `FAILED`。 | [`runs.py:40–105,495–530`](../../app/web/runs.py)、Web/官方图集成测试；实现＋测试。 | **普通异常只记日志、保留 `RUNNING` 待租约到期再领**；`claim_one` 增加 attempts 但未见次数上限。永久模型错误或图无进展会反复领取，且可能阻塞同用户更晚请求。此为源码可确定的路径，尚缺定向故障注入。 |
| Todo→Google 镜像 | 本地 Todo 先提交，独立 Worker 约 15 s 扫描；稳定远端事件 ID、owner/ETag、远端读回和有界重试，不能确定则 `REVIEW`。 | [`todo_calendar_sync.py:24,165–263`](../../app/agent/todo_calendar_sync.py)、[真实 Google 时间块创建及清理](../../experiments/agent_business_baseline_20261103/REMINDER_AND_U13_FIX_RESULTS_20261104.md)；实现＋真实账号定向联调。 | 非跨服务原子事务；Todo 成功不等于镜像已经同步。`REVIEW` 需操作员处理；不宜自动补偿删除用户 Todo。 |
| 提醒/简报 | Scheduler 从业务状态生成唯一 occurrence，READY 和投递意图同事务；过期/取消时核对 Todo 版本；Outbox 再投递，occurrence 对账为 `DELIVERED` 等。 | [`proactive.py:80–87,126–257,257–339,397–485`](../../app/scheduler/proactive.py)、[2026-11-04 真实提醒](../../experiments/agent_business_baseline_20261103/REMINDER_AND_U13_FIX_RESULTS_20261104.md)；实现＋一次真实送达。 | 送达可能延迟；Telegram 未确认的发送仍可能重复。一次真实送达不是长期可靠性指标。 |
| 记忆 | 当前默认 `memory_mode=transcript_lexical`，只用 Profile/Transcript 来源检索；当前工具列表没有 `control_memory`，旧抑制记录仍过滤。 | [`config.py:84–100`](../../app/config.py)、[`memory_runtime.py:47–87,126–137,237–247`](../../app/agent/memory_runtime.py)、[`official_read_tools.py:254–273`](../../app/agent/official_read_tools.py)、[退役测试](../../tests/integration/postgres/test_pg_official_memory_retirement.py)。 | R3 N06 的模型式遗忘失败后重新抽取是**旧实验路径**，默认生产模式不再做这种自动抽取；不能列为当前同一漏洞。明确的物理删除/任意 Profile 语义遗忘并未承诺。若运行时显式改用旧语义模式，应另审。 |

### 能力状态总表

| 故障场景 | 当前是否支持恢复 | 现有机制 | 证据文件/代码位置 | 当前缺陷 | 是否必须补齐 |
|---|---|---|---|---|---|
| Calendar/Gmail 瞬时读取失败 | **已具备且有隔离验证** | 2 次/35 s；分类错误 ToolMessage | `official_read_tools.py:34–103`、`tests/test_official_read_retry.py` | 模型可再提新 Call，整体次数无统一预算 | 不必重写工具重试；纳入 P1-1 终止策略 |
| 独立 Todo 成功、Calendar 读失败 | **已具备且有真实模型/隔离 PG 证据** | OA11 错误隔离、原 ID 配对、Todo 证明 | `official_graph.py:113–125,159–162`、OA11 修复记录 | 只能证明已测部分回复，不能保证所有表述 | 否；继续以业务终态评测 |
| 先读失败后仍提依赖排期 | **已具备且有定向验证** | 同轮成功候选的标题/起止核对 | `official_permissions.py:20–69`、`tests/test_official_calendar_slot_boundary.py` | 模糊自然语言窗口边界尚有争议 | P1-2 先补证据 |
| Qwen 参数 Schema 错 | **已具备且有真实模型证据** | ToolNode 错误消息、模型自行修参 | Qwen 消融 B/C/D S1/S5 原始 JSON | 反复无进展时图级异常而非有意义收尾 | P1-1 |
| 模型持续超时或永久 401/403 | **局部恢复；整轮有明确缺口** | 有限模型切换，入口重领 | `official_model_failover.py:36–87`、`durable_runtime.py:114–164`、`runs.py:495–530` | Telegram 可 DEAD 无回复，Web 普通异常无限重领 | **P1-1，近期** |
| 本地 Todo 提交后 ToolMessage 丢失 | **已具备且有隔离故障注入** | Saver 原入口续跑、ActionExecution 同事务取回 | `official_runtime.py:176–205`、`durable.py:694–804`、PG OA11 测试 | 仅持久入口保证；不能泛化到任意调用 | 否 |
| 双审批主 checkpoint 不推进 | **已具备且有真实 Telegram 复测** | 按 action_id 的 pending ToolMessage 回执 | `webhook.py:103–199`、真实双审批验收 | 远端消息仍无 exactly-once | 否，事故已修复 |
| Calendar 写入结果未知 | **已有实现并有定向测试，远端规模验证有限** | 稳定 ID/ETag 读回、有限证据重试、PARKED_REVIEW | `calendar_create.py:211–258`、`calendar_modify.py:199–263` | 最终需人工复核，不能假设远端完全可观测 | 否，不新增盲重试 |
| Todo→Google 异步镜像失败 | **已有实现；真实成功链已验收一次** | RETRY/REVIEW、稳定事件 ID | `todo_calendar_sync.py:165–263`、真实提醒闭环记录 | 长期失败率、人工 REVIEW 响应未验证 | 暂不新增补偿框架 |
| Receipt 处理异常/Worker 崩溃 | **已具备且有隔离测试** | FAILED/租约恢复、线程锁/fence | `durable.py:243–393`、`durable_runtime.py:114–237` | DEAD 无用户结果；硬崩溃不受失败次数检查完全约束 | P1-1；高并发极端情形 P2 |
| Outbox 发送结果不明 | **已实现但远端验证不充分** | UNCERTAIN、分块有界重试 | `durable.py:475–691` | Telegram 可能重复收到，无远端幂等保证 | 否；接受并明确展示限制 |
| Scheduler 提醒失败/取消 | **已有实现且有一次真实送达** | occurrence/Outbox 对账及版本核对 | `proactive.py:126–339`、真实闭环记录 | 延迟和未知发送结果不能消除 | 否；长期可靠性未经证明 |
| N06 旧模型式遗忘 | **旧版本、默认已停用** | 当前是 Profile/Transcript 词法来源；旧抑制过滤保留 | `config.py:84–100`、`memory_runtime.py:237–247`、退役测试 | 若手动启用旧模式须重新审计 | 否，不能列成当前故障 |
| 大规模多节点容灾、全局 Saga、远端 Exactly-Once | **超出当前个人项目合理范围** | 无统一协议 | 现有各业务边界与外部 API 语义 | 需要产品、成本和外部支持 | 否，P2 |

## 3. 故障恢复决策矩阵

“已有”指代码行为；“部分”指只能恢复某一层或缺端到端证明。相同用户请求可能包含多个独立目标，**每个操作分别判定**。

| 故障类型 | 现有处理方式 | 正确动作 | 为什么；是否已有及缺口 |
|---|---|---|---|
| 模型连接/超时/502/503 | 若配置备用，主最多 2 次＋备用各 1 次；全失败异常上抛 | `RETRY→STOP/RECOVER_ASYNC` | 重试**模型请求**安全，不能重跑已完成工具；已有局部重试，欠整轮失败后的有限终态/用户反馈。 |
| 模型 429 | 与其他瞬时错误同样固定 0.25 s 再试 | `RETRY`（遵守服务等待与总预算） | 目前模型层未使用 `Retry-After`；读工具层已用。无真实 429 故障证据，暂不把它当 P0。 |
| 模型 401/403 | 不切换、抛异常；上层 Telegram/Web 仍按入口策略恢复 | `STOP`＋安全失败状态 | 认证/权限不是瞬时；模型层正确，但入口未区分其永久性，Web 可能重复领取、Telegram 可 5 次后 DEAD。 |
| 输出截断/无法解析 Tool Call | `max_tokens=4096`；框架可产生无效工具调用/异常，未见项目专门判定 `finish_reason` 的逻辑 | `REPAIR` 一次或 `STOP` | 是否可修正取决于有无可见错误反馈；当前缺定向原始例，不能说已实现可靠恢复。 |
| 参数 Schema 错 | ToolNode 给错误 ToolMessage，模型可自行改参 | `REPAIR`；无进展则 `STOP` | 消融真实 Qwen 两次把 list 填成字符串后第三次成功，说明可修复；重复次数无独立上限。 |
| 同一错误 Tool Call 反复出现 | Prompt 告知不要重试；图有 `recursion_limit=50`，无 `ModelCallLimitMiddleware`/`ToolCallLimitMiddleware` | `STOP` 并报告此前已提交项 | [真实 Qwen B 组](../../experiments/cross_tool_qwen_ablation_20261104/raw_B_S1_repeats_20261104/B-S1-r1.json)在近似**旧授权**下 7 次同类误拒后 `GraphRecursionError`；当前 D 组未证明此频率。图级上限是异常护栏，不是用户可理解的任务收尾。 |
| Calendar/Gmail 403、OAuth 失效、资源不存在 | 不做局部瞬时重试，带类别错误 ToolMessage | `STOP` 当前读取；独立目标 `CONTINUE`，授权缺失 `ASK_USER` | 已有；不能把失败当空结果或自动切换别人的账号。 |
| Calendar/Gmail 503、网络超时 | 已分类最多 2 次/35 s，耗尽后错误 ToolMessage | `RETRY→CONTINUE/DEGRADE` | 已有；降级只能用已有可靠证据或询问，不能编造空档。模型另发新 Call 是另一层，可能放大次数。 |
| Todo 成功＋独立 Calendar 失败 | OA11 后两条 ToolMessage 可同时进入图；模型可说明部分成功 | `CONTINUE`＋逐项真实反馈 | 真实 Qwen/隔离 PG OA11 已证实一次；**不是** Calendar 任务成功，也不能保证所有最终表述都正确。 |
| Calendar 失败/空结果＋依赖排期 | 当前候选来源校验只认本轮成功、同目标、精确时间的 ToolMessage | `STOP` 依赖写入，其他独立目标 `CONTINUE` | 同批提前写入、错误/空候选被测试拦；[真实 Qwen 消融](../../experiments/cross_tool_qwen_ablation_20261104/RESULTS_AND_CONCLUSION.md)未见同轮依赖调用。Prompt 引导不是确定性保证。候选查询范围与用户模糊“下午”的一致性仍有边界。 |
| Todo 权限/owner/时间来源拒绝 | 项目边界返回错误 ToolMessage，实际未写 | `STOP` 或 `ASK_USER`（仅真缺信息）；不把永久拒绝当网络错误 | U15 原假拒绝已修；复杂合法措辞可能仍保守拒绝。普通免审批创建的逐标题语义误写风险是已接受产品边界，不属于“重试可修复”。 |
| 用户拒绝审批/版本变更 | HITL/目标版本检查阻止提交 | `STOP` 原操作；新授权才重提 | 已有，不能由模型把拒绝当暂时故障重试。 |
| 本地 Todo 已提交、ToolMessage 未完整保存 | 同 Receipt/WebRequest 原身份恢复，ActionExecution 返回原文本，Saver 续未完成节点 | `RECOVER_ASYNC` 原身份、`VERIFY` 证明 | WebRequest 与 Receipt 图级故障注入已验证；不同入口/参数/owner 不复用。 |
| Calendar 写请求超时或结果未知 | `UNCERTAIN`，按稳定事件 ID/ETag 对账，必要时 `PARKED_REVIEW` | `VERIFY`，否则 `ASK_USER`/人工 | 已有，不盲写。不能因为 ToolMessage `status=error` 就再发一条新写入。 |
| Worker 崩溃/Receipt lease 过期 | 5 秒扫描到期记录，同线程锁与 fence；原入口重放 | `RECOVER_ASYNC` | 已有；普通失败计数 5，反复“硬崩溃且未进 fail_receipt”的次数上限未由该代码保证。 |
| 业务已提交、回复未发送 | Receipt 持久 outcome/intent，Outbox 独立投递 | `RECOVER_ASYNC` 回复，不再执行工具 | 已有；重复发送的远端不确定性仍保留。 |
| Telegram 发送调用超时 | `UNCERTAIN` 后有界重试，最多 5 次 | `VERIFY` 如有可靠 Telegram 证据，否则清楚标不确定 | 当前 Bot API 无同等业务幂等响应证明；重试可能重复消息。是已知取舍。 |
| Todo 镜像/提醒投递失败 | 镜像 `RETRY/REVIEW`；occurrence `FAILED/EXHAUSTED`、Outbox `UNCERTAIN/EXHAUSTED` | `RECOVER_ASYNC→VERIFY/人工` | 已有状态机；一次真实送达验证了主链，未证明任何时刻送达或远端不重复。 |

## 4. 多工具任务：调用恢复 ≠ 任务恢复

一次请求中的**独立** Todo 创建和 Calendar 查询可同批：只读失败经局部重试和 ToolErrorMiddleware 得到配对错误结果，已提交 Todo 得到成功结果。OA11 修复前是 CalendarError 上抛导致该 ToolNode 整批更新失败，但已完成的 Todo 事务不会回滚；修复后模型有证据说“Todo 已创建、日历没读到”。`asyncio.gather` 抛异常不等于已成功的兄弟工具被撤销。该用户目标是**部分完成**。

有依赖的“先读空档再创建带时间 Todo”不同：失败/空候选意味着下游写入不能运行。当前 [`_verified_calendar_slot`](../../app/agent/official_permissions.py) 核对本轮、同标题、原 Call ID 配对的成功 ToolMessage、精确起止；同批提议时尚无候选 ToolMessage，因此不能借此通过。原 T2 的问题是**先读后写正确但旧授权假拒绝**，已修；U15 提醒假拒绝、U13 正式删除工具选择漏调用也已针对已见表达修复。2026-11-04 真实 Qwen 消融未观察到同轮依赖调用，却见参数 Schema 错和近似旧授权下重复误拒/图级终止；没有证据说当前 D 组已解决所有重复错误。

最终答复仍由模型撰写。业务结果已持久化，并不自动保证文字逐项忠于数据库；旧 ScriptedModel T2 曾在 0 Todo 时宣称“已安排”，这不是当前真实 Qwen 的失败率证据。当前 [16 条冻结未见基线](../../experiments/agent_business_baseline_20261103/UNSEEN_EVALUATION_RESULTS.md)在当时版本未发现最终回复与终态矛盾（0/16），但样本小、版本早于 U13/U15 修复。`run_agent` 在没有普通终局 AIMessage 时还有 `tr("processed")` 兜底（`official_runtime.py:259–263`）；它不能当作“业务成功”，需要在任务级失败收尾测试中核对。

## 5. 持久故障窗口与仍需注意的边界

| 崩溃点 | 已知事实与恢复方式 | 尚不保证 |
|---|---|---|
| Webhook 已接收但未提交 Receipt | HTTP 失败，Telegram 可重投相同 `update_id`；成功提交后唯一约束读回同一 Receipt。 | Telegram 必定继续重投、外部网络时效不由本项目保证。 |
| Todo 事务与 ActionExecution 已提交，Saver 尚无完整 ToolMessage | 同一原入口重放 `ainvoke(None)`，同 Call ID/参数摘要取回原结果；Web/Receipt 故障注入覆盖。 | 不同新请求不能接管，测试入口无持久证明不自动得到同样保证。 |
| 双审批 A 已执行、B 待决定；主 checkpoint ID 不前进 | 持久 pending write 中按 A 的 action_id 找 ToolMessage，反馈 A 已处理且仍待 B；原回执重试不再 resume A。真实 Telegram 已复测。 | 不能声称全图已完成、B 已执行或远端 Exactly-Once。 |
| Receipt 已 PROCESSED、Telegram 回复发送失败 | 完整业务结果与固定回复意图可由 Outbox 恢复；不重跑 Agent。 | Bot API 超时可能已经发送，重投可能重复消息。 |
| 普通异常持续发生 | Telegram 记录 `FAILED`，每 30 s 再试，`fail_receipt` 在 `attempts>=5` 时 `DEAD`；Web 普通异常保留 `RUNNING` 等 lease 后重领。 | Telegram DEAD、Web 无限重领均没有本轮发现的用户可见完整终局回复；这是明确的**任务级恢复空白**，但尚无当前生产配置的端到端故障注入。 |

Outbox `UNKNOWN/UNCERTAIN` 是**发送状态**，Calendar `UNCERTAIN/PARKED_REVIEW` 是**远端业务状态**，Saver pending write 是**图状态**；三者不能互相替代。Calendar 镜像与 Todo、Scheduler occurrence 与 Telegram 发送也不是跨系统原子事务。

## 6. 当前真正的缺口与近期工作

### P0：当前没有经现有证据确认、必须立即改动的 P0

双审批事故、OA11 已提交后丢反馈、合法 Calendar 候选误拒、U15 提醒假拒、U13 已见措辞漏审批均有修复及相应回归；不能重新列作待开发 P0。普通 Todo 免审批的语义误写风险**仍存在且已明确是产品取舍**；如果产品要求“明确禁止项绝不可误写”，才需重新确定确认策略，这是权限产品决策，不能通过故障重试解决。

### P1-1：整轮无进展/永久失败的有限终态和真实反馈（**唯一近期开发建议**）

**证据：**真实 Qwen 在近似旧授权 B 组重复七次同类失败后触发 `GraphRecursionError`；当前 D 组主矩阵也有模型超时。生产图仅有 `recursion_limit=50`（[`official_runtime.py:168–171`](../../app/agent/official_runtime.py)），没有显式模型/工具调用预算。Telegram 普通异常最多 5 次后可 `DEAD`，Web 普通异常由 [`runs.py:66–93,524–530`](../../app/web/runs.py) 按 lease 反复领取，没有次数上限。模型 401/403 本来不会切换，但入口仍可能重试。这里的**确定性代码缺口**是永久错误不能形成有界、用户可知的结论；B 组重复错误属于旧授权近似配置，不能夸称当前 D 组已重现相同循环。

**最小方案：**先做针对性故障注入，覆盖模型 401/403、持续 5xx/超时、同一错误签名反复出现、Todo 已提交后模型终局失败。若复现确认：在现有 Agent/入口边界增加**每入站 turn 的有限模型调用和无进展判定**，仅在可信业务快照可读时用持久 ToolMessage/ActionExecution/业务表组成部分结果状态；Telegram 持久终止失败回执或可查询状态，Web 将永久错误/预算耗尽转为 `FAILED`，临时失败仍保留原身份、限次恢复。若图中有 `UNCERTAIN` 写入，仅说“结果待核实”，不尝试新写。可评估官方 `ModelCallLimitMiddleware`/`ToolCallLimitMiddleware`，但它们只能限次数，**不能自动给出可信终局回复**；不建议一口气加两套。改动面预计中等，重点是正确区分已提交与未执行，审批中断不能被当故障终止。最小验收：一个请求先成功创建 1 Todo，随后模型持续失败；最终 Todo/ActionExecution 仍各 1，Receipt 或 WebRequest 有**有限终态**，用户得到“Todo 已建、另一项未完成/未知”的可核对结果；重复入口不新增 Todo。另测纯 401 不再无意义反复、正常 HITL 不退化。不做则可能出现用户长期无回复或同用户请求队列阻塞。

### P1-2：模糊时间排期的语义边界，**先补证据，不先写通用恢复框架**

[真实 Qwen 消融](../../experiments/cross_tool_qwen_ablation_20261104/RESULTS_AND_CONCLUSION.md)有一次将“下午”查询成 12:00–18:00 并创建 12:00–13:00 Todo。冻结合同未明确下午起点，因此这条是**语义争议/待人工判断**，不是确定违规数。当前 `_verified_calendar_slot` 验证“写入等于读到的候选”，不验证“模型选的查询窗口等于用户意图”。先冻结含明确 13:00–18:00 的全新请求与反例；若模型仍越界，最小修复应在排期工具/执行端核对**用户已明确给出的时间边界**，不得再扩展脆弱的中文语义正则。测试须同时覆盖合法候选、空结果和直接指定时间，比较误拒成本。若不做，用户可能得到与其时间约束不符、但工具来源表面正确的 Todo。此项属于**业务正确性证据缺口**，近期不应抢在 P1-1 前重构。

### P2：暂不增加

- 不叠加全局 `ToolRetryMiddleware`：当前 Calendar/Gmail 已有 2 次/35 s 局部策略和安全错误回传；框架的通用工具重试若误覆盖写入，会把未知结果当失败重做。官方 `ToolErrorMiddleware` 本身**不重试**；参数绑定错误由 ToolNode 更早处理。[LangChain Fault Tolerance](https://docs.langchain.com/oss/python/deepagents/fault-tolerance)、[Middleware 参考](https://reference.langchain.com/python/langchain/agents/middleware)给出这些能力，但 Deep Agents 示例不是当前 `create_agent` 自动启用的配置。
- 模型 429 的 `Retry-After`/总时限值得在出现真实限流时按 [`OfficialModelFailover.awrap_model_call`](../../app/agent/official_model_failover.py) 最小完善；当前无原始 429 故障，暂不单列近期任务。微软[瞬时故障指南](https://learn.microsoft.com/en-us/azure/architecture/best-practices/transient-faults)要求按异常类型、等待建议与总预算设计重试，而非无差别叠层。
- 不为 Todo/Calendar/Telegram 加通用 Saga、自动撤销本地 Todo、跨服务事务或“端到端恰好一次”。Google 镜像与独立日程写入已有稳定身份、读回、ETag 和人工 REVIEW；补偿是**具体业务决定**，自动撤销可能破坏用户真实 Todo。[微软补偿事务](https://learn.microsoft.com/en-us/azure/architecture/patterns/compensating-transaction)也明确补偿依赖应用语义且可能失败。
- 不把 ActionExecution、Saver、Receipt、Outbox 合并成一份“统一状态”；它们分别证明不同事实。AWS [幂等 API 文章](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)支持请求身份与参数匹配、执行证明和副作用原子提交的思路，但不意味着 Google/Telegram 远端自然支持同一事务。

## 7. 官方方案适用性与最终结论

已安装版本为 LangChain `1.4.6`、LangGraph `1.2.12`、`langchain-openai 1.6.6`（本地依赖元数据）；本轮读取了对应类构造签名：`ToolRetryMiddleware(max_retries=2, tools=..., retry_on=..., on_failure=...)`、`ToolErrorMiddleware(on_error=..., tools=...)`、`ModelCallLimitMiddleware(run_limit=..., exit_behavior=...)`、`ToolCallLimitMiddleware(tool_name=..., run_limit=..., exit_behavior=...)`。官方[参考](https://reference.langchain.com/python/langchain/agents/middleware)说明 ToolError 不做重试、工具参数校验在 ToolNode 上游，限次中间件只控制调用次数。现有项目无需用新框架替换这些已验证的局部协议。

**近期最多一项代码工作：**经定向故障注入确认后，收拢“永久错误/持续无进展”的任务级有限终态与可信部分结果反馈。排期“下午”先做明确窗口新案例再决定是否修。当前恢复能力已经超过普通 `create_agent` Demo，但还不能说“任何多工具请求失败后都能自动恢复完成”。达到合格 Agent 个人项目的最低下一步，不是再叠工具重试，而是让**无法继续的请求也有明确、真实、不会重复提交业务的结束方式**。
