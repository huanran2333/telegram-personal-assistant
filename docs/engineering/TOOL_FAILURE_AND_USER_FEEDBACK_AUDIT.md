# 统一 Agent 工具故障与用户反馈审查

日期：2026-11-04。依据当前 `feature/agent-capability-routing` 工作树，HEAD `8c59d364b46243fcfe66fcc23b93b24cd2550ed9`；生产 Agent、配置和部分测试有既存未提交修改，故不能把 HEAD 当成本次运行版本。本轮未访问真实 Google、Telegram 或业务库，未改生产代码。文中的「本轮」仅指源码检查与一项同步错误映射探针；历史模型、PG 与真实联调均明确注明。

## 1. 从请求到用户可见消息

Telegram 正常文本由 [`webhook.py:932–942`](../../app/telegram/webhook.py) 调用 `run_agent`。持久入口实际由 [`durable_runtime.py:114–165`](../../app/telegram/durable_runtime.py) 领取 Receipt、运行图、保存业务结果与回复意图，之后 Outbox 投递；`PROCESSED` 只证明该 Update 的持久处理完成，不能证明 Telegram 已收到。WebRequest Worker 在 [`runs.py:392–416,495–530`](../../app/web/runs.py) 调同一个 Agent 图，正常完成时保存 `{"response": ...}`，权限等明确异常变 `FAILED/safe_error`；其他异常记录日志、等待租约恢复。Web 入口是后端运行接口与审批入口，不能由此推断 Mini App 有聊天界面。

[`official_runtime.py:48–105,155–263`](../../app/agent/official_runtime.py) 用 PostgreSQL Saver 和 `create_agent`，图里有官方 ToolNode、HITL、ToolErrorMiddleware，以及项目的 `OfficialToolAuthorization`、`AllGuardedPreflight`（[`official_graph.py:135–168`](../../app/agent/official_graph.py)）。一个 Tool Call 可能得到：成功 ToolMessage、`status=error` 的可解释拒绝、HITL 中断，或者未处理异常使本次图调用退出。**ToolMessage 是模型输入，不是已经发送给用户的消息**；最终回复由模型生成，只有入口成功持久化并经 Outbox/前端展示才是用户可见结果。

证据等级：**代码保证**指确定性执行分支；**既有测试**指仓库中有断言及已保存的历史结果；**历史真实模型**与**真实 Telegram**分别注明；**未验证**不得当成已送达。

## 2. 故障处理和反馈矩阵

下表「用户看到」按已保存原文、确定的入口状态或「不保证」分别陈述；没有原始最终回复的行不编造答复。读取工具的单次 Call 最多两次 Adapter 尝试、单次 20 秒、总 35 秒，`Retry-After` 只有落在截止时间内才等待（[`official_read_tools.py:34–103`](../../app/agent/official_read_tools.py)）。模型自己再提一个新 Call 不计入这个两次限制。

| 故障场景 | 当前处理机制 | 是否重试 | 实际业务状态 | Agent 收到什么 | 用户最终看到什么 | 是否合理、证据 |
|---|---|---|---|---|---|---|
| Calendar 正常查到空列表 | MCP 查询（有查询词时空结果可能再用 `list-events` 核对）返回 `[]` | 不作故障重试 | 只读，无写入 | 成功 ToolMessage `[]` | 模型可能说无匹配日程；具体措辞未在本轮验证 | 合理；[`calendar.py:208–244`](../../app/agent/calendar.py)、[`official_read_tools.py:112–121`](../../app/agent/official_read_tools.py) |
| Calendar 网络超时/连接断开 | `CalendarError` 因果链内若有已识别传输异常，有限重试；耗尽由 ToolErrorMiddleware 转错误 | 最多两次 | 未得到可靠读取结果 | 原 Call ID、`error`，`connection_timeout` 或 `service_unavailable` | 取决于模型及入口投递；不能说「日历为空」 | 代码＋既有边界测试；MCP 若只给无类型错误文本，分类可能退化为 `read_failed` |
| Calendar 429、500/502/503 | 可识别状态重试，遵守可用的 `Retry-After`；耗尽给错误 ToolMessage | 最多两次，超预算不重试 | 查询未证实 | `rate_limited` / `service_unavailable` | 历史真实 Qwen 的 503 原文：「Google 日历服务当前**暂时不可用**……这**不代表**当天没有日程……建议您稍后重试」；不是当前工作树重测 | 合理；[旧验收及原始 JSON](../../experiments/official_agent_eval_rebuild_20261023/EVALUATION_FINAL_ACCEPTANCE_20261023.md) |
| Calendar 401/403 | 认证/访问错误不按瞬时故障重试 | 一次 | 未取得日历事实 | 有类型状态时 `permission_denied`，不包含 Token | 历史真实 Qwen 403 原文说明「认证或访问权限不足」「无法确认当天是否有日程」；该案例的 Todo 当时被**旧授权规则**拒绝，不能拿它证明当前部分成功 | 合理；`tests/test_official_calendar_error_boundary.py:51–80` 与上述旧验收 |
| Calendar 单事件 404/410 | 不自动重试；映射为资源不存在**或无权访问** | 一次 | 不能确证「不存在」 | `resource_unavailable`，要求不据此推断空结果 | 无当前最终回复样本，取决于模型 | 合理的保守解释；[`official_graph.py:62–110`](../../app/agent/official_graph.py) |
| Calendar 响应结构异常 | `availability`/候选时间检查忙碌列表；错误转 `CalendarError`；未识别程序异常仍可传播 | 通常不重试 | 无可信空档 | 已分类错误消息，或图异常且无完整工具反馈 | 前者取决于模型，后者本轮没有明确业务回复保证 | 代码保证；[`calendar.py:179–205`](../../app/agent/calendar.py)、[`scheduling.py:42–78`](../../app/agent/scheduling.py) |
| Gmail 正常查询无匹配 | API `messages` 缺省为空列表，`gmail_search/unread/thread` 序列化 `[]` | 不重试 | 只读 | 成功 ToolMessage `[]` | 可说「没有匹配邮件」，须限于查询范围 | 代码保证；[`gmail.py:189–213`](../../app/agent/gmail.py)、[`official_read_tools.py:172–207`](../../app/agent/official_read_tools.py) |
| Gmail 网络超时、429、5xx | `GmailUnavailable` 携带 `httpx` 原因；只读有限重试，耗尽脱敏分类 | 最多两次 | 未取得邮件事实 | 原 Call ID、`error`，相应 `connection_timeout/rate_limited/service_unavailable` | 历史真实 Qwen 503 原文：「Gmail 服务当前不可用……**不能**据此判断‘没有未读邮件’……建议稍后重试」 | 合理；[`gmail.py:163–186`](../../app/agent/gmail.py)、[旧验收](../../experiments/official_agent_eval_rebuild_20261023/EVALUATION_FINAL_ACCEPTANCE_20261023.md) |
| Gmail 401/403、OAuth 授权失效 | 不作只读重试；原因映射到权限或重新授权 | 一次 | 未取得邮件事实 | `permission_denied`；异常原文不进入 ToolMessage | 无当前最终回复样本，不能保证模型给出正确操作指引 | 代码＋既有测试；[`official_graph.py:62–107`](../../app/agent/official_graph.py) |
| Gmail 404、坏 JSON/坏邮件结构 | 404 为资源不可用；无效响应为 `read_failed`，不会转成空列表 | 一次 | 邮件内容未知 | `error` ToolMessage | 无当前最终回复样本 | 合理的错误/空结果分离；[`gmail.py:95–103,163–213`](../../app/agent/gmail.py) |
| Todo 创建/更新/提醒时间来源不匹配，或 Calendar 候选时刻不匹配 | `OfficialToolAuthorization` 在 handler 前返回错误；同轮、同标题、原 Calendar Call ID、精确起止是候选证明 | 不重试业务写入 | **本次 Tool Call 未执行**；其他独立 Call 可已执行 | 原 Call ID 的 `error`：`Tool request rejected (TodoTimeNotAuthorized); nothing executed.` | 历史 U15 用户实际收到「设置提醒被拒绝（TodoTimeNotAuthorized）……可能原因：时间授权校验未通过」，未能说明具体错在哪里；U15 假拒绝现已修复 | 安全边界正确，但对真实拒绝的解释仍笼统；[`official_permissions.py:20–69,151–193`](../../app/agent/official_permissions.py)、[U15 修复证据](../../experiments/agent_business_baseline_20261103/REMINDER_AND_U13_FIX_RESULTS_20261104.md) |
| owner/工具不可用、目标缺失或不唯一 | 动态过滤＋执行前检查；Todo 删除预检只按 owner 找目标，失败不生成审批 | 不重试 | 被拒绝的写入未执行 | `Tool unavailable...`、`Todo target is missing or not unique...`；部分 Calendar/邮件预检只给 `PermissionError`/`ValueError` 类名 | 最终措辞依赖模型；无本轮用户端证明 | 执行隔离合理，预检反馈不够具体；[`official_permissions.py:140–165`](../../app/agent/official_permissions.py)、[`official_todo_approval.py:89–116`](../../app/agent/official_todo_approval.py)、[`official_guarded_tools.py:38–104`](../../app/agent/official_guarded_tools.py) |
| Tool Schema 参数类型错误 | 框架 ToolNode 在参数绑定阶段返回校验错误；工具函数未运行 | 不按网络故障重试；模型可自行修参 | 无该次业务写入 | 校验 ToolMessage；真实 Qwen 曾把 `preferred_day_parts` 填成字符串后修正 | 最终是否解释参数问题取决于模型；反复无进展可触发图递归上限 | 已有真实模型消融，非本轮新测试；[消融结果](../../experiments/cross_tool_qwen_ablation_20261104/RESULTS_AND_CONCLUSION.md) |
| HITL 用户拒绝 | 官方中断恢复返回拒绝 ToolMessage，项目阻止本轮同目标重复提案 | 不重试该目标 | 未执行拒绝的操作 | 官方 `User rejected the tool call` 错误结果 | 真实双审批联调已证明 A 批准删除、B 拒绝保留且用户获部分结果；本轮未重做 | 合理；[`official_todo_approval.py:73–103`](../../app/agent/official_todo_approval.py)、[真实事故复测](../../experiments/official_agent_migration_20261020/FINAL_LIVE_ACCEPTANCE_20261020.md) |
| Calendar/Gmail 审批**预检读取**失败 | `AllGuardedPreflight.aafter_model` 调 `prepare_calendar_create/update/delete` 或邮件来源核验；这些函数可抛 `CalendarError`/`GmailError`，但该钩子仅捕获 `PermissionError/ValueError` | 此钩子无局部重试；入口可能重试整轮 | 提案未形成、危险写入未执行；同轮其他独立工具也未进入 ToolNode | **没有此 Call 的 ToolMessage**；`ToolErrorMiddleware` 只包工具调用，包不到模型后预检 | Telegram Receipt 记录失败并等重领，Web 普通异常留 RUNNING 等租约；该次无明确业务回复保证 | **已由源码确认的错误处理缺口**，完整图注入仍缺；[`official_guarded_tools.py:28–104`](../../app/agent/official_guarded_tools.py)、[`unified_tools.py:306–362,365–412`](../../app/agent/unified_tools.py) |
| 批准后 Todo 目标版本变化 | ActionExecution 同事务记录 `FAILED/todo_target_changed`，不删除 | 不自动重试 | 目标保持或已被其他动作改变 | 目前 `_execute` 仅返回 `status=FAILED. Do not claim...`，未转达执行器已有的具体原因 | 若模型只见通用文字，用户未必知道要重新查询/审批；无最终回复实测 | **反馈缺口**，业务状态保护已有；[`executor.py:122–159`](../../app/agent/executor.py)、[`official_guarded_tools.py:107–133`](../../app/agent/official_guarded_tools.py) |
| Calendar 创建请求发出前失败/明确拒绝 | 未发送时返回 `FAILED`；明确首次相似性拒绝记 `FAILED_DEFINITE`；审批前冲突不发写入 | 不盲重试 | 有证据时明确未提交 | `FAILED` ToolMessage，但当前正文多为通用 `Do not claim...` | 没有实际最终回复样本 | 状态分类合理，原因传递有限；[`calendar_create.py:112–145,181–209`](../../app/agent/calendar_create.py) |
| Calendar 创建/修改/删除远端成功但响应丢失 | 稳定资源 ID/标记或批准 ETag，按原 action_id 读回；成功证据才记 `CONFIRMED` | 仅在证实原状态且同身份时有限重试 | 实际远端可能已成功；读回确认后已执行 | 确认则 `EXECUTED`；未确认仍 `EXECUTING` | 现有 ToolMessage 明确 `outcome=UNCONFIRMED` 时不能说成功或失败；具体最终用户答复未做本轮真实模型测试 | 合理；[`calendar_create.py:147–258`](../../app/agent/calendar_create.py)、[`calendar_modify.py:128–263`](../../app/agent/calendar_modify.py)、`tests/test_r5_2_create_reconciliation.py:108–132`、`tests/test_r5_3_calendar_modify.py:159–172` |
| Calendar 404/409/412 或持续不可核实 | 409 先读回同 ID；修改/删除 412 或 ETag 变化进 `PARKED_REVIEW`；不可读可 `UNCERTAIN` | 不用新 ID 盲写；原 ID 有界证据重试 | 可能成功、失败或与他人版本冲突，分别保留 | `status=EXECUTING` 且 `outcome=UNCONFIRMED`，没有把未知说成明确失败 | 用户端结果取决于模型，缺完整图故障样本 | 状态协议合理；[`calendar_modify.py:161–263`](../../app/agent/calendar_modify.py)、`tests/test_r5_3_calendar_modify.py:145–160,273–317` |
| Todo→Google 异步镜像失败 | 本地 Todo 独立提交；镜像 Worker 同一稳定远端 ID 对账、`RETRY`（指数间隔，最多 5 次）或 `REVIEW` | 有界后台重试 | Todo 成功，Google 可能未同步 | **原 Agent ToolMessage 只证明 Todo 创建**，不会收到后来镜像失败的 ToolMessage | 原请求可能已回复 Todo 已创建；用户需另看镜像状态，不能把镜像说成已完成 | 机制存在；主链真实成功一次，失败通知到用户未验证；[`todo_calendar_sync.py:88–149,151–249`](../../app/agent/todo_calendar_sync.py) |
| 独立 Todo 成功＋Calendar/Gmail 读失败 | 错误 ToolMessage 与成功 ToolMessage 同批进入图；Todo+ActionExecution 持久 | 只重试读；Todo 重放取原证明 | Todo 仅一条，读取目标失败 | 各自原 Call ID 的成功/错误结果 | OA11 历史真实 Qwen 原文区分「日历失败、Todo 已创建」；Gmail+Todo 有**脚本模型＋隔离 PG**断言，不能说已验证真实 Qwen 用户反馈 | 合理的局部恢复；[`test_pg_official_oa11.py:98–154`](../../tests/integration/postgres/test_pg_official_oa11.py)、[`test_pg_official_gmail_read_error.py:29–78`](../../tests/integration/postgres/test_pg_official_gmail_read_error.py)、[OA11 原始回复](../../experiments/official_agent_eval_20261020/oa11_real_qwen_post_fix_valid_20261021/OA11.json) |
| Calendar/Gmail 读失败后依赖排期/邮件派生写入 | 日历排期需本轮成功候选 ToolMessage；邮件派生写入需来源核实并经审批 | 不重试依赖写入 | 无可信上游事实时应无下游写入 | 拒绝 ToolMessage 或未产生提案 | 最终澄清/失败文字依赖模型；没有覆盖所有表达的端到端证明 | 执行端有约束；[`official_permissions.py:20–69`](../../app/agent/official_permissions.py)、[`unified_tools.py:366–413`](../../app/agent/unified_tools.py) |
| 工具未知程序异常、整个图异常 | 未被只读 Middleware 处理的异常向上抛；Telegram 记 FAILED 重试，Web 普通异常留 RUNNING 待租约重领 | 入口级按原身份重试 | 可能已有独立工具提交，须依证明重放 | 无完整该批 ToolMessage 或最终 AIMessage | **不能保证用户收到明确业务回复**；日志和 Receipt/Web 状态不是聊天消息 | 现有故障恢复边界；OA11 原事故已修，其他未知异常仍可能触发；[`durable_runtime.py:114–165`](../../app/telegram/durable_runtime.py)、[`runs.py:495–530`](../../app/web/runs.py) |

### 原始用户回复与版本限制

2026-10-21 OA11 的真实 Qwen 合成请求最终回复原文核心句为：「日历服务读取失败，未获得可靠结果。无法确认当天是否有日程安排……已成功创建待办『OA11-离线资料』」。对应原 Call ID 的 Calendar `error` ToolMessage、Todo `success` ToolMessage、PG 仅一条 Todo 见[原始 JSON](../../experiments/official_agent_eval_20261020/oa11_real_qwen_post_fix_valid_20261021/OA11.json)。这是**历史模型＋隔离 PG＋Fake Calendar**证据，不能作为当前每次用户端 Telegram 已送达的证据。

2026-10-23 的 503 真实 Qwen 最终回复见[原始 `calendar_503.json`](../../experiments/official_agent_eval_rebuild_20261023/error_reason_acceptance_raw_20261023/calendar_503.json)和[`gmail_503.json`](../../experiments/official_agent_eval_rebuild_20261023/error_reason_acceptance_raw_20261023/gmail_503.json)。同日 Todo+Calendar 403 的模型准确报告两项失败，**因为当时的旧 Todo 语义授权拦截**，不是「Todo 成功＋Calendar 失败」；该旧检查目前已撤除。历史双审批是**真实 Telegram 联调**，其 pending write/主 checkpoint ID 不前进导致的单项回执误判已经修复，不能归为本轮新缺陷。

## 3. 已确认问题与证据不足

| 类别 | 结论 | 证据 | 严重性 |
|---|---|---|---|
| 反馈缺陷 | `TodoTimeNotAuthorized` 是服务端时间来源/目标绑定拒绝；ToolMessage 只给代码和「nothing executed」，不指出该待办的时间为何未获证实。模型可能将其泛化成权限问题。 | [`official_permissions.py:172–192`](../../app/agent/official_permissions.py)；历史 U15 用户端曾收到代码及笼统解释，虽该**假拒绝本身已修**。 | 值得改善；不改变授权判断 |
| 反馈缺陷 | `AllGuardedPreflight` 将 Calendar/邮件预检的 `PermissionError`/`ValueError` 压成异常类名，丢失已知的「时段已过/未证实空闲/来源未核实」等安全原因。 | [`official_guarded_tools.py:38–104`](../../app/agent/official_guarded_tools.py)、[`unified_tools.py:306–413`](../../app/agent/unified_tools.py)。 | 值得改善；只映射白名单错误码 |
| 错误处理/业务流程缺陷 | Calendar 建议写入的空档读取、修改/删除的目标版本读取，以及邮件派生提案的来源读取都发生在 `aafter_model`，可抛 `CalendarError`/`GmailError`。现有钩子只捕获 `PermissionError/ValueError`，异常会在 ToolNode 前中止本轮；既有只读 `ToolErrorMiddleware` 无法介入，独立工具也失去本轮反馈。 | [`official_guarded_tools.py:28–104`](../../app/agent/official_guarded_tools.py)、[`unified_tools.py:306–362,365–412`](../../app/agent/unified_tools.py)、[`durable_runtime.py:114–165`](../../app/telegram/durable_runtime.py)。这是确定的代码分支，**尚无本轮完整图故障注入**。 | **必须修复的局部缺口**；不应扩大捕获到未知写入异常 |
| 反馈缺陷 | 批准后 Todo 版本冲突在业务层已有明确 `todo_target_changed`，但当前官方工具的失败 ToolMessage 只说 `status=FAILED`；模型收不到「目标已变化，重新查询与审批」这一原因。 | [`executor.py:141–149`](../../app/agent/executor.py)、[`official_guarded_tools.py:125–133`](../../app/agent/official_guarded_tools.py)。 | 值得改善；不改变事务 |
| 反馈/分类边界 | MCP Calendar `_call` 对任意异常构造 `CalendarError`。若底层异常不保留可识别的 HTTP/传输类型，现有分类只能给 `read_failed`，也不能安全决定是否重试；宽捕获还会把某些内部异常当读失败。 | [`calendar.py:179–195`](../../app/agent/calendar.py)、[`official_read_tools.py:60–103`](../../app/agent/official_read_tools.py)、[`official_graph.py:62–125`](../../app/agent/official_graph.py)。尚无当前真实 MCP 403 原始异常类型。 | **潜在风险/验证不足**，不能擅称现有 403 必被误分 |
| 状态/用户反馈边界 | Calendar 远端写入不确定时，内部保留 `EXECUTING/UNCERTAIN/PARKED_REVIEW`，ToolMessage 明令不重写；但本轮没有完整图及 Telegram/Web 终端样本证明模型每次如实表达「待核实」。 | `tests/test_official_guarded_tool_feedback.py:13–49` 验证内部 ToolMessage；R5 对账测试验证远端状态，**没有自动等价于用户回复测试**。 | 验证不足；当前无证据说它已误报成功 |
| 业务流程边界 | ToolNode 同批发生未经处理的异常时，兄弟工具可能已提交而本批没有完整反馈；OA11 的已知 CalendarError 已隔离，未知异常仍可触发入口重试。 | [OA11 前后轨迹](../../experiments/official_agent_eval_20261020/OA11_FIX_RESULTS_20261021.md)、[`official_graph.py:113–125`](../../app/agent/official_graph.py)。 | 对**未知**故障属潜在风险，不将 OA11 再列为未修复 |

没有发现当前版本「Calendar/Gmail 已知 403/503 被当作空列表」「明确的 Calendar 写入未知被当作失败后盲目重发」「Todo+Gmail 独立部分成功在脚本图中丢失」的证据。也没有完整证明所有真实模型最终答复都忠于业务状态。

## 4. 最小改进判断

**必须修复：**审批**预检读取**失败会在 ToolNode 前传播，无法形成该 Call 的错误 ToolMessage；这是已确认的局部流程缺口。建议先用 Fake Calendar/Gmail 在 `aafter_model` 注入类型化 403/503，并在同批加入独立 Todo，固定失败轨迹与数据库状态。最小修复位置是现有 `AllGuardedPreflight`：只处理已知**预检读取**异常，给原 Call 的 `prepared_actions` 保存脱敏错误；既有 `when` 不触发审批，`awrap_tool_call` 返回配对错误 ToolMessage，独立工具照常进入 ToolNode。不得将已经发出的写入结果未知归入这个分支。修复后以完整图、业务快照和最终消息回归。

**值得改善：**在已有拒绝分支做**安全错误码到业务解释的白名单映射**，覆盖 `TodoTimeNotAuthorized`、审批前目标/时段/owner 拒绝和批准后 `todo_target_changed`；说明「该次写入未执行」或「结果待核实」及下一步，继续用原 Tool Call ID。批准后的 Calendar `EXECUTING` 绝不能被翻译成明确失败。测试要贯穿 ToolMessage→ScriptedModel 最终消息→Web/Telegram 持久回复意图，并用至少一条真实 Qwen 合成请求检查模型是否照实表达；同时核对数据库/远端 Fake 状态。这只改善反馈，不改变授权或业务协议。

**值得先验证再决定：**对真实 MCP Calendar 只读 403/429/5xx 的底层异常类型做脱敏故障注入；若状态只藏在文本里，最小适配点是 Calendar Adapter 的类型化错误保留，不能靠解析任意异常字符串猜测。另用 Fake Calendar 与完整图固定测试「远端已成功但响应丢失」「不可核实」，分别核对 ToolMessage、最终回复和业务证明；目前业务对账的既有测试充分，不需重做协议。

**暂不需要：**全局 ToolRetryMiddleware、写工具盲重试、新状态机或 Saga。只读 `status=error` 并不表示全部任务失败，写入 `status=error/UNCONFIRMED` 也不表示远端没有执行。

## 5. 本轮验证记录与限制

- 本轮只读运行 `git status`、源码/测试/实验记录检索，并用 Python 同步调用 `_read_tool_error`：合成 Calendar 403 得到 `calendar_events 读取失败（permission_denied：认证或访问权限不足）；未获得可靠结果。不要据此判断没有日程。`；合成 Gmail 503 得到 `gmail_unread 读取失败（service_unavailable：服务暂时不可用）；未获得可靠结果。不要据此判断没有邮件。`，故意放在异常中的 `token=SECRET` 未泄露。这些只是**项目错误映射函数结果**，并非新的 ToolMessage、重试次数、模型回复或 Telegram 消息。直调使用系统 Python 3.12 与本地 `.venv312` 依赖路径；未把它当作正式运行环境验收。
- 曾尝试运行 `tests/test_official_calendar_error_boundary.py`、`tests/test_official_read_retry.py`、`tests/test_official_guarded_tool_feedback.py`。当前 `.venv` 指向不存在的 Python 3.14；改用系统 Python 3.12 加 `.venv312/Lib/site-packages` 后，pytest 的 `tests/conftest.py` 自动 fixture 在 setup 阶段产生 17 个 ERROR、0 个测试函数执行；单独 asyncio 定向调用也未在 30 秒内完成，已中止。故**本轮没有新的单元或 PG 通过数**，不能把历史 `16 passed/19 passed` 当作本轮结果。未连接隔离或真实 PostgreSQL。
- 当前原始 Trace 足以证明历史 OA11 部分成功、旧 403/503 模型措辞、U15 用户体验及 R5 远端对账定向测试；尚缺当前工作树在同样场景的真实 Qwen 与用户端送达新证据。本轮未修改生产、测试、数据库或简历；之前尚未验收的反馈文案改动已逐处撤销，原有未提交工作保留。

## 6. 三个直接回答

1. **处理是否足够合理？** 已知 Calendar/Gmail 只读瞬时故障有界重试、永久错误不重试，错误与空结果分开；Calendar 写入有稳定身份和远端对账，Todo 镜像有后台 `RETRY/REVIEW`。核心状态处理基本合理，但审批预检读取异常能在 ToolNode 前中止本轮，用户任务完成与最终回复也没有同等强度的普遍保证。
2. **哪里会反馈不足？** 时间来源拒绝只给内部码；部分审批预检只给异常类名；批准后 Todo 版本冲突的具体业务原因没有传进当前官方工具结果；远端写入待核实虽已在 ToolMessage 说明，但缺完整用户端验证。未知异常造成图退出时可能根本没有业务回复。
3. **只做一次优化选什么？** 优先补 `AllGuardedPreflight` 的已知 Calendar/Gmail **预检读取异常**隔离，保证原 Call 的错误 ToolMessage 和独立工具反馈，再验证用户最终消息；这是现有 Middleware 的局部修补。安全拒绝原因映射排在其后。Calendar MCP 的类型化异常先以脱敏故障注入确认证据，再决定是否需要改 Adapter。
