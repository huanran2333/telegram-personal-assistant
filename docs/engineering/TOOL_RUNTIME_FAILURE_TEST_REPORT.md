# 统一 Agent 工具实际执行异常验证

日期：2026-11-04。分支 `feature/agent-capability-routing`，HEAD `8c59d36`；**测试对象是含既有未提交修改的当前工作树**。本轮未修改生产代码、预检或历史实验，只新增两份定向测试和本报告。没有访问真实 Google、Telegram 或业务库。

## 1. 范围、证据等级与执行入口

生产图由 [`official_graph.py:135–168`](../../app/agent/official_graph.py) 构建：官方 ToolNode 执行 Todo、Calendar/Gmail 读取、待审写入、会议准备、网页搜索，配置记忆编排器时另有 `search_memory`。`ToolErrorMiddleware` **仅**包四个 Calendar 读取工具、三个 Gmail 读取工具与 `web_search`；已获批准的 Calendar 写入不在其捕获范围。`OfficialToolAuthorization` 在执行前校验可用性和时间授权；本报告关注通过边界后**工具函数内或审批恢复后的执行**。旧 `unified_loop`、旧 Workflow 工具清单不算当前生产工具。

证据记号：**G**＝真实 `create_agent` 完整图＋脚本模型＋隔离 PG18；**C**＝当前生产协调器/工具函数＋Fake 服务＋本地测试数据库；**S**＝源码分支或已有测试但本轮未完整图复跑；**H**＝历史真实模型/Telegram 记录。G/C 均不能证明真实模型会如何措辞，更不能证明 Telegram 已投递。测试图中的第二轮 AIMessage 是**预设脚本回复**，只证明执行链能够继续并返回这段文字。

隔离 PG18 由 [`run_preflight_verified_local.ps1`](../../experiments/official_agent_eval_20261020/run_preflight_verified_local.ps1)先核对 data directory、系统标识 `7696790601313522276`、停止状态和端口 `55704`，随后启动、运行并停止。专用库中的合成用户可被测试写入；正式业务库未连接。测试命令以仓库根目录为工作目录。

## 2. 故障与反馈矩阵

表内“用户可见”指**本轮实测**；“未验”表示没有真实模型＋入口投递验证。下表的 ToolMessage 均为图内消息，不能当作已送达通知。`n` 是**单个工具调用内部**次数，不包括模型另发新 Tool Call 或入口重放。

| 故障场景 | 工具执行阶段 | 捕获与重试次数 | ToolMessage / Agent 所见 | 业务状态 | Agent 最终回复 | 用户可见结果与证据 |
|---|---|---|---|---|---|---|
| Calendar 搜索超时 | `calendar_events → _read_with_retry → Adapter.search_events` | 类型化超时；n=2 | 原 Call ID、`error`、`connection_timeout`，明确未获可靠结果 | 无写入 | 脚本回复“读取失败，不能据此判断结果为空” | 未验；G，[探针](../../tests/integration/postgres/test_pg_official_runtime_failure_probe.py) |
| Calendar 403 | 同上 | n=1，不按瞬态重试 | `permission_denied：认证或访问权限不足` | 无写入 | 同上 | 未验；G |
| Calendar 429，`Retry-After: 0` | 同上 | n=2；若等待超出 35 秒总期限则 n=1 | `rate_limited：服务请求限流` | 无写入 | 同上 | 未验；G；长 `Retry-After` 由[`test_official_read_retry.py`](../../tests/test_official_read_retry.py)验证 |
| Calendar 503 持续失败 | 同上 | n=2 | `service_unavailable：服务暂时不可用` | 无写入 | 同上 | 未验；G |
| Calendar 异常响应转 `CalendarError` | Adapter 读取/解析 | n=1；不当作空日历 | `read_failed`，不回显异常中的 `token=SECRET` | 无写入 | 同上 | 未验；G；**未证明底层所有程序异常都能正确归类** |
| Gmail 超时、403、503 | `gmail_unread → _read_with_retry → GmailSource.unread` | 分别 n=2、1、2 | 原 Call ID、`error`；分别为 `connection_timeout`、`permission_denied`、`service_unavailable` | 无写入 | 脚本回复“读取失败” | 未验；G |
| Gmail 邮件/线程 404 | Gmail API 读取 | n=1 | `resource_unavailable`；不将不可访问等同“邮箱为空” | 无写入 | 脚本回复“读取失败” | 未验；G 的 404 注入在 `gmail_unread` 调用内；**未单独重跑 `gmail_thread` 404** |
| Gmail 响应结构错误 | Gmail 解析抛 `GmailError` | n=1 | `read_failed`，敏感异常原文未进入消息 | 无写入 | 脚本回复“读取失败” | 未验；G |
| Calendar 创建 403，远端明确拒绝 | 已批准 `create_event` 的协调器发送 | **n=2：错误地重发相同写请求**；随后 `PARKED_REVIEW` | 若协调器结果经工具返回，会成为 `status=error`、`outcome=UNCONFIRMED`；本轮未作完整图批准测试 | Fake 未创建事件；`CalendarActionEvidence=PARKED_REVIEW`、`ActionExecution=executing` | 未测 | 未验；C，[403 写入探针](../../tests/test_official_runtime_write_failure_probe.py) |
| Calendar 更新/删除 403 | 已批准的条件写 `If-Match` | **各 n=2：同样重发**；之后 `PARKED_REVIEW` | 同上；最终仍是结果待核实语义 | Fake 对象未变；证据与账本同上 | 未测 | 未验；C |
| Calendar 创建 409 | 发送后按**原稳定事件 ID、标记与字段**读回 | 匹配则 n=1 并 `CONFIRMED`；不匹配则 n=1 并 `PARKED_REVIEW`，不换 ID 重建 | 协调器 `EXECUTED` 或 `EXECUTING/UNCONFIRMED` | 与 Fake 远端和账本一致 | 未测 | 未验；C，[创建对账测试](../../tests/test_r5_2_create_reconciliation.py) |
| Calendar 更新/删除 412 或 ETag 被人工改变 | 条件写/原对象读回 | 检测到版本冲突不再重发该批准操作 | `EXECUTING/UNCONFIRMED`，要求核实/复审 | `PARKED_REVIEW`，未覆盖人工修改 | 未测 | 未验；C，[修改对账测试](../../tests/test_r5_3_calendar_modify.py) |
| Calendar 修改/删除 404、410 | 条件写后的精确资源读回 | 更新的不可访问不当作成功；删除仅在原作用域可核实且确证对象已不存在/取消时记成功 | `EXECUTING` 或 `EXECUTED`，随证据而变 | `UNCERTAIN/PARKED_REVIEW` 或 `CONFIRMED` | 未测 | 未验；C，`test_delete_inaccessible_404_is_not_absence`、`test_delete_verified_410_after_armed_attempt_is_observed_absence` |
| Calendar 创建/修改/删除远端已执行，响应超时 | 原 action_id 对账 | 读到匹配的 ID/标记或修改后置条件即确认，不重复执行；与远端未成功的超时分开 | `EXECUTED`；无法核实时 `EXECUTING/UNCONFIRMED` | 证实者 `CONFIRMED`；无法核实者维持不确定 | 未测 | 未验；C，R5.2/R5.3 对账测试 |
| Calendar 远端持续不可核实 | 精确读回/作用域核验 | 不盲目换 ID；最多 2 次有条件发送，随后停在核实/人工复审 | `outcome=UNCONFIRMED`，不应声称确定失败 | `UNCERTAIN/PARKED_REVIEW`、账本 `executing` | 未测 | 未验；C＋[`official_guarded_tools.py:178–196`](../../app/agent/official_guarded_tools.py) |
| Calendar 发送函数内部未料异常 | 已批准发送阶段 | 本轮 Fake `RuntimeError` 被协调器记录后对账，n=2，最终 `PARKED_REVIEW` | 协调器返回不确定；**不是普通只读错误** | 没有 Fake 远端写入，账本 `executing` | 未测 | 未验；C；异常发生在发送前还是发送后，真实服务上不能仅凭异常类推断 |
| Calendar **批准后再次读空闲**失败 | `execute_approved_proposal` 中、远端写入前 | `CalendarError` 未被局部转换；无工具内重试 | **没有配对 ToolMessage**，异常向图外传播 | 写入未启动；本次调用停住 | 无最终回复 | 未验；函数级 C，[批准后重查探针](../../tests/test_official_runtime_write_failure_probe.py)；完整图/入口后果据代码推断 |
| Todo 事务前异常 | `commit_todo_tool_operation` 校验 Receipt/Web fence 或 `perform` 前；或 SQL 事务出错 | 不按网络错重试；异常传播，事务回滚 | 本次无完整 ToolMessage | 未提交则无 Todo/证明 | 无最终回复，入口按原身份恢复 | 未验；S，[`durable.py:694–797`](../../app/telegram/durable.py)、[`test_todo_ledger.py`](../../tests/integration/postgres/test_todo_ledger.py) |
| Todo 批准后目标版本变化 | 删除工具执行器 `SELECT ... FOR UPDATE`、版本比对 | 不删除、不自动替换审批；同 action_id 记 `FAILED/todo_target_changed` | 原 Call ID、`error`，说明目标变化、需重新查询和审批 | Todo 保留，ActionExecution 失败证明同事务提交 | 未测 | 未验；C＋隔离 PG 的 `test_bulk_delete_rejects_approved_version_after_todo_edit` |
| Todo 提交后 ToolMessage 丢失 | ToolNode 完成前故障 | 原 Receipt/WebRequest、Call ID、owner、参数摘要重放；复用执行证明 | 恢复后配对成功 ToolMessage | Todo 与证明各一份 | 脚本最终可得原业务结果 | 未验本轮用户端；G，[OA11 恢复测试](../../tests/integration/postgres/test_pg_official_oa11.py) |
| 提醒授权/时间错误 | `OfficialToolAuthorization` 或 `set_reminder` 业务解析 | 不自动按网络故障重试 | 授权边界拒绝为 `error`；若业务函数**返回错误文字而非抛错**，ToolNode 可标 `success` | 拒绝/解析失败无提醒写入 | 脚本回复可说明失败，真实模型未验 | G，[提醒图测试](../../tests/integration/postgres/test_pg_official_reminder_authorization.py)；S，[`tools.py:374–407`](../../app/agent/tools.py) |
| Todo `cancel_reminder` 目标不存在 | 实际业务函数在 owner 范围内查询 | 无重试 | **`status=success`，内容却是“未找到”** | 没有 Todo/提醒变化 | 脚本最终说未找到 | 未验真实用户；G，本轮探针 |
| 同批 Todo 成功＋已处理 Calendar/Gmail 错误 | 两个实际工具在 ToolNode | 只读错误本地至多两次；Todo 不因它重建 | 各自原 Call ID 的成功/错误 ToolMessage | Todo 恰一条 | 脚本/历史真实 Qwen 可区分部分结果 | 本轮 Telegram 未验；G，[OA11](../../tests/integration/postgres/test_pg_official_oa11.py)、[Gmail 混合](../../tests/integration/postgres/test_pg_official_gmail_read_error.py)；H，[OA11 原轨迹](../../experiments/official_agent_eval_20261020/oa11_real_qwen_post_fix_valid_20261021/OA11.json) |
| 同批 Todo 已提交＋未知 `RuntimeError` | Calendar 实际读取函数，等待 Todo 提交后抛异常 | 不被只读错误中间件接纳；图调用退出 | 该批未形成完整配对 ToolMessage，Saver 仍有未完节点 | **Todo 恰一条，最终 AI 回复缺失** | 无最终回复 | 未验 Telegram；G，本轮受控竞态探针；恢复须依原持久入口和账本，不可作为新请求重写 |

读取重试常量为单 Call 最多 2 次、单次 20 秒、总 35 秒、默认间隔 0.25 秒；`Retry-After` 只有落在总期限内才等待。[`official_read_tools.py:34–108`](../../app/agent/official_read_tools.py)。Calendar 写入最多 2 次**有条件**发送，读回延迟 `(0, 0.25, 0.5)` 秒；已证实成功与结果不确定由证据/账本区分。[`calendar_create.py:24–26,181–258`](../../app/agent/calendar_create.py)、[`calendar_modify.py:19–21,161–263`](../../app/agent/calendar_modify.py)。

## 3. 实际发现的问题

### A. 明确缺陷：Calendar 403 被当作可重试不确定写入

创建 `_send` 对除 `similarity_precheck` 外的 `CalendarProviderError` 都进入 `_reconcile(... allow_retry=True)`，只对 `resource_conflict` 关闭重试；403 没有单独分支。[`calendar_create.py:181–205`](../../app/agent/calendar_create.py)。更新/删除 `_send` 只特殊处理 `etag_conflict`，其他异常走 `_reconcile`；原对象未变时仍可重发。[`calendar_modify.py:161–197,248–263`](../../app/agent/calendar_modify.py)。

本轮同条件 Fake 403 实际观察：**创建、更新、删除各发送两次**，Fake 均未改远端对象，`CalendarActionEvidence=PARKED_REVIEW`、`attempt_no=2`、`ActionExecution=executing`。该结果把明确拒绝表现为“不确定待复审”，多耗一次请求，还可能误导用户以为远端有成功可能。它不等于已发生重复远端写入；403 的 Fake 明确拒绝了两次。**建议最小修复**只在写入协调器识别类型化 401/403（以及核实为明确拒绝的状态），将原 action_id 终结为确定失败并阻止第二次发送；保持超时、409、412、不可核实结果的既有对账。用同一 Fake 403 红绿测试加原 R5.2/R5.3 回归验证。本轮未修改。

### B. 明确的局部反馈缺口：批准后再查空闲时失败

Calendar 创建第一次批准后，[`execute_approved_proposal`](../../app/agent/agent_approval.py)在发送前再次调用 `availability`；这一步抛 `CalendarError` 时，当前[受审工具 `_execute`](../../app/agent/official_guarded_tools.py)没有把“**尚未发送远端写入**”转为对应 Call 的失败 ToolMessage。函数级探针确认异常传播且远端执行器未被调用；当前 ToolErrorMiddleware 不覆盖写工具。图/Telegram 中多久能按原身份恢复，**本轮未做完整图＋Receipt 故障注入**，不能宣称永久卡死。最小修复点应只包这次已知的发送前读取，并保存明确未执行的结果；绝不能包住已经发出的写请求或把其超时说成失败。

### C. 可解释性缺口：Todo 业务失败文字仍可带 `success` 状态

`cancel_reminder` 找不到 owner 范围内的 Todo 时返回 `todo_not_found` 普通字符串；ToolNode 因函数正常返回而标为 `success`。隔离 PG 完整图实测没有业务写入，配对 ToolMessage 的内容能说明“未找到”，脚本模型也据此回复；**真实模型是否会被 `success` 状态误导未验**。这属于反馈语义不一致，不是越权写入。最小修复可只让已知业务失败返回 `status=error` 的配对结果，保留原文、Call ID 和事务重放身份；需同时验证 `set_reminder`/`update_todo` 等同类返回值及正常路径，避免为改状态重构 Todo 事务。当前未修改。

### D. 已知边界，不直接列为本轮新修复

同批未知运行异常会使本轮图调用退出，即使独立 Todo 已提交：本轮用事件门控确保 Todo 先提交，然后 Calendar 抛直接 `RuntimeError`，图未生成最终回复，Saver 保留未完节点，数据库恰一条 Todo。这是**未知程序异常继续传播的既定安全边界**；入口须按原 Receipt/WebRequest 和执行证明恢复，不能把未知异常吞成“只读失败”。此前 WebRequest/Telegram 同窗口恢复已有独立测试，不能用本轮第一次异常的快照直接宣称无法恢复。

## 4. 其他实际工具与用户入口

当前图的模型可见工具清单由[`official_graph.py:135–168`](../../app/agent/official_graph.py)与[`unified_tools.py:152–168`](../../app/agent/unified_tools.py)确定，按账号/配置过滤。Todo：`create/list/update/search/set_reminder/cancel_reminder`；Calendar 只读：四项；Gmail 只读：三项；受审写入：`propose_todo_delete`、批量标签删除、Calendar 创建/更新/删除、邮件来源派生 Todo/日程；另有 `prepare_meeting`、`web_search`，配置记忆时有 `search_memory`。Gmail 没有发送/修改/删除邮件工具。

`web_search` 仅对列入 `_WEB_READ_ERRORS` 的 Tavily/Requests 异常返回安全错误结果；缺 Key 时工具会被过滤，未知异常仍传播。[`official_graph.py:55–125`](../../app/agent/official_graph.py)、[`official_read_tools.py:236–253`](../../app/agent/official_read_tools.py)，已有[`test_official_tool_normalization.py`](../../tests/test_official_tool_normalization.py)验证已知超时映射。`prepare_meeting` 和可选的 `search_memory` 不在 ToolErrorMiddleware 列表，内部未处理异常会传播；本轮**没有对这两项做实际故障注入**，不能据此断言用户端已经出现缺失回复。会议准备的正常 PG 图路径见[`test_pg_official_meeting.py`](../../tests/integration/postgres/test_pg_official_meeting.py)，本轮未重跑。工具 Schema 参数错误由框架绑定/校验，不等同远端网络故障，也没有作为本轮“已进入工具函数”的异常统计。

成功的 `run_agent` 回复仍须经入口持久化与投递：Telegram 的 Receipt/Outbox 见[`durable_runtime.py:114–165`](../../app/telegram/durable_runtime.py)，WebRequest 见[`runs.py:392–416,495–530`](../../app/web/runs.py)。普通未捕获异常在 Telegram 记录 Receipt 失败等待恢复；Web Worker 对非特定权限异常记录日志并保留租约恢复，不立即生成准确业务回复。**本轮所有“用户可见结果”均未实际向 Telegram 发送或在 Web UI 展示**；表中脚本回复只是图返回值。

## 5. 运行记录与限制

1. 新增 [`test_pg_official_runtime_failure_probe.py`](../../tests/integration/postgres/test_pg_official_runtime_failure_probe.py)：`run_preflight_verified_local.ps1 -OutputName runtime_tool_matrix_final_20261104 -PytestTarget tests/integration/postgres/test_pg_official_runtime_failure_probe.py`，**12 passed**；[原始日志](../../experiments/official_agent_eval_20261020/runtime_tool_matrix_final_20261104/pytest.log)。包括 10 个读取故障参数、1 个同批未知异常与 Todo 已提交、1 个 Todo“未找到但 success”案例。
2. 新增 [`test_official_runtime_write_failure_probe.py`](../../tests/test_official_runtime_write_failure_probe.py)：当前最终聚焦运行 **5 passed**，覆盖批准后再读失败、创建/修改/删除 403 和未料发送异常。相关 R5.2/R5.3、ToolError、读取重试、提醒及工具清单单元回归使用仓库固定 LangChain 1.4.6 路径运行，**75 passed，1 个依赖弃用警告**。命令：`$env:PYTHONPATH="$(Resolve-Path .local-tools\official-langchain-1.4.6);$(Resolve-Path .venv312\Lib\site-packages);$(Get-Location)"; python -m pytest -q -o addopts='' -p no:cacheprovider tests/test_official_runtime_write_failure_probe.py tests/test_official_calendar_error_boundary.py tests/test_official_read_retry.py tests/test_official_guarded_tool_feedback.py tests/test_r5_2_create_reconciliation.py tests/test_r5_3_calendar_modify.py tests/test_official_reminder_authorization.py tests/test_official_tool_normalization.py`。**5 项包含在 75 项内，不能相加**。
3. 隔离 PG Todo 事务账本：**11 passed**，[日志](../../experiments/official_agent_eval_20261020/runtime_todo_ledger_20261104/pytest.log)；提醒完整图：**5 passed**，[日志](../../experiments/official_agent_eval_20261020/runtime_reminder_20261104/pytest.log)。OA11 完整图/同身份恢复 **18 passed**、双审批 **6 passed**、Gmail＋独立 Todo **1 passed**为同一工作树此前已存的隔离库回归：[OA11](../../experiments/official_agent_eval_20261020/preflight_oa11_regression_20261104/pytest.log)、[审批](../../experiments/official_agent_eval_20261020/preflight_approval_regression_20261104/pytest.log)、[Gmail](../../experiments/official_agent_eval_20261020/preflight_gmail_regression_20261104/pytest.log)；本轮复用，未把它们冒充新运行。
4. 首次单元命令只载入本机默认依赖，出现 **67 passed、3 failed**，三个失败均为 `HumanInTheLoopMiddleware.__init__()` 不接受本地版本的 `interrupt_mode`，图尚未执行工具。改用项目固定 LangChain 路径后同批 **70 passed**，再加入新探针最终 **75 passed**。这是测试环境版本错误，**不是生产工具故障**，失败记录未隐藏。PG 脚本产生的 pytest 缓存权限警告不影响已执行断言。

本轮没有真实 Qwen、真实 Google、Telegram 或 Web UI 验收；Calendar 403 写入探针使用本地 SQLite＋Fake，证明当前协调器的分支与账本状态，**不能推断所有真实 Google 403 响应都具有同样包装类型**。不把不同批次通过数相加为整体覆盖率。

## 6. 直接结论

**已经合理：**已知 Calendar/Gmail 读取故障有类型化分类、有限重试、脱敏且配对的 ToolMessage；正常空结果与失败区分；独立 Todo 在已处理只读故障下能够保留；Calendar 超时与 409/412 的写入对账有明确副作用边界；Todo 提交后同身份重放、批准版本冲突和提醒授权已有回归。

**真实缺陷：**合成且类型化的 Calendar 写入 403 在创建、更新、删除三条当前协调器路径均被发送两次，并最终误列“待核实”；批准后创建日程的再读空闲失败没有配对 ToolMessage；Todo 目标不存在的业务失败可被标为成功 ToolMessage。前两项优先级更高，最小修复只需各改现有局部分支，保留超时/未知结果对账和原审批。

**仅缺用户反馈验证：**已处理错误的真实 Qwen 终答是否总与业务状态一致、Calendar 写入 `UNCONFIRMED` 在 Telegram/Web 的实际文案和投递、会议/记忆工具未知异常的最终用户体验。本轮没有发送真实消息，因此不能把图内 ToolMessage 或脚本回复写成用户已收到。
