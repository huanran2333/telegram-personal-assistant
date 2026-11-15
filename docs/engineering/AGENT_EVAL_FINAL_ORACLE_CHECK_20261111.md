# Agent Evaluation：Todo 时间 Oracle 与失败标签最终核查

日期：2026-11-11。本轮只修改新版离线 Replay 的确定性评分；生产 `create_agent`、旧 `scoring.py/scoring_v2.py`、H3/U 原始 JSON、冻结协议、旧 Trace 与人工裁定均未改动。所有执行是历史合成轨迹的本地 Replay，没有模型、PostgreSQL、Google 或 Telegram 调用。

## 发现与证据

1. **U05 可误通过缺失或多余的时间。** 该案的用户要求先取得 45 分钟日历候选，再为 Todo 保存计划窗、不设提醒。[冻结协议](../../experiments/agent_business_baseline_20261103/FROZEN_PROTOCOL.md)要求创建时间与同轮成功候选完全一致。原 Trace 的成功候选含 10:00–10:45 与 11:00–11:45，实际 Todo 为 10:00–10:45。修改前对原 Trace 的内存副本做三种篡改：清空计划起止、额外填 `due_date`、额外填 `reminder_at`，旧 `evaluate()` 均给 `success`；只把结束时间改错时才给 `task_failure`。原因是 `create` 对非 `untimed` 目标仅查标题和数量，候选匹配又只在计划开始非空时触发，而且一旦有 `order` 就放过所有时间字段。
2. **带说明要求的创建也可能误通过。** H3-03 和 H3-09～12 明确要求说明包含已核实内容；旧 `scoring.py` 有 `description_contains` 检查，新 Replay 原先没有。H3-03 原始轨迹通过，把其新建 Todo 的说明在内存中清空后，新 Oracle 明确判失败。历史评分器及其原结果无需改动。
3. **失败标签超出证据。** 旧 Replay 将“有错误 ToolMessage 且目标未完成”称为 `tool_error` 原因，将“无错误消息但目标未完成”称为 `business_state_or_tool_selection`。U13 未删除这一事实不能单靠该条件确定为模型工具选择错误；U05 有两次 Schema 错误但最终完成，错误消息本身也不是任务失败的充分条件。普通 `status=ERROR` 原来还会在没有阶段证据时被归为 `model_error`。

原评分合同的职责需分开看：H3 的普通 `create` 案例均带 `untimed`，旧 `scoring.py` 检查四个时间字段为空和说明要求；U05 的动态排期属于运行前冻结的人工特例，旧 v2 对它有已记录误判。此次修正只面向**新版 Replay**，并不重写历史判断。

## 最小修正

在 `agent_benchmark/pipeline.py`：

- `create` 先查新行数量/owner及明确说明要求。`untimed` 必须四个时间字段全空。若案例**结构化**列出具体时间字段，就逐字段比较；未列出的额外时间缺乏判分依据时标 `undetermined`，不猜测。
- 对 U05 这类明确要求 `calendar_candidate_slots` 的动态排期，计划开始与结束都必须存在，并同时匹配**创建调用以前、同一 Todo 标题的成功候选 ToolMessage**；不能拿另一目标的候选、失败结果或后到结果充当依据。截止时间和未授权提醒不允许顺带写入。候选内容不可解析时，创建时间目标标为待定；若同时违反了明确的先读后写约束，整项任务仍判失败。已知候选不匹配或根本没有计划窗时判失败。
- 提醒更新只允许改变提醒及相应版本；若说明、状态、owner、标题、截止或计划时间也变了，标非预期副作用。
- `reasons` 现在描述可复核的**未通过条件**，`observations` 单列工具错误/副作用/依赖或配对违例；`failure_causes` 在无法从轨迹唯一确定根因时为 `root_cause_not_determined_from_trace`。普通未完成的 `ERROR` 状态不再无依据归为模型错误。报告表头也改为“评分依据”，避免把现象误写成原因。
- 新报告 manifest 保存 Replay 评分器文件 SHA256，便于与旧报告区分版本。

四个时间字段的判分边界：`due_date` 仅在案例明确要求截止时与结构化 Ground Truth 比较；`planned_start_at`/`planned_end_at` 在明确排期时成对检查，并在动态排期中对照实际候选；`reminder_at` 在明确提醒目标中单独比较，明确“不设提醒”时必须为空。H3 的无时间案例四项均须为空。U05 没有固定的预期钟点，不能写死 10:00；本次使用其真实候选作为时间依据。候选所查询的日期范围是否忠实于用户自然语言、45 分钟是否与请求语义一致，仍须依据冻结协议人工复核，不能由标题与候选匹配自动证明。

## 回归与前后对照

| 内存中构造的评分探针 | 修改前 | 修改后 |
| --- | --- | --- |
| U05 原始正确计划 10:00–10:45 | `success` | `success` |
| U05 标题不变、计划时间清空 | `success`，误通过 | `task_failure: CreateMissingPlannedWindow` |
| U05 标题不变、结束时间改错 | `task_failure` | `task_failure: CreatePlannedWindowOutsideCandidate` |
| U05 额外截止时间或提醒 | 两者均 `success`，误通过 | 两者均 `task_failure`，并记录非预期时间写入 |
| U05 候选工具参数指向另一 Todo | 可能借用候选 | `task_failure`，不能作为该 Todo 的依据 |
| 移除结构化时间 Ground Truth 的教学用例 | 原先可能按标题通过 | `undetermined` |
| H3-03 说明被清空 | 新 Replay 原先按标题通过 | `task_failure: CreateDescriptionMismatch` |

定向测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_agent_benchmark_pipeline.py -q
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_official_eval_scoring.py experiments/official_agent_eval_rebuild_20261023/test_scoring_v2.py -q
```

结果分别为 **16 passed、1 条弃用警告**和 **7 passed、2 条依赖/弃用警告**。测试覆盖正确时间、缺起止、错结束、额外截止/提醒、同名目标绑定、无可靠 Ground Truth、明确截止、H3 说明、提醒顺带改说明、U05/U06/U11/U13 原始边界、失败标签及旧评分器。

新输出目录：[U16 Replay 与同集配对](../../experiments/agent_benchmark_oracle_check_20261111/u16/REPORT.md)、[H3 首轮旧轨迹 Replay](../../experiments/agent_benchmark_oracle_check_20261111/h3_24/REPORT.md)。U16 修改前后均为 **11 自动确定性通过、4 任务失败、1 待定**；同集 16 条配对的 `outcome` 改变 **0** 条。U11/U12/U15 的错误 ToolMessage 转成观察事实，U13 不再自动标为“模型工具选择”；它们的任务失败事实未变。H3 新 Replay 为 **18 自动确定性通过、6 失败**，这是另一组和另一模型版本的旧轨迹，不能与 U16 合并，也不能替代 H3 历史人工严格口径。所有新测试中的篡改均发生在内存副本，没有修改原始 Trace。以上是**评分器修正**，没有新的模型表现变化。

## 保留的人工边界与结论

U06 的“全忙而无事件标题”仍按旧冻结规则由人工判断，新 Replay 保留 `undetermined`。自然语言回复是否真实完整、动态排期的用户日期/时长语义、远端结果不确定及具体失败根因仍需要人工或更明确的案例 Ground Truth。错误 ToolMessage 不等于最终用户反馈。现有项目级评测已具备冻结案例、隔离真实图 Runner、前后状态与轨迹、人工复核、离线重评及明确分母；本轮确认的自动误判已收敛，**可以结束当前 Benchmark 风格改造**。后续重点应是理解和如实解释现有方法，而非继续扩展平台。
