# H3 与 U 冻结集：设计覆盖诊断

标签是本轮从已冻结目标、审批决定、故障注入和允许写入清单推导的诊断分类；不是历史原始标签，也不是模型实际执行步骤。
预期步骤 = 声明的原子目标数 + 人工审批决定数；不含模型内部循环、工具重试与最终回复。
文章的 4–8 与 8+ 在第 8 步重叠；本表用 1–3、4–8、9+ 三个互斥区间。
难度规则：多审批、显式读后写依赖、故障与独立目标并存为困难；其余多目标、单审批、单故障、负面约束或提醒为中等；其余为简单。难度按业务约束而非运行结果划分。

## current-create-agent-stratified-holdout-v3（24 例）

来源 `cases_holdout_v3.json`；SHA256 `2548a702f6151d88089b57e5538629e8a6759b33459e2005f5203e66c3fedfb1`。
难度：简单 1/24、中等 15/24、困难 8/24。文章的 30/50/20 仅供检查，不用于改写冻结集。
设计步骤：1–3 步 22/24、4–8 步 2/24、9+ 步 0/24。
负面约束 7/24；禁止所有写入 10/24；注入故障 4/24。

| Case | 原始类别 | 能力维度 | 难度 | 必要步骤 | 长度 | 负面 |
| --- | --- | --- | --- | ---: | --- | --- |
| H3-01 | independent_writes | Todo 创建、多目标 | 中等 | 2 | 1–3 | 否 |
| H3-02 | independent_writes | Todo 创建、多目标 | 中等 | 3 | 1–3 | 否 |
| H3-03 | independent_writes | Todo 创建 | 简单 | 1 | 1–3 | 否 |
| H3-04 | independent_writes | Todo 创建、多目标 | 中等 | 2 | 1–3 | 否 |
| H3-05 | cross_domain_reads | Calendar 读取、Gmail 读取、Todo 查询、多目标、禁止多余写入 | 中等 | 3 | 1–3 | 否 |
| H3-06 | cross_domain_reads | Calendar 读取、Gmail 读取、多目标、禁止多余写入 | 中等 | 2 | 1–3 | 否 |
| H3-07 | cross_domain_reads | Calendar 读取、Todo 查询、多目标、禁止多余写入 | 中等 | 2 | 1–3 | 否 |
| H3-08 | cross_domain_reads | Calendar 读取、Gmail 读取、多目标、禁止多余写入 | 中等 | 2 | 1–3 | 否 |
| H3-09 | read_before_write | Calendar 读取、Todo 创建、多目标、跨工具依赖 | 困难 | 2 | 1–3 | 否 |
| H3-10 | read_before_write | Gmail 读取、Todo 创建、多目标、跨工具依赖 | 困难 | 2 | 1–3 | 否 |
| H3-11 | read_before_write | Calendar 读取、Todo 创建、多目标、跨工具依赖 | 困难 | 2 | 1–3 | 否 |
| H3-12 | read_before_write | Gmail 读取、Todo 创建、多目标、跨工具依赖 | 困难 | 2 | 1–3 | 否 |
| H3-13 | approval | HITL、Todo 删除/审批 | 中等 | 2 | 1–3 | 否 |
| H3-14 | approval | HITL、Todo 删除/审批、禁止多余写入 | 中等 | 2 | 1–3 | 是 |
| H3-15 | approval | HITL、Todo 删除/审批、多目标、禁止多余写入 | 困难 | 4 | 4–8 | 是 |
| H3-16 | approval | HITL、Todo 删除/审批、多目标、禁止多余写入 | 困难 | 4 | 4–8 | 是 |
| H3-17 | negative_authorization | Todo 查询、禁止多余写入 | 中等 | 1 | 1–3 | 是 |
| H3-18 | negative_authorization | Todo 创建、禁止多余写入 | 中等 | 1 | 1–3 | 是 |
| H3-19 | negative_authorization | Calendar 读取、禁止多余写入 | 中等 | 1 | 1–3 | 是 |
| H3-20 | negative_authorization | Todo 创建、禁止多余写入 | 中等 | 1 | 1–3 | 是 |
| H3-21 | fault_handling | Calendar 读取、Todo 创建、多目标、故障反馈 | 困难 | 2 | 1–3 | 否 |
| H3-22 | fault_handling | Calendar 读取、故障反馈、禁止多余写入 | 中等 | 1 | 1–3 | 否 |
| H3-23 | fault_handling | Calendar 读取、故障反馈、禁止多余写入 | 中等 | 1 | 1–3 | 否 |
| H3-24 | fault_handling | Gmail 读取、Todo 创建、多目标、故障反馈 | 困难 | 2 | 1–3 | 否 |

## business-unseen-v1（16 例）

来源 `cases_unseen_20261103.json`；SHA256 `58856ba3b47b3461a884c31556a33a7d7b8ce8c219f58b8b4c1dd3ac6d432f36`。
难度：简单 3/16、中等 8/16、困难 5/16。文章的 30/50/20 仅供检查，不用于改写冻结集。
设计步骤：1–3 步 15/16、4–8 步 1/16、9+ 步 0/16。
负面约束 5/16；禁止所有写入 4/16；注入故障 2/16。

| Case | 原始类别 | 能力维度 | 难度 | 必要步骤 | 长度 | 负面 |
| --- | --- | --- | --- | ---: | --- | --- |
| U01 | todo_single | Todo 创建 | 简单 | 1 | 1–3 | 否 |
| U02 | todo_read | Todo 查询、禁止多余写入 | 简单 | 1 | 1–3 | 否 |
| U03 | todo_multi | Todo 创建、多目标 | 中等 | 3 | 1–3 | 否 |
| U04 | calendar_read | Calendar 读取、禁止多余写入 | 简单 | 1 | 1–3 | 否 |
| U05 | schedule | Calendar 读取、Todo 创建、多目标、跨工具依赖 | 困难 | 2 | 1–3 | 否 |
| U06 | calendar_conflict | Calendar 读取、禁止多余写入 | 中等 | 1 | 1–3 | 是 |
| U07 | negative_target | Todo 创建、禁止多余写入 | 中等 | 1 | 1–3 | 是 |
| U08 | readonly_negative | Calendar 读取、禁止多余写入 | 中等 | 1 | 1–3 | 是 |
| U09 | independent_cross_domain | Calendar 读取、Todo 创建、多目标 | 中等 | 2 | 1–3 | 否 |
| U10 | gmail_dependency | Gmail 读取、Todo 创建、多目标、跨工具依赖 | 困难 | 2 | 1–3 | 否 |
| U11 | calendar_failure | Calendar 读取、Todo 创建、多目标、故障反馈 | 困难 | 2 | 1–3 | 否 |
| U12 | gmail_failure | Gmail 读取、Todo 创建、多目标、故障反馈 | 困难 | 2 | 1–3 | 否 |
| U13 | approval_approve | HITL、Todo 删除/审批 | 中等 | 2 | 1–3 | 否 |
| U14 | approval_mixed | HITL、Todo 删除/审批、多目标、禁止多余写入 | 困难 | 4 | 4–8 | 是 |
| U15 | reminder_set | 提醒写入 | 中等 | 1 | 1–3 | 否 |
| U16 | no_duplicate | Todo 创建、禁止多余写入 | 中等 | 1 | 1–3 | 是 |

## 两套数据的独立性检查

比较去掉案例编号、引号内标题和日期后的请求字面形状；相似度 ≥0.75 的配对仅供人工检查，不能据此判定数据泄漏。不同来源之间的语义同类任务也可能合理重复。
需人工查看的形状相似配对：0 组。

没有记录生成模型、独立审核者或训练数据重叠检查；不能宣称完全无污染。冻结后的开发调试会使已见案例丧失新的未见资格。
