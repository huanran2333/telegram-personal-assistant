# 离线 Agent Evaluation 流水线

这套包对**既有生产 `create_agent` Runner 产生的原始轨迹**做数据校验、Replay、确定性评分和报告。Replay 不调用模型、PostgreSQL、Telegram 或 Google；不能把报告称为新的模型实验。真实 Runner 仍在 `experiments/official_agent_eval_rebuild_20261023/test_current_eval.py`，隔离 PG18 与真实 Qwen 的启动只通过它已有的 `run_verified.ps1`，须另外取得模型调用授权并核验测试数据库身份。

## 当前推荐：统一报告

当前评测以 `project-oracle-2a-1`、`--report-mode unified` 为推荐组合：报告展示综合 Pass Rate、原子目标 Partial Score、工具调用、Token 与耗时，并对有工具错误的案例单列处理分析。统一 Judge 在一次调用中独立判断任务完成语义与故障处理；确定性 Oracle 已证明的业务失败不能被 Judge 改判通过。默认 `--judge-mode off` 不调用模型，未经 Judge 判断的语义结论保持待定。公开副本未提供私人原始轨迹，实际 Replay 须另行提供与数据集指纹匹配的 Trace。

## 历史 Replay 与数据适配

以下命令展示旧版回放入口；输出目录必须不存在，且需要提供对应的原始轨迹：

```powershell
.\.venv\Scripts\python.exe -m agent_benchmark --cases U01,U09,U11 --output experiments/agent_benchmark_example_1 --mine
.\.venv\Scripts\python.exe -m agent_benchmark --case U05 --output experiments/agent_benchmark_example_2
.\.venv\Scripts\python.exe -m agent_benchmark --category calendar_read --output experiments/agent_benchmark_example_3
.\.venv\Scripts\python.exe -m agent_benchmark --all --output experiments/agent_benchmark_example_4
```

可用 `--dataset`、`--traces` 指定同版本数据和原始轨迹目录；每份 Trace 必须带匹配的数据集 SHA256。`--compare 旧目录/results.json` 仅在 SHA256 完全相同时生成同案例配对状态，缺失的 Case 不会伪装为成败。输出 `results.json`、`results.jsonl`、`results.csv` 和 `REPORT.md`；`--mine` 另写 `failure_candidates.json`。它只含待人工复核的失败模式与少量确定性标题变体，初始世界与 Ground Truth 必须重新审核，**不会写回冻结数据集**。

自动评分只检查可确定的状态、工具结果配对、明确要求的依赖顺序与越界写入。自由文本回复统一标 `manual_review_required`；复杂占用、冲突、邮件内容等语义不足以自动判断时标 `undetermined`。原始任务失败与故障反馈正确是两个不同指标。2026-11-03 的 16 例历史人工结论仍以冻结协议与原报告为准；本工具的 Replay 自动数值不能替换它。

数据集适配保留每条旧案例和世界配置，并增加 Case ID、版本、来源、类别、目标、约束、标签与源文件指纹。旧档没有记录的生成模型、难度和审核者不会补造。完全重复的 ID/请求直接拒绝；疑似近重复只列人工复核，不自动删案例。更改规则时创建新数据集版本，不修改旧 JSON、原始 Trace 或旧评分合同。

## 方法诊断

`python -m agent_benchmark.methodology --output <一个尚不存在的 Markdown 文件>` 对 H3 的 24 条和 U 的 16 条**冻结案例**生成能力、设计难度、设计必要步骤、负面约束及跨集字面相似度表。标签由当前规则推导，不是历史原始标签；设计步骤也不是模型实际调用次数。该命令不会启动 Runner。参考 [方法说明](../docs/engineering/AGENT_EVALUATION_METHOD_AND_DESIGN_20261111.md)；输出目录和 Replay 一样须选择新路径，避免覆盖历史记录。

Replay 报告另列请求与原子目标的分母、不可判定数、资源数据覆盖、故障注入和非预期写入。`success` 仍只代表可自动判定的部分；最终回复真实性须按冻结规则人工复核。

## 阶段 2A：可选 Ground Truth / Oracle 版本

默认 `legacy-replay` 仍按原规则判分。离线复核须显式选择 `project-oracle-2a-1`，并写入新目录：

```powershell
.\.venv\Scripts\python.exe -m agent_benchmark --case U06 --oracle-version project-oracle-2a-1 --output experiments/replay_2a_u06
```

Ground Truth 视图只读复用原始请求、初态、目标、许可写入、顺序与审批决定。U05/U06/U15/U11/U12 的补充裁定仅在数据集及首次运行前冻结协议的 SHA256 匹配时应用；H3-24 的邮件内容禁令来自原始请求。结果标明独立 Oracle、Ground Truth 版本和源码指纹。不同 Oracle 版本不得用 CLI 配对比较解释为模型表现变化。

新版可核对 U05 候选与 45 分钟计划窗、U06 的可信全忙证据、U15 的提醒字段及其余字段不变；对 H3-24 的未知说明来源和最终回复语义保留人工复核。U11/U12 分开记录独立 Todo 和失败的外部读取。`success` 仍只表示**确定性目标与安全检查**通过；最终回复仍需人工复核，不等于严格任务总成功。
