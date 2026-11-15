# Agent Evaluation：统一 Judge 实施记录

## 结果与边界

正式入口使用 `--report-mode unified`。每条可判案例把同一份脱敏证据包交给一个固定版本 Prompt，最多调用一次 Judge；返回 `task_completion` 和 `fault_handling` 两个独立结论。旧的四维评分及双 Judge 执行入口已从发布副本移除；历史结果仍保留在原报告中。统一模式只调用一个 Judge。`concise` 离线报告仍可用于确定性指标查看。默认 `--judge-mode off`，因此离线 Replay 不访问模型。

证据包复用已有 Ground Truth、按顺序记录的 Tool Call/ToolMessage 及 ID 配对、本地 Todo 前后状态与执行状态、审批证据、最终回复。源运行没有可核实的工具描述与 Schema 快照，明确标为不可用。用户请求和工具返回均作为待分析数据，不能改写评分指令。缺失业务状态或真正的最终回复时，任务语义保持待定。

Prompt 版本为 `unified-task-fault-judge-1`，融合协议为 `oracle-unified-judge-fusion-1`。任务输出 Pass / Fail / Undetermined；故障输出 Reasonable / Unreasonable / Undetermined / NotApplicable。完整轨迹没有错误 ToolMessage 时，程序固定为 NotApplicable；轨迹不完整时不推断没有故障。非法 JSON、引用不存在的证据、请求失败均保守待定。Oracle 已证实业务失败时综合结果始终 Fail，但统一 Judge 仍可独立记录故障处理表现。故障处理结论不参与 Pass Rate。

综合 Pass Rate = 综合 Pass 的有效案例数 / 全部选入的有效案例数；Fail 和 Undetermined 单列。Partial Score 与工具、Token、耗时继续沿用 2A Oracle 和原 Trace。新的报告仅展示这几项指标及实际发生工具错误的案例分析。离线 Replay 不等于新模型实验，评分协议变化不等于 Agent 性能变化。

## 离线验证

执行 87 项相关测试，全部通过，其中统一 Judge 新测试 12 项；使用 Mock 完成函数证明每个案例只有一次调用、U/H3 使用同一 Prompt、Oracle Fail 不能被覆盖、无故障返回 NotApplicable、坏输出和缺失证据不伪造通过。没有实际访问 Judge、Qwen、Telegram、Google 或数据库。

最终版本离线报告分别位于 `experiments/agent_benchmark_unified_20261111/u16_verified` 和 `experiments/agent_benchmark_unified_20261111/h3_24_verified`。U：Oracle 确认失败 4/16，另 12 例综合待定；Partial Score 20/24。H3：Oracle 确认失败 6/24，另 18 例综合待定；Partial Score 32/41。两组 Judge 实际调用均为 0，因此当前综合 Pass Rate 的 0 个确认 Pass 是**待定较多时的通过下界**，不能解释为 Agent 实际成功率 0%。

## 如需真实 Judge 校准

项目 `.env` 中已有 Judge 的地址、模型、密钥和 `AGENT_EVAL_JUDGE_MAX_TOKENS=8192`；程序仅在显式 `--judge-mode live` 时读取并调用，不把密钥写入 Trace 或报告。建议先选 U05、U11 两例，在仓库根目录运行：

```powershell
python -m agent_benchmark --oracle-version project-oracle-2a-1 --report-mode unified --judge-mode live --cases U05,U11 --output experiments/agent_benchmark_unified_calibration_001
```

该命令最多发起两次 Judge 请求，不重新运行被测 Agent。正式调用前须单独确认。按当前配置的硅基流动 DeepSeek-V4.1-Flash 白天价格，输入约 ¥2/百万 Token、输出约 ¥8/百万 Token；成本公式是 `2×输入 Token/百万 + 8×输出 Token/百万` 元。两次请求的输出上限合计 16,384 Token，对应输出费用上限约 ¥0.131，另加实际输入费用；最终以提供商记录的用量和当时价格为准。价格来源：[硅基流动模型价格页](https://www.siliconflow.cn/pricing)。

真实模型尚未执行，也没有以人工样本校准 Judge 的误通过、误失败或 JSON 稳定性。只有完成少量人工对照后，才适合把综合 Pass Rate 作为正式成绩。
