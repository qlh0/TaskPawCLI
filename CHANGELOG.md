# Changelog

## Unreleased

### Added

- Plan 模式失败梯度：子任务重试（默认 3 次，预算耗尽/权限拒绝等确定性失败不重试）→
  失败子图局部重规划（默认 1 次）→ 依赖级联跳过 → 如实汇报。
- 新增 `plan` 配置段：`max_task_retries`、`max_repairs`。
- 新事件：`plan_task_retry`、`plan_repair_started`、`plan_repair_result`、`plan_repaired`。

### Fixed

- 被 LoopGuard 收尾的子任务不再被误记为已完成（现在按失败处理并进入重试/修复）。
- 失败子任务及其重试消耗的 token / turns 现在计入总账。
- Plan 失败时 `stop_reason` 为 `plan_failed`，不再误报 `end_turn`。
- 依赖失败任务的下游任务标记为「已跳过」，最终摘要区分完成 / 失败 / 跳过。

## v0.1.0

首个公开版本（fork 自 [PaiCLI-Python](https://github.com/itwanger/PaiCLI-Python)，MIT）。

### Added

- **LoopGuard 循环治理**：重复动作 / 重复错误连续 3 次提醒、5 次收尾；网络类瞬态错误豁免；
  停滞检测；整轮 token 预算；轮数上限 + 有可见进展时自动扩容。
- `done` 事件新增 `stop_reason`（`end_turn` / `turn_cap` / `hard_cap` / `repeated_action` /
  `repeated_error` / `stagnation` / `budget_exhausted`），CLI 与 REPL 会显式提示提前终止原因。
- GitHub Actions CI：`ruff check`、`ruff format --check`、`pytest`（Python 3.11 / 3.12）。

### Changed

- 项目重命名为 **TaskPawCLI**：包名 `taskpaw`、命令 `taskpaw`、环境变量 `TASKPAW_*`、
  用户数据目录 `~/.taskpaw`、项目配置目录 `.taskpaw/`。
- 合并两份重复的 ReAct 循环实现，统一走 `src/taskpaw/agent/query.py`，Plan / Team 子 Agent 复用同一套守卫。

### Fixed

- Windows 上任务 ID 可能重复（时间戳精度不足），现在附加随机后缀。
