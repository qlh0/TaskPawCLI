# Contributing

## 开发环境

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --extra dev
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
```

## 提交前自检

- 新增或修改的行为要有测试（`tests/`，pytest；全部离线，用 FakeClient 驱动，不需要真实 API Key）
- `ruff check .` 和 `ruff format --check .` 必须通过
- 提交信息用 `type: summary`，例如 `feat: add word_count tool`、`fix: avoid duplicate task ids`

## 目录结构

| 路径 | 说明 |
|---|---|
| `src/taskpaw/agent/` | Agent 循环（ReAct / Plan / Team）、`loop_guard.py` 循环治理 |
| `src/taskpaw/tools/` | 工具定义、注册表、执行器、策略与审计 |
| `src/taskpaw/llm/` | OpenAI-compatible 客户端、模型配置、计价 |
| `src/taskpaw/prompt/` | 静态前缀 + 动态后缀的 prompt 组装 |
| `src/taskpaw/memory/` `skill/` `snapshot/` `mcp/` `runtime/` | 记忆、技能、快照、MCP、Runtime API |
| `tests/` | 单元测试与集成测试 |

## 配置

运行时配置来自 `TASKPAW_*` 环境变量、`~/.taskpaw/config.json` 或项目级 `.taskpaw/config.json`，
优先级见 README 的「配置」一节。`.env` 与 `.taskpaw/` 已在 `.gitignore` 中，不要提交密钥。
