# TaskPawCLI

[![CI](https://github.com/qlh0/TaskPawCLI/actions/workflows/ci.yml/badge.svg)](https://github.com/qlh0/TaskPawCLI/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

TaskPawCLI（命令 `taskpaw`）是一个运行在终端里的 AI Agent CLI，面向真实项目开发场景：读写文件、搜索代码、执行命令、联网检索、调用 MCP 工具、保存记忆、生成快照、恢复现场，并通过 Runtime API 对外提供线程、turn、事件和后台任务能力。

## 功能特性

- 交互式终端 Agent，基于 Rich 和 prompt-toolkit 渲染
- 单次 prompt 模式，适合脚本、管道和自动化调用
- OpenAI-compatible 流式 LLM 客户端，默认面向 DeepSeek 配置
- 支持 `DEEPSEEK_API_KEY` 等 provider-specific API Key
- ReAct 工具调用循环，支持 thinking、tool call、tool result、final output 和 usage 事件
- Plan-and-Execute 模式，使用独立 Planner 生成 DAG，并按依赖批次执行可并行任务
- Multi-Agent 协作模式，包含 Planner、Worker、Reviewer、依赖调度、并行 worker、review
  重试，以及可切换到独立 Plan-and-Execute 的子 Agent
- 内置文件、Shell、grep、glob、记忆、网页搜索、网页抓取、代码搜索等工具
- HITL 人工确认、命令/路径安全策略和 JSONL 审计日志
- MCP client，支持 stdio 和 Streamable HTTP MCP server
- Skill 系统，支持内置、用户级和项目级分层、输入 Top-K 匹配、`load_skill` 当前回合懒加载，
  以及经 HITL 确认的 `save_skill` 流程沉淀
- Chrome DevTools MCP 配置助手
- TaskPaw 自身也可以作为 MCP server 暴露内置工具
- Runtime API，支持有历史的 thread、turn、事件日志，以及带原子抢占、租约恢复、取消保护、
  项目隔离和 `react|plan|team` 模式的持久化后台任务
- 静态项目记忆 + SQLite 动态长期记忆，支持元数据、去重、TTL、容量治理和相关性召回
- 上下文预算与压缩：达到可用输入预算的 80% 后压缩旧轮次，保留近期消息和完整工具调用对
- 完整 usage、缓存命中/未命中 Token、reasoning Token 和可配置成本估算
- Agent run 前后自动创建快照，支持恢复现场
- 支持本地图片和远程图片输入，并根据模型能力自动降级

## 环境要求

- Python 3.11 或更新版本
- [uv](https://docs.astral.sh/uv/)
- 可选：`rg`，用于更快的本地搜索
- 可选：Chrome DevTools MCP 需要 Node.js 20.19.0 LTS 或更新版本、npm/npx 和 Chrome

## 快速开始

```bash
git clone https://github.com/qlh0/TaskPawCLI.git
cd TaskPawCLI
uv sync --extra dev
uv run taskpaw --help
```

启动交互模式：

```bash
uv run taskpaw
```

单次查询：

```bash
uv run taskpaw -p "帮我总结这个项目"
```

选择运行模式并输出机器可读的 usage/cost：

```bash
uv run taskpaw --mode plan -p "先读取 README，再验证项目" --json
uv run taskpaw --mode team --worker-mode plan -p "并行审计核心模块" --json
```

检查当前环境：

```bash
uv run taskpaw doctor --cwd .
```

## 配置

TaskPaw 的配置优先级如下：

1. 内置默认配置
2. `~/.taskpaw/config.json`
3. 项目级 `.taskpaw/config.json`
4. 项目级 `.env`
5. CLI 参数
6. 当前进程环境变量

可以像 Java 项目一样，把 DeepSeek Key 写到项目 `.env` 里：

```dotenv
TASKPAW_PROVIDER=deepseek
TASKPAW_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=your_key_here
```

也可以使用 TaskPaw 通用 Key：

```dotenv
TASKPAW_PROVIDER=deepseek
TASKPAW_MODEL=deepseek-v4-flash
TASKPAW_API_KEY=your_key_here
```

当前支持的 provider-specific API Key 包括：

- `DEEPSEEK_API_KEY`
- `ZAI_API_KEY`（GLM 官方推荐）
- `GLM_API_KEY`
- `STEP_API_KEY`
- `KIMI_API_KEY`

通过命令行临时覆盖 provider 和 model：

```bash
uv run taskpaw --provider deepseek --model deepseek-v4-flash
```

连接本地 OpenAI-compatible 服务：

```bash
TASKPAW_PROVIDER=openai-compatible \
TASKPAW_BASE_URL=http://127.0.0.1:11434/v1 \
TASKPAW_MODEL=qwen2.5-coder \
uv run taskpaw -p "解释这个仓库"
```

## 交互命令

进入 `uv run taskpaw` 后，可以使用这些 slash commands：

```text
/help
/exit
/clear
/context
/memory
/memory search <query>
/memory stats
/memory delete <id>
/memory clear
/save <fact>
/config
/tools
/hitl default|auto
/policy
/audit [N]
/index [path]
/search <query>
/plan <task>
/team <task>
/team --plan <task>
/model
/model <model-id>
/model <provider> <model-id>
/usage
/skill
/skill list
/skill show <name>
/skill on <name>
/skill off <name>
/skill reload
/mcp
/task
/task add [--mode react|plan|team] <task>
/task cancel <task_id>
/task log <task_id>
/snapshot
/snapshot clean
/restore <snapshot-id-or-index>
```

`/model` 会打开交互式模型选择器：`Tab` 或左右方向键在 `Default`、`Custom`
之间切换，上下方向键选择模型，`Enter` 立即切换当前 Agent。`Custom` 中可以选择已保存的
BYOK 模型、创建新的 DeepSeek/GLM/OpenAI-compatible 配置，或按 `d` 删除配置。自定义配置
保存在权限为 `0600` 的 `~/.taskpaw/models.json`；建议填写 API Key 环境变量名，只有显式输入
API Key 时才会把密钥写入该文件。

## 内置工具

TaskPaw 内置了一组 Agent 可以调用的本地工具和联网工具：

- `read_file`
- `write_file`
- `list_dir`
- `glob` / `glob_files`
- `grep` / `grep_code`
- `bash` / `execute_command`
- `web_search`
- `web_fetch`
- `save_memory`
- `update_memory`
- `search_memory`
- `load_skill`
- `save_skill`
- `search_code`
- `revert_turn`

写文件、执行命令、远程 MCP 写操作、恢复快照等危险动作，会经过 policy、HITL 和 audit 处理。
`save_skill` 也必须经过 HITL；模型可以提议沉淀，但不会静默改变后续行为。

交互模式下按 `Shift+Tab` 可在两种会话权限模式间切换：

- `Default`：使用启动时的 HITL、工作区路径和命令安全策略。
- `Auto (full access)`：当前会话内不再请求审批，并关闭路径与命令守卫；再次按
  `Shift+Tab` 会恢复启动时的默认策略。

## Skill 匹配与沉淀

Skill 按 `builtin -> user -> project` 加载，同名时后层覆盖前层：

- builtin：产品默认能力
- user：`~/.taskpaw/skills/*/SKILL.md`，跨项目复用
- project：`.taskpaw/skills/*/SKILL.md`，最贴近当前仓库并拥有最高优先级

每次用户输入先用 name、description、tags 做中英文词法/字符 n-gram Top-K 匹配，再把候选交给模型决定是否调用 `load_skill`。Skill 正文只在真正加载后进入当前 ReAct 的下一模型轮；每个并发子 Agent 都有独立 Skill 缓冲区，不会串线。

当一次成功流程具备稳定输入、明确步骤和可复用边界时，模型可以调用 `save_skill` 提议写入 project 或 user 层。该工具默认拒绝覆盖已有 Skill，并强制人工确认。

## 记忆、动态 Prompt 与上下文压缩

TaskPaw 把记忆分成三层：

- 短期记忆：当前 thread/session 的原始消息、工具调用和工具结果
- 静态长期记忆：`AGENTS.md`、`TASKPAW.md`、`.taskpaw/TASKPAW.md` 及自定义 prompt 文件；人工维护、可版本控制
- 动态长期记忆：按项目 scope 隔离的 SQLite 记录；包含 kind、source、importance、confidence、TTL、访问次数和内容哈希

动态记忆不会再无条件取“最近 8 条”。每个请求会按当前问题自动召回 Top-K，并把结果放进明确标注为 untrusted data 的动态 Prompt；模型觉得候选不足时，还可以调用 `search_memory` 深搜。写入端会拒绝空值/超长值，通过规范化哈希去重，并按项目容量淘汰低价值记录。

记忆支持按 id 原地更新：`update_memory` 会保留原 id 与创建时间，只更新内容、哈希和更新时间；如果新内容与同一项目下的另一条记忆哈希相同，会把对方合并进来（取较高重要度/置信度、较早创建时间、累加访问次数）并删除重复行。当用户表达修正意图（“改为”“不再使用”“之前说错了”等）时，模型应先 `search_memory` 定位旧记忆，再 `update_memory`，而不是新增一条重复记录。

## 向量检索（可选）

装好可选依赖后，长期记忆会从“纯关键词召回”升级为“关键词 + 向量”的混合召回：

```bash
uv sync --extra dev --extra embeddings
```

在 `.taskpaw/config.json` 打开：

```json
{
  "embeddings": {
    "enabled": true,
    "backend": "fastembed",
    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "dim": 384,
    "min_similarity": 0.25
  }
}
```

- 本地 ONNX 推理（fastembed），不依赖 torch，模型首次使用时下载到 `~/.taskpaw/models`；
- 写入/更新/删除记忆时同步维护向量，`sqlite-vec` 可用时用它做近邻检索，不可用时自动退回纯 Python 余弦；
- 两路召回用 RRF 融合，去重后取 Top-K；向量不可用或模型变化时自动降级为关键词召回；
- 换模型后需要重建：`taskpaw memory reindex --cwd .`，`taskpaw memory stats` 可查看状态。

注意：`uv sync` 是精确同步，只写 `--extra dev` 会把 embedding 依赖卸掉。

## 代码库 RAG

代码索引现在按"块"而不是"行"组织：

```bash
taskpaw index --cwd .            # 增量索引（文件哈希没变就跳过）
taskpaw index --cwd . --rebuild  # 全量重建
```

- **切块**：Python 用 AST 按函数/类/方法切分（类只保留头部，方法单独成块），其他文本文件按空行块切分；单块上限 120 行，超长自动分段；
- **增量**：每个文件记录内容哈希，未变化不重复切块与编码；文件删除后对应块自动清理；
- **检索**：关键词（FTS5，缺失时回退 LIKE）与向量两路召回，RRF 融合后返回 5~8 个块；
- **工具**：`search_code` 返回 `path:起始行-结束行` 和块名（如 `[method PlanExecuteAgent._repair_plan]`），模型能直接定位源码；
- **降级**：没装 embedding 依赖或模型不可用时只走关键词，行为与以前一致。

索引文件在项目的 `.taskpaw/code_index.sqlite3`（已加入 `.gitignore`）。

Prompt 分为可缓存的静态前缀和逐请求重建的动态后缀。静态前缀承载身份、规则和项目指令；动态后缀承载当前时间、cwd、模型、工具以及与当前问题相关的记忆。

可用输入预算按 `context_window - max_output_tokens - reserve_tokens` 计算。默认在该预算的 80% 触发压缩，分两级渐进进行：

1. **清理旧工具结果**：把较早的工具输出替换成占位符（保留消息本体与 `tool_call_id`，协议关系不变），
   最近 `tool_result_keep_rounds` 轮工具交互、以及小于 `tool_result_min_chars` 的结果不动；
   清到目标预算即停止，**不消耗额外模型调用**。
2. **摘要旧对话**：只有清理后仍超预算，才沿用 `_summarize()` 把更早的对话压成一条滚动摘要，
   仍超目标时再压缩摘要长度。

压缩摘要只属于短期会话，不会自动晋升为长期记忆。

相关配置在 `memory` 段：

```json
{
  "memory": {
    "compression_threshold": 0.8,
    "compression_target": 0.55,
    "tool_result_clear_enabled": true,
    "tool_result_keep_rounds": 2,
    "tool_result_min_chars": 120
  }
}
```

## 循环保护

ReAct 循环由 `LoopGuard` 守护，避免死循环、空转和静默截断：

- 重复动作 / 重复错误：连续 3 次提醒（提醒会作为消息写回上下文，并给出替代策略），
  5 次进入收尾；网络超时、429、5xx 等可恢复错误不计入重复。
- 停滞：连续多轮没有新信息、没有文件变更、没有成功命令时提醒，继续空转则收尾。
- Token 预算：整轮 run 的累计 token 达到 80% 提醒、100% 收尾；预算默认取
  `context_window × 1.5`，可用 `loop.max_run_tokens` 覆盖。
- 轮数：默认 20 轮。到达上限时，如果本轮有可见进展（写入成功、命令成功）且预算充足，
  自动扩容 `+20` 轮（最多 2 次，硬上限 60）；否则进入收尾。

收尾不是硬截断：模型会被要求禁用工具并基于已有信息给出最终答案，`done` 事件会带上
`stop_reason`（`end_turn` / `turn_cap` / `hard_cap` / `repeated_action` / `repeated_error` /
`stagnation` / `budget_exhausted`），CLI 与 REPL 会把终止原因显式打印出来。

参数可以在项目 `.taskpaw/config.json` 或 `~/.taskpaw/config.json` 的 `loop` 段覆盖：

```json
{
  "loop": {
    "max_turns": 20,
    "grace_turns": 2,
    "extension_turns": 20,
    "max_extensions": 2,
    "hard_cap_turns": 60,
    "repeat_warn": 3,
    "repeat_abort": 5,
    "repeat_window": 6,
    "stagnation_warn": 3,
    "stagnation_abort": 6,
    "max_run_tokens": 0,
    "budget_multiplier": 1.5,
    "budget_warn": 0.8,
    "budget_abort": 1.0
  }
}
```

## Plan 模式：审批与失败处理

**计划审批门**：REPL 里规划完成后会先停下来确认（`y=执行 / n=取消 / e=提修改意见`）。
选 `e` 会把意见带回规划器重新生成计划（默认最多 2 次），选 `n` 则一个任务都不会执行、
`stop_reason=plan_cancelled`。审批门由 `plan.require_approval` 控制，且只在调用方提供了
审批回调时生效——REPL 已内置；`-p` 等非交互路径默认不拦截，通过 SDK 调用时可以自行传入回调。

**子任务验收**：子任务结束时不会直接算"完成"，而是先过两道验收——
① 声明过的 `file:` / `dir:` 资源必须真实存在（确定性检查，不花 token）；
② 一次模型验收调用判断产出是否真的完成了任务描述，不通过则带着验收意见重试。
验收器输出无法解析时按"通过"处理，避免因为解析问题卡住整个计划。可用 `plan.verify_tasks=false` 关闭。

**失败梯度**：失败不是"一失败就结束"，而是：
1. **子任务重试**：失败后带着上一次的错误原因重试，默认最多 3 次；预算耗尽、权限拒绝
   这类确定性失败不会重试。
2. **局部重规划**：重试仍失败时，只把「失败任务 + 它的下游」这一段交给规划器重写，
   已完成的任务保留、结果复用；最多 1 次。
3. **依赖级联跳过**：仍然失败时，依赖它的下游任务标记为「已跳过」，其余独立分支继续执行。
4. **如实汇报**：最终摘要区分「已完成 / 失败 / 已跳过」，`stop_reason` 为 `plan_failed`
   而不是误报 `end_turn`。

被 ReAct 层 `LoopGuard` 收尾的子任务同样按失败处理（不会再被记成"已完成"），
失败尝试消耗的 token 也会计入总账。

配置项在 `plan` 段：

```json
{
  "plan": {
    "max_task_retries": 3,
    "max_repairs": 1,
    "require_approval": false,
    "max_plan_revisions": 2,
    "max_parallel_tasks": 4,
    "verify_tasks": true,
    "keep_history": true,
    "max_history_plans": 50
  }
}
```

## Plan 模式：并行调度与写冲突

规划器会为每个任务声明它会写入/影响的范围（`resources`）：

```json
{"id": "task_2", "type": "FILE_WRITE", "resources": ["file:src/cli.py"], "dependencies": []}
```

支持的取值：`file:<path>`、`dir:<path>`、`workspace`（整个工作区独占）、`service:<name>`。
只读任务可以省略。没声明资源的写任务和命令任务会保守地按 `workspace` 处理。

调度规则：

- **写冲突自动串行**：两个任务资源重叠（含"一个写、一个读同一文件"）时，计划会自动补一条依赖边，
  摘要里会显示 `因资源冲突改为串行：task_2 ← task_1`；
- **并发上限**：`max_parallel_tasks` 控制同时运行的子任务数（默认 4，`0` 表示不限，`1` 表示串行）；
- **不阻塞等待**：调度器只在任务入场时检查资源，不会让运行中的任务等锁，因此不存在死锁。

## Plan 历史：list / show / events

每次 Plan 运行都会把计划、任务状态和关键事件写入 `~/.taskpaw/plans.db`（可用
`plan.keep_history=false` 关闭，`max_history_plans` 控制保留条数）：

```bash
taskpaw plan list                 # 最近的计划：状态 / 任务数 / token / 终止原因
taskpaw plan show <plan_id>       # 每个任务的状态、尝试次数、失败原因
taskpaw plan events <plan_id> --task task_2   # 事件时间线
```

## 模型、Token 与费用

默认 provider/model 是 `deepseek/deepseek-v4-flash`。DeepSeek V4 Flash/Pro 的内置 profile 使用 1M 上下文，并带有截至 2026-07-17 的官方每百万 Token 价格；价格会变化，因此可以用 `llm.context_window` 和 `llm.prices` 覆盖，未知 OpenAI-compatible 模型应显式配置。

流式请求开启 `stream_options.include_usage`，并解析 `choices=[]` 的 usage-only 块、cache hit/miss 和 reasoning Token。REPL 用 `/usage` 查看最近一次普通 ReAct，单次 CLI 用 `--json` 获取完整 usage/cost。成本以供应商返回的实际 Token 为准，不能只用“代码行数”精确推算。

## 联网工具

`web_search` 使用 DuckDuckGo HTML 搜索，返回标题、URL 和摘要。

`web_fetch` 可以抓取公开 HTTP/HTTPS 页面，并做基础正文提取。它会拒绝 `file://`、loopback、私有网络和内网地址，降低 SSRF 风险。

如果需要登录态、浏览器状态或 JS 渲染页面，建议使用 Chrome DevTools MCP。

## MCP

TaskPaw 可以连接 MCP server，并把远端工具动态注册为：

```text
mcp__<server-name>__<tool-name>
```

初始化项目级 Chrome DevTools MCP 配置：

```bash
uv run taskpaw mcp init-chrome --scope project
```

它会写入 `.taskpaw/mcp.json`，内容类似：

```json
{
  "mcpServers": {
    "chrome-devtools": {
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "chrome-devtools-mcp@latest",
        "--no-usage-statistics"
      ]
    }
  }
}
```

连接已有 remote-debugging Chrome：

```bash
uv run taskpaw mcp init-chrome \
  --scope project \
  --browser-url http://127.0.0.1:9222
```

查看已配置的 MCP server：

```bash
uv run taskpaw mcp list
```

把 TaskPaw 自身作为 MCP server 暴露：

```bash
uv run taskpaw mcp serve --transport stdio
uv run taskpaw mcp serve --transport http --port 3000
```

HTTP smoke：

```bash
curl -sS -X POST http://127.0.0.1:3000 \
  -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

Chrome DevTools MCP 会把浏览器页面和 DevTools 状态暴露给 Agent。不要随意把包含个人账号、敏感数据或生产后台的 Chrome 会话授权给 Agent。

## Runtime API

TaskPaw 内置轻量 Runtime API，适合外部系统接入线程、turn、事件和后台任务。

启动服务：

```bash
TASKPAW_RUNTIME_API_KEY=dev-key \
uv run taskpaw serve --http --port 8080
```

创建线程：

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/threads \
  -H 'x-api-key: dev-key'
```

发送 turn：

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/threads/<thread_id>/turns \
  -H 'content-type: application/json' \
  -H 'x-api-key: dev-key' \
  -d '{"message":"总结这个项目"}'
```

读取事件：

```bash
curl -sS http://127.0.0.1:8080/v1/threads/<thread_id>/events \
  -H 'x-api-key: dev-key'
```

创建并查看后台任务：

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/tasks \
  -H 'content-type: application/json' \
  -H 'x-api-key: dev-key' \
  -d '{"message":"后台总结这个仓库","mode":"plan"}'

curl -sS http://127.0.0.1:8080/v1/tasks \
  -H 'x-api-key: dev-key'
```

也可以只启动队列消费者，不暴露 HTTP：

```bash
uv run taskpaw worker --workers 2 --cwd .
```

任务队列按项目目录隔离；worker 使用 SQLite 原子事务领取任务，并通过 lease/heartbeat 恢复崩溃任务。运行中取消会阻止 worker 把迟到结果重新覆盖为 completed。

## 图片输入

TaskPaw 支持在 prompt 里引用图片：

```text
分析这张截图 @image:./screenshots/page.png
```

也支持绝对路径和远程图片：

```text
解释这张图 @image:/Users/me/Desktop/diagram.png
看看这个图片 @image:https://example.com/image.png
```

本地图片会自动压缩、缩放，并在需要时把透明底铺成白底，再转为 data URL。如果当前 provider/model 不支持多模态输入，TaskPaw 会自动降级为文本元信息，不会把不支持的图片 payload 发给模型。

## 快照

每次 Agent run 都会尽力创建项目快照：

- `pre-turn`
- `post-turn`

快照保存在 `~/.taskpaw/snapshots/`，不会写入项目 `.git`。

REPL 中可以使用：

```text
/snapshot
/restore 1
/snapshot clean
```

## SDK

```python
from taskpaw.sdk import create_default_engine

engine = create_default_engine(cwd=".")
result = engine.ask_complete("解释这个项目")
print(result.text)

plan_result = engine.plan_complete("先读取 README，再总结项目结构")
team_result = engine.team_complete("让多个 Agent 并行检查核心模块")
```

## 开发

安装开发依赖：

```bash
uv sync --extra dev
```

运行检查：

```bash
uv run python -m ruff check .
uv run python -m ruff format --check .
uv run python -m pytest
uv build
```

常用 smoke：

```bash
uv run taskpaw --version
uv run taskpaw --help
uv run taskpaw doctor --cwd .
uv run taskpaw --plain -p hello
```

贡献流程与目录说明见 [CONTRIBUTING.md](CONTRIBUTING.md)，版本变更见 [CHANGELOG.md](CHANGELOG.md)。

## License

MIT. See [LICENSE](LICENSE).
