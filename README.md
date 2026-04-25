# ubiservice — 本地模拟服务器

## 简介

ubiservice 是 UbiOracle 竞赛平台的**轻量级本地模拟服务器**，供选手在本地开发、调试和测试自己的推理服务客户端。

它完整模拟了竞赛的核心协议：**注册 → 查询任务 → 竞标 → 推理 → 提交结果 → 查看排行榜**，但评分逻辑做了简化（每完成一个任务固定得 1.0 分），方便选手专注于推理流程的开发。

## 安装依赖

```bash
pip install fastapi uvicorn redis pydantic uvloop httptools requests
```

同时需要安装 Redis：

```bash
# Ubuntu/Debian
apt-get install redis-server

# macOS
brew install redis
```

## 快速开始

### 1. 启动模拟服务器

```bash
cd ubiservice
python bin/start.py
```

服务器会自动完成以下操作：
- 启动 Redis（端口 6379）
- 启动后台任务生成器（持续生成模拟任务）
- 启动 Matcher API（端口 8003）

看到 `Mock server ready!` 即可开始使用。

### 2. 运行示例客户端

新开一个终端：

```bash
python examples/client_example.py
```

示例客户端会注册一个测试队伍，执行 5 轮"查询→竞标→推理→提交"循环，最后打印排行榜。

### 3. 停止服务器

在服务器终端按 `Ctrl+C`，会自动清理 Redis 和 Matcher 进程。

---

## API 接口

### 注册

**请求**：`POST /register`

```json
{"name": "team_alpha", "token": "your_secret_token"}
```

**响应**：

```json
{"status": "ok"}
```

`token` 是选手的唯一标识。所有后续请求都需要携带 `token`。

### 查询任务

**请求**：`POST /query`

```json
{"token": "your_secret_token"}
```

**响应**（200）：

```json
{
  "task_id": 1,
  "target_sla": "Gold",
  "target_reward": 1520.0,
  "max_winners": 1
}
```

**响应**（404）：当前无可用任务，稍后重试。

**响应**（429）：请求过于频繁，稍后重试（每用户限制 30 req/s）。

**限制**：每个用户最多同时持有 48 个已查询的任务（含正在处理的）。

### 竞标任务

**请求**：`POST /ask`

```json
{
  "token": "your_secret_token",
  "task_id": 1,
  "sla": "Gold"
}
```

**初赛规则**：`sla` 字段必须与 `target_sla` 完全匹配，否则被拒绝。

**响应**（接受）：

```json
{
  "status": "accepted",
  "task": {
    "overview": { ... },
    "messages": [
      {
        "ID": 0,
        "prompt": "Question: Which of the following...",
        "eval_request_type": "generate_until",
        "eval_gen_kwargs": {
          "temperature": 0.1,
          "top_p": 0.9,
          "max_gen_toks": 128,
          "until": ["\n"]
        },
        "eval_continuation": null
      },
      {
        "ID": 1,
        "prompt": "Context text...",
        "eval_request_type": "loglikelihood",
        "eval_gen_kwargs": null,
        "eval_continuation": " choice A"
      },
      {
        "ID": 2,
        "prompt": "Context text...",
        "eval_request_type": "loglikelihood",
        "eval_gen_kwargs": null,
        "eval_continuation": " choice B"
      },
      {
        "ID": 3,
        "prompt": "A long document text...",
        "eval_request_type": "loglikelihood_rolling",
        "eval_gen_kwargs": null,
        "eval_continuation": null
      }
    ]
  }
}
```

**响应**（拒绝）：

```json
{"status": "rejected"}
```

或

```json
{"status": "closed"}
```

**限制**：每个用户最多同时持有 32 个已竞标但未提交的任务。

### Message 字段说明

每个 message 代表一条推理请求。根据 `eval_request_type` 的不同，选手需要进行不同的推理：

#### generate_until

选手需要：
1. 使用 `prompt` 作为输入
2. 按 `eval_gen_kwargs` 中的参数（temperature、top_p、top_k 等）进行生成
3. 遇到 `until` 中的停止符或达到 `max_gen_toks` 时停止
4. 将生成的文本填入 `response` 字段

#### loglikelihood

选手需要：
1. 计算 `P(eval_continuation | prompt)` 的对数概率
2. 将 logprob 值填入 `accuracy` 字段

**注意**：一道多选题会拆分成多个 message（每个候选答案一条）。平台根据各候选的 logprob 选择 argmax 来判断选手的回答是否正确。

#### loglikelihood_rolling

选手需要：
1. 计算整段 `prompt` 文本的 rolling log-likelihood
2. 将总 logprob 值填入 `accuracy` 字段

### 提交结果

**请求**：`POST /submit`

```json
{
  "user": {
    "name": "team_alpha",
    "token": "your_secret_token"
  },
  "msg": {
    "overview": { ... },
    "messages": [
      {
        "ID": 0,
        "prompt": "...",
        "response": "生成的文本",
        "eval_request_type": "generate_until",
        ...
      },
      {
        "ID": 1,
        "prompt": "...",
        "accuracy": -2.35,
        "eval_request_type": "loglikelihood",
        ...
      }
    ]
  }
}
```

**结果填充规则**：

| request_type | 填充字段 | 类型 | 说明 |
|-------------|---------|------|------|
| `generate_until` | `response` | string | 生成的文本 |
| `loglikelihood` | `accuracy` | float | 对数概率 |
| `loglikelihood_rolling` | `accuracy` | float | 总对数概率 |

**响应**：

```json
{"status": "ok"}
```

### 查看排行榜

**请求**：`GET /leaderboard`

**响应**：

```json
{
  "timestamp": 1711929600,
  "round_elapsed_s": 3600.0,
  "leaderboard": [
    {
      "rank": 1,
      "name": "team_alpha",
      "score": 5.0,
      "tasks_completed": 5,
      "tasks_accepted": 5,
      "avg_correctness": 1.0,
      "avg_latency_ms": 2.2,
      "credit": 1.0
    }
  ]
}
```

---

## SLA 等级

从低到高：Bronze → Silver → Gold → Platinum → Diamond → Stellar → Glorious → Supreme

等级越高，要求的响应延迟越低，但竞标排名更有优势。

## 公平性机制

- **Boost**：竞标同分失败的选手会累积 boost 分数（+0.5/次），提高下次获胜概率，赢后重置为 0
- **Rate Limit**：每个用户 `/query` + `/ask` 合计限制 30 req/s，防止刷接口

## 注意事项

- **这是模拟服务器**，评分逻辑已简化：每完成一个任务固定得 **1.0 分**（真实比赛中得分取决于推理质量和延迟）
- 任务由 `FakeGenerator` 随机生成，prompt 是随机拼接的单词，不需要真正的语言模型
- `eval_gen_kwargs` 中的采样参数（temperature、top_p 等）是评测条件的一部分，真实比赛中必须遵守
- 默认为初赛模式（`preliminary`），可通过环境变量 `COMPETITION_MODE=final` 切换到决赛模式
- 每个用户最多同时查询 1024 个任务，最多同时竞标持有 64 个任务
- 竞标窗口为 10 秒，超时后任务关闭

## 本地评测

ubiservice 提供了一个本地评测脚本，可以模拟正式比赛的评测流程。

### 快速体验（使用自带的 fake submission）

```bash
cd ubiservice

# 1. 打包示例选手代码
tar czf /tmp/fake_submission.tar.gz -C examples/fake_submission .

# 2. 运行评测（30 秒）
python bin/run_eval.py \
    --submission /tmp/fake_submission.tar.gz \
    --team-name test_team \
    --team-token test_token \
    --duration 30 \
    --output /tmp/result.json

# 3. 查看结果
cat /tmp/result.json
```

预期输出：score > 0，tasks_completed > 0，setup_ok=true，startup_ok=true。

### 评测你自己的代码

选手代码目录结构：

```
my_submission/
├── run.sh              # 启动脚本（必须）
├── setup.sh            # 环境安装脚本（可选）
├── client.py           # 你的 client 代码
└── ...
```

`run.sh` 示例：

```bash
#!/bin/bash
cd "$(dirname "$0")"
source /tmp/my_env/bin/activate  # 如果 setup.sh 创建了虚拟环境
python client.py
```

打包并评测：

```bash
# 打包（注意：从代码目录内部打包，不要包含外层目录名）
tar czf my_submission.tar.gz -C /path/to/my_submission .

# 评测
python bin/run_eval.py \
    --submission my_submission.tar.gz \
    --team-name my_team \
    --team-token my_token \
    --duration 60 \
    --output result.json
```

评测脚本会自动：解压代码 → 执行 setup.sh（如有） → 启动模拟平台 → 生成 contest.json → 启动你的 run.sh → 等待指定时间 → 收集分数。

你的 run.sh 可以通过以下环境变量获取配置：

| 变量 | 说明 |
|---|---|
| `PLATFORM_URL` | 评测平台地址（如 `http://127.0.0.1:8003`） |
| `CONFIG_PATH` | 比赛配置文件路径（JSON） |
| `CONTESTANT_PORT` | 选手 HTTP 服务端口（如需要） |
| `MODEL_PATH` | 模型权重路径 |
| `TEAM_NAME` | 队伍名称 |
| `TEAM_TOKEN` | 队伍 Token |

> **注意：本地评测将选手代码和平台运行在同一台机器上，仅供开发调试使用。正式比赛中，选手代码和评测平台部署在不同的 Pod（容器）内，选手只能通过 HTTP 接口与平台通信，无法直接访问平台内部组件（如 Redis）。**

## 项目结构

```
ubiservice/
├── bin/
│   ├── start.py              # 一键启动脚本
│   ├── start_platform.py     # 平台启停工具（供 run_eval 调用）
│   └── run_eval.py           # 本地评测脚本
├── config/
│   └── defination_base.json  # 长度/采样/SLA 配置
├── examples/
│   ├── client_example.py     # 示例客户端（交互式）
│   └── fake_submission/      # 示例选手提交包
│       ├── run.sh            # 启动脚本
│       └── client.py         # 简单的 query→ask→submit 循环
├── src/
│   ├── defination.py         # 数据模型定义
│   ├── fake_generator.py     # 随机消息生成器
│   ├── matcher.py            # Matcher API 服务
│   └── task_builder.py       # 任务构建器
└── README.md
```
