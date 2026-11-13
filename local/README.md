# local/ — RoleCraft-ZH 本地推理基线（Qwen3.5-2B / 4B）

本目录只放本地推理链路脚本，数据/评测产物分别落在 `data/`、`baselines/`、`results/`。

## 1. 模型产物（E:\RBM\models\qwen-gguf\）

| 文件 | 大小 | 说明 |
|---|---|---|
| `qwen35_2b-Q8_0.gguf` | 1.87 GB | Qwen3.5-2B 文本 track，Q8_0 |
| `qwen35_4b-Q8_0.gguf` | 4.17 GB | Qwen3.5-4B 文本 track，Q8_0 |

- 来源：`E:\RBM\models\Qwen3.5-2B` / `Qwen3.5-4B`（`Qwen3_5ForConditionalGeneration`，原生多模态）
- 转换：llama.cpp **b11552** 自带 `convert_hf_to_gguf.py --outtype bf16 --no-mtp`
  - `--no-mtp`：checkpoint 带 `mtp.*` 权重（`mtp_num_hidden_layers: 1`），文本 baseline 不需要 MTP/speculative 层，直接排除
  - vision tower（`model.visual.*`）由 TEXT 路线自动跳过；文本 track 只用文本
  - 复现：`powershell -NoProfile -File local/convert.ps1 -Models all`
- 转换耗时：2B ≈ 1.5 min（bf16）+ 6.4 s（Q8_0）；4B ≈ 3 min（bf16）+ 13.7 s（Q8_0）
- 量化：`llama-quantize.exe <bf16> <Q8_0> Q8_0`
- 中间 bf16 文件可按需删除（`local/convert.ps1` 默认删）

## 2. 服务（local/serve.ps1）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File local/serve.ps1 -Action start
powershell -NoProfile -ExecutionPolicy Bypass -File local/serve.ps1 -Action status
powershell -NoProfile -ExecutionPolicy Bypass -File local/serve.ps1 -Action stop
```

| 模型 | 端口 | 命令参数 |
|---|---|---|
| qwen3.5-2b | 8088 | `-m qwen35_2b-Q8_0.gguf -c 32768 -ngl 999 -fa on -t 8 -np 4 --host 127.0.0.1 --port 8088 --alias qwen3.5-2b --jinja --reasoning off --no-webui` |
| qwen3.5-4b | 8089 | 同上，换成 4B gguf / 8089 / alias qwen3.5-4b |

关键点：

- **`-c 32768` + `-np 4`**：llama.cpp 把 `-c` 按 slot 均分，日志 `n_ctx_slot = 8192`，即每个会话 8K 上下文
  （单场景 system prompt + 5 轮对话约 1.5–2.5K token，够用）。直接 `-c 8192` 会变成 2048/slot，长场景会截断。
- **`--reasoning off` 必须加**：Qwen3.5 的 chat template 在 `enable_thinking=false` 时插入空 `<think></think>` 块（非 thinking）。
  不显式关闭时 llama-server（默认 `--reasoning auto`）会传 `enable_thinking=true`，模型会先输出一大段思考
  （`finish_reason=length`、`content` 为空、思考在 `reasoning_content`）——本 benchmark 要非 thinking 行为。
- 4B 的 chat_template 与 2B 略有差异（未定义 enable_thinking 时默认**开启**思考），所以同样必须 `--reasoning off`。
- 运行时需要 `llama_cpp\cudart` 在 PATH（脚本自动加）；日志/pid 在 `logs/local/`。
- 显存：两个 Q8_0 模型同时常驻约 8.5 GB（RTX 3090 24GB 充裕）。

## 3. 基线 runner（local/run_baseline.py）

```powershell
# 冒烟（local/mock_data，4 张 mock 卡 × 3 场景）
& .\.venv\Scripts\python.exe local\run_baseline.py --mock --model-slug qwen3.5-2b --base-url http://127.0.0.1:8088 --concurrency 4

# 正式基线（2B 先跑，4B 可并行；输出 baselines/<slug>/<card_id>.jsonl）
$env:PYTHONIOENCODING='utf-8'
& .\.venv\Scripts\python.exe local\run_baseline.py --model-slug qwen3.5-2b --base-url http://127.0.0.1:8088 --concurrency 4 --log-file logs\local\baseline_2b.log
& .\.venv\Scripts\python.exe local\run_baseline.py --model-slug qwen3.5-4b --base-url http://127.0.0.1:8089 --concurrency 4 --log-file logs\local\baseline_4b.log
```

行为：

- 输入 `data/characters/<card_id>.json` + `data/scenes/<card_id>.json`（或用 `--characters-dir/--scenes-dir/--cards/--limit-cards` 指定）
- system prompt 模板：`你将扮演「{name}」。以下是人设卡：\n{persona_card}\n\n规则：…`
- 场景内按 turn 串行，历史 assistant 回复回填；场景/卡之间用线程池并发（并发数 ≈ `-np`）
- 请求：`/v1/chat/completions`，`temperature=0, max_tokens=512, stream=false`，单请求失败重试 3 次
- 输出一行一场景，schema 对齐 `SCHEMA.md`（含逐 turn `latency_s` 与场景级 `latency_s`），另写 `_manifest.json` 统计
- 断点续跑：已完成的场景跳过，出错场景默认重跑（`--no-retry-errors` 关闭）；`--dry-run` 只校验数据

## 4. 冒烟验证记录（2026-11-14）

- 2B：mock 4 卡 × 12 场景全部成功（5.2 s，25 个 turn 请求），非 thinking，中文角色内回答正常，断点续跑 0 请求
- 4B：mock 4 卡 × 12 场景全部成功（8.6 s，25 个 turn 请求），非 thinking
- 产物示例：`local/mock_data/baselines/qwen3.5-{2b,4b}/<card>.jsonl`

## 5. 正式基线结果（2026-11-14，50 卡 × 30 场景 = 1500 场景 / 2616 turns）

| 模型 | 场景 | turn | 总耗时 | 场景延迟 mean/p50/p95 | 每轮延迟 mean/p95 | 回复字数 mean |
|---|---|---|---|---|---|---|
| qwen3.5-2b | 1500/1500 | 2616/2616 | 486.8 s（8.1 min） | 1.265 / 1.194 / 2.330 s | 0.725 / 1.121 s | 65.9 |
| qwen3.5-4b | 1500/1500 | 2616/2616 | 892.5 s（14.9 min） | 2.338 / 2.345 / 4.006 s | 1.341 / 1.874 s | 70.5 |

- 失败场景 0、失败 turn 0、请求重试 0、空回答 0、`<think>` 泄漏 0
- 产物：`baselines/qwen3.5-2b/rc01.jsonl … rc50.jsonl`（1500 行）+ `_manifest.json`；4B 同构
- 人工抽检异常：2B 有 5 条 >300 字的复读退化（rc24_s26、rc20_s04、rc13_s14 ×2、rc33_s10，最长 710 字，应为 max_tokens 截断），4B 无复读退化（最长 149 字）；
  `*_s28` 元对话场景里出现 "AI" 关键词的回答均为角色内否认（期望行为），非自曝
- 校验/扫描脚本：`local/check_data.py`、`local/verify_baseline.py`、`local/anomaly_scan.py`

