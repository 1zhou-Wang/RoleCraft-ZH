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

## 6. 音频输入 track B1/B2 级联（2026-11-14）

链路：`data/voice/audio/<scene_id>_<idx>.mp3`（用户 TTS，784 条）
→ **Qwen3-ASR-0.6B** 转写 → **Qwen3.5-2B(:8088) / 4B(:8089)**（system prompt 与文字 track 完全相同，多轮带 history）
→ TTS 渲染回复文本（本地 Qwen3-TTS 样本 + 云端 qwen-audio-3.1-tts-next 全量）。

| 环节 | 脚本 | 产物 |
|---|---|---|
| ASR | `local/asr_transcribe.py` | `data/audio_track/asr_transcripts.jsonl`（784 行，含 char-level WER）、`asr_summary.json` |
| LLM 级联 | `local/audio_cascade.py` | `baselines/audio_pipeline_qwen3.5-2b/`、`...4b/`（各 50 卡 × 10 场景 = 500 行 / 784 turn）+ `_manifest.json` |
| 本地 TTS 样本 | `local/audio_pipeline/local_tts_samples.py` + `finalize_local_tts.py` | `audio/<scene_id>_<idx>.wav`（20 条，`tts_render="qwen3-tts-0.6b"`） |
| 云端 TTS 全量 | `local/audio_pipeline/cloud_tts_render.py` | `audio_cloud/<scene_id>_<idx>.mp3`（1548 条，`tts_render="qwen-audio-3.1-tts-next"`） |
| 异常清理/抽检/汇总 | `clean_bad_local_tts.py` / `check_cloud_audio.py` / `update_audio_summary.py` | `logs/audio_track/*.json`、`baselines/audio_pipeline_summary.json` |

环境要点（全部装到项目 `vendor/`，不动 torch venv 的 transformers 5.19 / torch 2.14.1+cu130）：
- `vendor/asr` = qwen-asr 0.0.6 + transformers 4.57.6（`--no-deps` 安装 + 手动补 tokenizers/huggingface_hub/accelerate/librosa/soundfile/sox/six 等；`kernels.py` 故意 raise ImportError 让 transformers 走 fallback；forced aligner 模块在脚本里用占位模块替换，避免拉 nagisa→dynet）
- `vendor/tts` = qwen-tts 0.1.1 + transformers 4.57.3（同上策略，另装 onnxruntime/sox/torchaudio 2.11.1(cpu)/einops；`kernels.py` 同样 raise ImportError）
- ASR mp3 解码走 ffmpeg 子进程（16k 单声道 float32），不依赖 librosa/av 解码
- 两个 pipeline 的 `turns[].user_text` 是 **ASR 实际转写文本**，`user_text_gt` 是原始用户台词，便于音频判官/错误归因；`assistant_audio` 指向 wav/mp3 相对路径

### 6.1 实测结果（2026-11-14）

- **ASR**：784/784 条，mean WER **0.0376**、median 0、p90 0.143、exact match 80.5%，100 s（batch 16，~0.07 s/条）
- **B1 级联（2B）**：500/500 场景、784/784 turn，299 s，turn 延迟 mean 1.51 s / p95 2.53 s
- **B2 级联（4B）**：500/500 场景、784/784 turn，415 s，turn 延迟 mean 2.09 s / p95 3.00 s
- **本地 Qwen3-TTS-0.6B 速度**：**无 GPU 竞争 ~20-30 s/条，RTF≈2**（音频均长 14.3 s，per-item 27.9 s）；EdgeInstant 占用 GPU 后 70-171 s/条 → 全量 1568 条需 ~8 h，不可接受，故只做链路验证样本
- **本地样本**：保留 **20 条** wav（B1 15 / B2 5；覆盖安全拒绝/重复问答/元对话等类别），另 2 条在 GPU 竞争下产生的异常文件（静音 112 s / 超长 65.8 s）已清理并用云端重渲
- **云端全量渲染**：**1548 条** mp3，4 次运行累计 2300 s（**40.4 条/分钟**，24 workers / 180 rpm），失败 **0**；音频均长 13.0 s、总计约 5.6 h，文件大小 6.4 KB~232 KB（均 85 KB）
- **音频覆盖**：B1/B2 各 **784/784 turn 有 assistant_audio**，文件缺失 0
- **已知异常**：2B 的 2 条 512 字复读退化回答（rc24_s26#1、rc10_s14#2）经云端 TTS 后音频时长明显短于文本期望（3.1 s / 18.4 s），判官可能按 asr_mismatch 暴露——这与该回答自身的 repetition 退化一致
- 判官消费方式见 `scripts/audio_judge.py`（`assistant_audio` + 文本），音频目录相对 baseline 文件解析

## 7. v0.2 官方解码配置（Plan A）重测（2026-11-14）

- 背景：Qwen3.5 README 非思考文本推荐 `temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0`（注意是 presence_penalty，非 repetition_penalty），固定 `seed=42`；v0.1 用 greedy 无惩罚，部分复读 badcase 由解码配置导致。**v0.1 全部目录与 results 保持不动**。
- 服务：`local/serve.ps1 -Action start -Official`（脚本新增 `-Official` 开关，追加 `--temp 1.0 --top-k 20 --top-p 1.0 --min-p 0.0 --presence-penalty 2.0 --seed 42`）；`/props` 已确认 server 默认采样生效；同 prompt 两次 temp=1 请求输出逐字一致（seed 可复现），temp=0 覆盖请求输出不同。
- 文字 track：`baselines/qwen3.5-2b-official/`、`baselines/qwen3.5-4b-official/`（各 1500 场景 / 2616 turn / 0 失败；588.2 s / 954.6 s）；`local/run_baseline.py --sampling official`
- 音频级联：`baselines/audio_pipeline_qwen3.5-2b-official/`、`...-4b-official/`（各 500 场景 / 784 turn / 0 失败；179.9 s / 325.9 s；ASR 转写复用 v0.1）；`local/audio_cascade.py --sampling official`
- TTS：1568 条回复全部云端 `qwen-audio-3.1-tts-next` 渲染为 mp3（`audio_cloud/`，24 kHz mono），回填 `assistant_audio` + `tts_render`；共 20 pilot + 1547 + 1 retry = 1568 条，累计 2720 s（34.6 条/分钟），最终 0 失败（1 条 B2 rc29_s03#2 被内容审核拒绝，换 seed=43 重试成功）
- **复读退化对比（v0.1 greedy → v0.2 official）**：文字 2B 0.19%→**0%**、4B 0.04%→**0%**；级联 B1 0.38%→**0%**、B2 0%→**0%**；回复均长 +10-12%（2B 65.9→76.8、4B 70.5→78.8）
- 汇总：`baselines/audio_pipeline_summary_official.json`；对比明细 `logs/audio_track/official_vs_greedy_compare.json`
- 新增/复用脚本：`local/run_baseline.py --sampling official`、`local/audio_cascade.py --sampling official`、`local/compare_official_vs_greedy.py`、`local/audio_pipeline/update_audio_summary_official.py`、`local/audio_pipeline/local_tts_fallback.py`（云端审核拒绝时的本地兜底）
- 服务保持运行：2B :8088 / 4B :8089（官方采样 flag），供判官/补跑使用

## 8. MiniCPM5-2B 链路加测（ASR + MiniCPM5-2B + TTS，2026-11-18）

新链路：`Qwen3-ASR-0.6B → MiniCPM5-2B → 云端 qwen-audio-3.1-tts-next`，同时补 MiniCPM5-2B 文字 track
（与 Qwen3.5-2B/4B 同场景/同 system prompt/同口径）。**v0.1/v0.2 现有目录一律不动**。

### 8.1 模型与部署
- GGUF：ModelScope 镜像 `OpenBMB/MiniCPM5-2B-GGUF` → `E:\RBM\models\minicpm-gguf\MiniCPM5-2B-Q8_0.gguf`
  （2,679,710,688 B ≈ 2.68 GB；`architecture=llama`，381 tensors，llama.cpp b11552 直接加载）
- chat_template 9060 字符，含 `enable_thinking` 分支；role-play 统一关闭 thinking：
  `--reasoning off` + 环境变量 `LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"enable_thinking":false}`（避免 Start-Process 引号转义坑）
- 两个 server（`local/serve_minicpm.ps1`）：
  - **greedy :8090**：server 无采样 flag，请求 `temperature=0`
  - **official :8091**：`--temp 1.0 --top-p 0.95 --min-p 0.0 --seed 42 --repeat-penalty 1.05`
    （README 要求 `--min-p 0.0`：llama.cpp 默认 0.05 会加重读；实测不加 `repeat-penalty 1.05` 时 rc01 会"快快快…"复读 1/10，加上后 10/10 干净）
- 验证：`local/audio_pipeline/minicpm_probe.py`（thinking 泄漏/采样生效/seed 复现）、`inspect_minicpm_gguf.py`（模板元数据）；
  greedy 与 official 均无 `<think>` 泄漏、无 reasoning_content、同 prompt 3 次逐字一致
- GGUF 下载说明：lead 的 HF 代理下载卡在 46.5MB，改用 ModelScope 直连（~18-25 MB/s）分段续传完成

### 8.2 结果（2026-11-18）
| track | slug | 场景 | turn | 失败 | 总耗时 | 每轮延迟 mean | 回复均长/max | 复读退化 |
|---|---|---|---|---|---|---|---|---|
| 文字 greedy | `baselines/minicpm5-2b-greedy/` | 1500/1500 | 2616/2616 | 0 | 376.3 s | 0.564 s | 60.8 / 954 | 0.54% |
| 文字 official | `baselines/minicpm5-2b-official/` | 1500/1500 | 2616/2616 | 0 | 497.0 s | 0.739 s | 72.3 / 514 | 0.04% |
| 级联 greedy | `baselines/audio_pipeline_minicpm5-2b-greedy/` | 500/500 | 784/784 | 0 | 125.7 s | 0.625 s | 64.1 / 954 | 0.51% |
| 级联 official | `baselines/audio_pipeline_minicpm5-2b-official/` | 500/500 | 784/784 | 0 | 157.5 s | 0.788 s | 78.3 / 1021 | 0.26% |

### 8.3 与 Qwen3.5-2B/4B 同口径对照（明细 `logs/minicpm5/minicpm_vs_qwen.json`）
| 模型/配置 | 文字 track 回复均长 | 文字复读率 | 文字每轮延迟 | 级联回复均长 | 级联复读率 | 级联每轮延迟 |
|---|---|---|---|---|---|---|
| Qwen3.5-2B greedy（v0.1） | 65.9 | 0.19% | 0.725 s | 65.6 | 0.38% | 1.514 s |
| Qwen3.5-2B official（v0.2） | 76.8 | 0% | 0.891 s | 73.9 | 0% | 0.901 s |
| Qwen3.5-4B greedy（v0.1） | 70.5 | 0.04% | 1.341 s | 70.6 | 0% | 2.087 s |
| Qwen3.5-4B official（v0.2） | 78.8 | 0% | 1.444 s | 77.9 | 0% | 1.643 s |
| **MiniCPM5-2B greedy** | 60.8 | 0.54% | **0.564 s** | 64.1 | 0.51% | **0.625 s** |
| **MiniCPM5-2B official** | 72.3 | 0.04% | 0.739 s | 78.3 | 0.26% | 0.788 s |

- MiniCPM5-2B 速度最快（2B 级），greedy 复读率略高于 Qwen（0.54% vs 0.19%），official 配置可压到 0.04%
- **云端 TTS**：两个 audio slug 共 **1568/1568 条**回复由 `qwen-audio-3.1-tts-next` 渲染为 mp3
  （`audio_cloud/<scene_id>_<idx>.mp3`，24 kHz mono，`tts_render="qwen-audio-3.1-tts-next"`），
  累计 **2476.7 s（41.3 min，38.0 条/分钟，24 workers/180 rpm），失败 0**；B1/B2 各 784/784 turn 有 `assistant_audio`，文件缺失 0
- 汇总：`baselines/audio_pipeline_summary_minicpm5.json`（含 text/audio 成功数、耗时、TTS 条数/失败/速率、复读率、ASR 指标）；
  失败清单：`logs/minicpm5/cloud_tts_failures.json`（空）

### 8.4 新增脚本
`local/serve_minicpm.ps1`、`local/audio_cascade_minicpm.py`、`local/audio_pipeline/{minicpm_probe,inspect_minicpm_gguf,update_audio_summary_minicpm5,local_tts_fallback}.py`、
`local/compare_minicpm_vs_qwen.py`、`local/run_baseline.py --sampling minicpm-official`、`local/audio_pipeline/cloud_tts_render.py --failures-file`。



