# EdgeInstant-Omni-2B 本地评测（RoleCraft-ZH）

本目录是 B3 链路 **EdgeInstant-Omni-2B-1106-Preview** 的音频输入 track + 纯文字 track runner。

## 环境（重要）

| 项 | 值 |
|---|---|
| 模型 | `E:\RBM\models\EdgeInstant-Omni-2B-1106-Preview`（6 shards，7.32GB，2.993B params） |
| Python | `E:\RBM\algorithms\draft\qwen3_5_infra_test\qwen_bench_win\.venv\Scripts\python.exe`（torch 2.14.1+cu130） |
| transformers | **必须 5.12.1**（模型卡 pin）。venv 自带 5.19 会在 audio tower 和 linear-attention 自定义层上报错 |
| 依赖位置 | `vendor/edgeinstant/`（--target 安装，不动共享 venv 和 torch）：transformers 5.12.1 + tokenizers 0.22.2 + hf-hub 1.33.0 |

**所有脚本运行时环境变量**：

```powershell
$env:EDGEINSTANT_VENDOR='1'   # sys.path 优先用 vendor/edgeinstant
$env:USE_HUB_KERNELS='NO'     # venv 的 kernels 0.17 与 5.12.1 Hub kernel 注册不兼容
$env:PYTHONIOENCODING='utf-8'
```

为什么 5.19 不行：
1. 音频路径：`get_valid_indices(chunk_lengths)` 在 5.19 需要 `n_window` 参数（音频 tower 直接报 TypeError）；
2. 文字路径也会走 `EdgeInstantDecoderLayer`（inference_edgeinstant.py 的 linear-attention CUDA-graph 层），5.19 把 `conv_states` 改成 dict → `'dict' object has no attribute 'shape'`。

其他两个坑（已封装进 `ei_common.py`）：
- `AutoProcessor.from_pretrained` 会因 `processor_config.json` 的 `image_processor_subfolder` 去加载图像处理器，5.19 需要 PIL、5.12.1 需要 torchvision → 本目录改用 `build_processor()` **手动构造** `EdgeInstantProcessor(..., image_processor=None)`。
- `device_map='cuda:0'` 加载后输入不会自动搬移，必须 `inputs.to(device)` 再 `generate`（官方 duplex_edgeinstant.py 也这么做）。

## 脚本

| 脚本 | 用途 |
|---|---|
| `ei_common.py` | 公共：vendor 激活、processor 构造、模型加载、ffmpeg 16k 解码（带缓存）、角色扮演 prompt、数据读取 |
| `load_smoke.py` | 门禁 smoke：load + 单条音频 generate + DuplexSession + 原生语音渲染 |
| `run_edgeinstant.py` | **音频输入 track（784 轮 / 500 场景）**：`--phase text` 只出文本；`--phase render` 补原生语音；`--phase all` |
| `run_edgeinstant_text.py` | **纯文字 track（2616 轮 / 1500 场景）**，对齐 Qwen3.5-2B/4B 文字基线 |
| `render_samples.py` | 从音频 track 结果抽 N 条 turn 渲染原生语音（语音判官冒烟） |
| `processor_selftest.py` | processor 输入接口自检（text/audio/asr 三个分支） |
| `debug_text.py` | 5.19 文本路径问题的调试脚本（保留供复查） |

## 输出

| 产物 | 路径 |
|---|---|
| 音频 track 文本回复 | `baselines/audio_edgeinstant/<card_id>.jsonl`（50 文件，一行一场景，schema 见 SCHEMA.md） |
| 音频 track 汇总 | `baselines/audio_edgeinstant/_manifest.json`、`logs/edgeinstant/audio_manifest.json` |
| 原生语音 wav（抽样 30 条） | `baselines/audio_edgeinstant/audio/<scene_id>_<idx>.wav`（24kHz mono s16） |
| 文字 track | `baselines/edgeinstant-omni-2b-text/<card_id>.jsonl`（一行一场景，turns 有 idx/user_text/assistant_text/latency_s） |
| 文字 track 汇总 | `baselines/edgeinstant-omni-2b-text/_manifest.json` |
| 日志 | `logs/edgeinstant/` |

## 复现命令

```powershell
$py = 'E:\RBM\algorithms\draft\qwen3_5_infra_test\qwen_bench_win\.venv\Scripts\python.exe'
$env:EDGEINSTANT_VENDOR='1'; $env:USE_HUB_KERNELS='NO'; $env:PYTHONIOENCODING='utf-8'

# 门禁 smoke（~1.5 min，含 DuplexSession warmup）
& $py local/edgeinstant/load_smoke.py

# 音频 track：文本（784 轮，~26 min @3090，max_new_tokens=256 greedy）
& $py local/edgeinstant/run_edgeinstant.py --phase text

# 原生语音抽样 30 条（~1 min，渲染完回填 assistant_audio）
& $py local/edgeinstant/render_samples.py --n-turns 30 --n-cards 10

# 纯文字 track（2616 轮，~50 min，max_new_tokens=512 greedy）
& $py local/edgeinstant/run_edgeinstant_text.py
```

所有 runner 都幂等可续跑：已完成的场景直接跳过，中途失败重跑即可。

## 实测数据（RTX 3090 24GB，2026-11-14）

| track | 规模 | 成功 | 均延迟 | 吞吐 | 总耗时 |
|---|---|---|---|---|---|
| 音频输入（入音频→出文本） | 500 场景 / 784 轮 | 784/784 | 1.89s/轮（p95 3.70s） | 30.6 轮/min | 25.5 min |
| 原生语音渲染 | 抽样 30 轮 | 30/30 | RTF 0.117 | — | 66s |
| 纯文字（文本→文本） | 1500 场景 / 2616 轮 | 见 manifest | 1.00s/轮（测速） | 59.5 轮/min | ~50 min |

模型显存：加载后 torch 分配 5.71GB / 整卡约 7GB；DuplexSession（TTS CUDA graph）额外 +0.9GB。

## 已知限制

- 本 preview 版模型卡 `RELEASE.json` 标注 `quality_passed: false`，且 `KNOWN_CASE_LIMITATIONS.json` 记录了数字单位/日期类错误；语音自然度不作为本版评测重点。
- 音频 track 的 `user_text` 直接使用数据集 GT 文本（该链路原生吃音频，不做 ASR），`user_asr_text` 为 null；多轮 history 用 GT 文本回填。
- 原生语音使用自写整句渲染（官方 `DuplexSession._render` 会被 `BC_MAX_S=0.7s` 截断，只适合附和音）。
