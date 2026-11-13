# DATASET CARD · RoleCraft-ZH v0.1

## 基本信息
- 名称：RoleCraft-ZH（中文角色扮演全模态评测基准）
- 版本：v0.1（2026-11-14）
- 规模：50 张角色卡 / 1500 个场景 / 2616 轮用户台词 / 500 个语音场景（784 条用户 TTS 音频）/ 音频输入 track 784 轮
- 语言：简体中文
- 用途：评测端侧/小体量模型在"角色卡 → 多轮角色扮演"中的角色保持能力（身份、事实、性格、语体、关系、记忆、安全），文字 track + 音频输入 track
- 性质：内部评测集，不对外发布；不用于模型训练

## 构成
| 文件 | 内容 |
|---|---|
| data/characters/*.json | 人设卡（100–1000 字，第二人称）+ 内部事实表 + 来源 |
| data/scenes/*.json | 每卡 30 场景（类别配比固定）+ 每题期望行为/关键点/禁止项 + 语气标签 |
| data/references/*.json | DeepSeek-V4.1-Flash 生成的云端参考答案（示范，不唯一） |
| data/voice/selection.json | 每卡 10 个语音场景选择 |
| data/voice/manifest.jsonl | 逐轮 TTS 音频清单（文本/语气/文件/时长） |
| data/voice/audio/*.mp3 | 用户轮 TTS 音频（青年男声测试用户，26 种语气） |
| data/audio_track/asr_transcripts.jsonl | 784 条用户语音的 Qwen3-ASR-0.6B 转写 + WER |
| baselines/qwen3.5-2b、qwen3.5-4b、edgeinstant-omni-2b-text | 文字 track 基线原始回答（2616 轮/模型） |
| baselines/audio_pipeline_qwen3.5-2b、...4b、audio_edgeinstant | 音频输入 track 原始回答（784 轮/链路） |
| results/ | 逐轮判官结果、summary.json、summary.md、audio/report.md |
| quick_view.xlsx | 13 sheets：卡/场景/逐轮/评分/语气/语音/基线/音频 track/ASR |

## 场景配比（每卡 30）
角色追问 2、背景追问 3、故事追问 4、关系追问 4、交叉追问 3、情绪共情 2、情境演绎 1、错误角色 2、错误背景 2、重复问答 2、安全拒绝 2（含 1 个良性敏感对照）、元对话 2、记忆一致性 1。

## 构建方法
1. DeepSeek Search API 多查询召回 + Fetch API 抓取（百度百科/萌娘百科/Fandom 等）2–3 个来源；
2. DeepSeek-V4.1-Flash 仅基于抓取资料生成人设卡，自动校验 100–1000 字/字段/名称；
3. 按类别配比与 40 字/26 语气标签约束生成场景，自动校验并重生成；
4. DeepSeek-V4.1-Flash 以同一人设卡扮演角色生成参考答案（判官参考，非唯一标准）；
5. qwen-audio-3.1-tts-next 指令式 TTS，固定参考音色 + 语气标签合成用户语音；
6. 本地 Qwen3.5-2B/4B（llama.cpp Q8_0）以统一 system prompt、temperature 0 跑文字 track 基线，DeepSeek-V4.1-Flash（temperature 0）逐轮评分；
7. 音频输入 track：Qwen3-ASR-0.6B 转写 → 同 prompt 的 2B/4B 级联 → TTS 回放；EdgeInstant-Omni-2B 原生音频输入（vendor transformers 5.12.1）；语音维度用 qwen3.8-omni-flash 判分。

## 评分
维度（0–5）：相关性 / 正确性 / 角色一致性 / 语体风格 / 语境连贯 / 场景适配 / 自然度；安全性单独报告（仅安全类场景）。音频 track 另评：情感/语气匹配、语音清晰度（qwen3.8-omni-flash 听音频）。中文判官 prompt 与权重见 scripts/judge_text.py、scripts/audio_judge.py、scripts/aggregate.py。红旗：自曝 AI、破戏、提示泄漏、接受错误身份/背景、安全违规、复读、答非所问、严重编造。

## 已知局限
- 人设卡资料多来自中文百科/维基类站点，部分配角资料有限；来源链接人设卡文件内可查。
- 参考答案由单一云端模型生成，只作判官参考，不代表唯一正确答案；judge 与被判模型同源时存在偏好风险，所有模型用同一 judge，相对比较公平。
- 场景台词平均 1.74 轮，长程（5 轮）样本较少。
- 语音 track 本版只评测内容/情感/清晰度，不评音色/自然度/MOS/打断（待 talker 数据扩充后单独评测）。
- 本地 Qwen3-TTS-0.6B 在 RTX 3090 上 RTF≈2，全量 1568 条需 ~8h；只保留 20 条本地链路验证样本，全量音频判官用云端 qwen-audio-3.1-tts-next 渲染（1548 条）——语音维度会混入云端 TTS 韵律特性。
- EdgeInstant-Omni-2B 为 preview 版（模型卡 quality_passed=false），原生音频只抽样 30 条用于语音判官；其文字/音频 track 均使用与 Qwen 基线一致的角色扮演 system prompt，非模型自带 reply prompt。
- 安全类场景仅用于拒绝能力评测，不含具体有害细节。

## 版权与使用
- 角色名称、设定版权归各作品版权方所有；人设卡为基于公开资料的事实性摘要，仅内部研究/评测使用，非官方内容。
- 代码 Apache-2.0；数据仅限内部使用，未经许可不得再分发。
