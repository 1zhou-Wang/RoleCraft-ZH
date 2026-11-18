# DATASET CARD · RoleCraft-ZH v0.4

## 基本信息
- 名称：RoleCraft-ZH（中文角色扮演全模态评测基准）
- 版本：v0.4（2026-11-19）：在 v0.1 greedy / v0.2 Qwen 官方采样 / v0.3 MiniCPM5-2B 基础上，新增 **Qwen2.5-Omni-3B**（transformers 原生音频输入 + 文字 track，greedy）
- 规模：50 张角色卡 / 1500 个场景 / 2616 轮用户台词 / 500 个语音场景（784 条用户 TTS 音频）/ 音频输入 track 784 轮 / B3 原生语音 784/784 轮（3h30m）
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
| baselines/qwen3.5-2b、qwen3.5-4b、edgeinstant-omni-2b-text | 文字 track 基线原始回答（greedy，2616 轮/模型） |
| baselines/qwen3.5-2b-official、qwen3.5-4b-official | 文字 track 官方采样基线（2616 轮/模型） |
| baselines/minicpm5-2b-greedy、minicpm5-2b-official | MiniCPM5-2B 文字 track（greedy / 官方采样，2616 轮/配置） |
| baselines/audio_pipeline_qwen3.5-2b、...4b、audio_edgeinstant | 音频输入 track 原始回答（greedy，784 轮/链路；B3 为原生音频） |
| baselines/audio_pipeline_qwen3.5-2b-official、...4b-official | 音频输入 track 官方采样级联（784 轮/链路，回复音频已重渲） |
| baselines/audio_pipeline_minicpm5-2b-{greedy,official} | MiniCPM5-2B 音频级联（784 轮/配置，回复音频已重渲） |\n| baselines/qwen25-omni-3b-greedy | Qwen2.5-Omni-3B 文字 track（greedy，2616 轮） |\n| baselines/audio_pipeline_qwen25-omni-3b-greedy | Qwen2.5-Omni-3B 原生音频输入 track（784 轮，回复音频已重渲） |
| results/ | 逐轮判官结果、summary.json、summary.md、audio/report.md、badcases/report.md |
| results/badcases/ | 复读/死循环 badcase 记录（repetition_flags.jsonl + summary.json + report.md） |
| quick_view.xlsx | 16 sheets：卡/场景/逐轮/评分/语气/语音/基线/音频 track/ASR/复读 badcase/解码对比 |

## 场景配比（每卡 30）
角色追问 2、背景追问 3、故事追问 4、关系追问 4、交叉追问 3、情绪共情 2、情境演绎 1、错误角色 2、错误背景 2、重复问答 2、安全拒绝 2（含 1 个良性敏感对照）、元对话 2、记忆一致性 1。

## 构建方法
1. DeepSeek Search API 多查询召回 + Fetch API 抓取（百度百科/萌娘百科/Fandom 等）2–3 个来源；
2. DeepSeek-V4.1-Flash 仅基于抓取资料生成人设卡，自动校验 100–1000 字/字段/名称；
3. 按类别配比与 40 字/26 语气标签约束生成场景，自动校验并重生成；
4. DeepSeek-V4.1-Flash 以同一人设卡扮演角色生成参考答案（判官参考，非唯一标准）；
5. qwen-audio-3.1-tts-next 指令式 TTS，固定参考音色 + 语气标签合成用户语音；
6. 本地 Qwen3.5-2B/4B（llama.cpp Q8_0）以统一 system prompt 跑文字 track 基线：greedy（v0.1）与官方采样（v0.2，Qwen README 推荐）两套；MiniCPM5-2B（官方 GGUF Q8_0，标准 llama.cpp 加载）同样跑 greedy 与官方采样（T=1.0, top_p=0.95, min_p=0.0, repetition_penalty=1.05）两套；Qwen2.5-Omni-3B（ModelScope 官方 BF16，transformers 5.19）跑 greedy 文字 track；DeepSeek-V4.1-Flash（temperature 0）逐轮评分；
7. 音频输入 track：Qwen3-ASR-0.6B 转写 → 同 prompt 的 2B/4B/MiniCPM5-2B 级联 → TTS 回放；EdgeInstant-Omni-2B 原生音频输入（vendor transformers 5.12.1），784/784 轮用原生 talker 渲染 24kHz wav（RTF 0.1503）；Qwen2.5-Omni-3B 原生音频输入（transformers，文本输出 + 云端 TTS 回放），784/784 轮；语音维度用 qwen3.8-omni-flash 判分；
8. 复读/死循环 badcase 检测：程序化指标（连续重复单元、同字连续、唯一字符率、zlib 压缩比、连续重复句、长文本退化）∪ 判官 repetition flag，分 severe/moderate/suspected，跨 16 条评测链路对齐同一测试项输出 `results/badcases/report.md`。

## 评分
维度（0–5）：相关性 / 正确性 / 角色一致性 / 语体风格 / 语境连贯 / 场景适配 / 自然度；安全性单独报告（仅安全类场景）。音频 track 另评：情感/语气匹配、语音清晰度（qwen3.8-omni-flash 听音频）。中文判官 prompt 与权重见 scripts/judge_text.py、scripts/audio_judge.py、scripts/aggregate.py。红旗：自曝 AI、破戏、提示泄漏、接受错误身份/背景、安全违规、复读、答非所问、严重编造。

## 已知局限
- 人设卡资料多来自中文百科/维基类站点，部分配角资料有限；来源链接人设卡文件内可查。
- 参考答案由单一云端模型生成，只作判官参考，不代表唯一正确答案；judge 与被判模型同源时存在偏好风险，所有模型用同一 judge，相对比较公平。
- 场景台词平均 1.74 轮，长程（5 轮）样本较少。
- 语音 track 本版只评测内容/情感/清晰度，不评音色/自然度/MOS/打断（待 talker 数据扩充后单独评测）。
- 本地 Qwen3-TTS-0.6B 在 RTX 3090 上 RTF≈2，全量 1568 条需 ~8h；只保留 20 条本地链路验证样本，全量音频判官用云端 qwen-audio-3.1-tts-next 渲染——语音维度会混入云端 TTS 韵律特性。
- EdgeInstant-Omni-2B 为 preview 版（模型卡 quality_passed=false）；文字/音频 track 均使用与 Qwen 基线一致的角色扮演 system prompt，非模型自带 reply prompt；音频 track 已全量 784/784 原生语音（3h30m），其中 58.7% 回复文本含括号动作描述，原生 talker 会忠实朗读（judge 标 `stage_direction_read`，不扣清晰度分）。
- 复读 badcase 检测是启发式 + 判官辅助，不是完美分类器；severe 列表可高置信参考，moderate/suspected 建议人工复核。
- 解码配置对角色扮演质量影响显著：Qwen 官方采样（T=1.0+presence_penalty=2.0）可把复读 severe 全部消除（2B 5→0、级联 2→0），但角色一致性/正确性下降（2B 总分 3.608→3.112），破戏/编造红旗率上升；greedy 反之。MiniCPM5-2B 官方采样（T=1.0+repeat_penalty=1.05）把 severe 从 11 降到 1（文字）/6 降到 2（级联），但总分从 3.274 降到 2.898。两套结果并存供选择。
- MiniCPM5-2B 通用能力较强但角色扮演/人设保持明显弱于 Qwen3.5-2B（3.274 vs 3.608），且 greedy 下死循环最多（11 条 severe）；已按官方建议 `min_p=0.0` + `repeat_penalty=1.05` 校正解码。\n- Qwen2.5-Omni-3B 原生音频输入的文字 track 3.433 / 音频 track 3.488 / 语音 overall 4.001：正确性与连贯性较好、语音清晰度 4.94 最高，但角色一致性 3.20、语体 2.80、破戏率高（语音维度 21.3%）；另有 7 条文字 severe / 3 条音频 severe 复读，长篇输出（max 1484 字）需注意。
- 安全类场景仅用于拒绝能力评测，不含具体有害细节。

## 版权与使用
- 角色名称、设定版权归各作品版权方所有；人设卡为基于公开资料的事实性摘要，仅内部研究/评测使用，非官方内容。
- 代码 Apache-2.0；数据仅限内部使用，未经许可不得再分发。
