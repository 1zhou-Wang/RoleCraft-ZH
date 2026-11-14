# 解码配置对复读 badcase 的影响（一次小规模消融）

v0.1 评测使用的解码配置：**temperature=0（greedy），无 repetition_penalty / presence_penalty**。
Qwen3.5 官方 README（non-thinking / text）推荐：**temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0**（注意这里的 2 是 presence_penalty，不是 repetition_penalty）。
本消融在相同上下文下对比两种配置（llama.cpp b11552，Q8_0，seed=42）。

## 1. 复读 badcase 项

| 模型 | 测试项 | Greedy 字数 / tier / 判官分 | 官方配置 字数 / tier / 判官分 |
|---|---|---|---|
| audio_pipeline_qwen3.5-2b | rc10_s14#2 | 512 / 2 / 1.35 | 107 / 0 / 3.25 |
| audio_pipeline_qwen3.5-2b | rc24_s26#1 | 82 / 0 / 5.0 | 77 / 0 / 3.9 |
| qwen3.5-2b | rc13_s14#1 | 514 / 2 / 0.8 | 70 / 0 / 2.45 |
| qwen3.5-2b | rc13_s14#2 | 514 / 2 / 0.8 | 107 / 0 / 2.25 |
| qwen3.5-2b | rc20_s04#1 | 63 / 0 / 2.35 | 63 / 0 / 1.55 |
| qwen3.5-2b | rc24_s26#1 | 82 / 0 / 5.0 | 77 / 0 / 4.0 |
| qwen3.5-2b | rc33_s10#1 | 710 / 2 / 1.6 | 94 / 0 / 4.35 |

## 2. 正常对照项

| 测试项 | Greedy 字数 / tier / 判官分 | 官方配置 字数 / tier / 判官分 |
|---|---|---|
| rc02_s01#1 | 43 / 0 / 4.0 | 54 / 0 / 3.25 |
| rc05_s01#1 | 56 / 0 / 4.7 | 95 / 0 / 3.25 |
| rc12_s01#1 | 66 / 0 / 4.55 | 59 / 0 / 4.55 |
| rc18_s01#1 | 97 / 0 / 4.25 | 57 / 0 / 3.25 |
| rc25_s01#1 | 87 / 0 / 1.7 | 82 / 0 / 1.7 |

## 3. 结论

- 官方配置在 4 个可复现的 severe 项上**全部消除死循环**（512/514/514/710 字 → 70-107 字），判官总分从 0.8-1.6 提升到 2.25-4.35。
- 正常对照项的判官分平均略降（greedy ≈3.84 vs official ≈3.40，n=5；其中 2 项从 4.0-4.7 降到 3.25），说明 presence_penalty=2.0 + T=1.0 对稳定性/精确性有一定代价，需要全量数据确认。
- 注意：greedy 下并非所有原始 badcase 都能在“重建上下文”中复现（rc24_s26#1/rc20_s04#1 本次未复现），说明这类退化对上下文/服务槽位等细节敏感；推理框架/批处理也会影响结果。
- EdgeInstant 的原生推理代码是 greedy（do_sample=False），transformers generate 不支持 presence_penalty，只能加 repetition_penalty；若要和 Qwen 官方推荐完全对齐，需要其部署侧支持 presence_penalty 采样。
- 建议：把**官方解码配置作为 v0.2 主配置**重跑全部 track，同时保留 v0.1 greedy 结果做可复现对照；或在同一批 300 场景上双配置评测后决定。
