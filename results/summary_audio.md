# RoleCraft-ZH 基线结果（文字 track）

| 模型 | 轮数 | 总分 | relevance | correctness | character_consistency | style_fidelity | coherence | scenario_adaptation | naturalness | 安全 | 关键点命中 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| audio_pipeline_qwen3.5-2b | 784 | **3.671** | 3.964 | 3.536 | 3.62 | 3.756 | 3.939 | 3.152 | 3.798 | 3.229 | 0.606 |
| audio_pipeline_qwen3.5-4b | 784 | **4.313** | 4.527 | 4.224 | 4.292 | 4.298 | 4.541 | 3.973 | 4.347 | 4.52 | 0.758 |
| audio_edgeinstant | 784 | **3.622** | 3.977 | 3.565 | 3.556 | 3.546 | 3.903 | 3.21 | 3.578 | 3.673 | 0.637 |

## 红旗率

| 模型 | ai_self_disclosure | break_character | prompt_leak | accept_wrong_identity | accept_false_background | safety_violation | repetition | irrelevant | hallucination |
|---|---|---|---|---|---|---|---|---|---|
| audio_pipeline_qwen3.5-2b | 0.3% | 4.0% | 0.5% | 12.1% | 4.5% | 1.5% | 11.2% | 4.2% | 9.8% |
| audio_pipeline_qwen3.5-4b | 0.0% | 0.6% | 0.0% | 5.1% | 1.4% | 0.5% | 4.3% | 1.4% | 4.7% |
| audio_edgeinstant | 0.1% | 5.6% | 0.8% | 11.7% | 4.1% | 1.3% | 16.4% | 4.1% | 12.9% |
