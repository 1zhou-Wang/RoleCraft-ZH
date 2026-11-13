# RoleCraft-ZH 基线结果（文字 track）

| 模型 | 轮数 | 总分 | relevance | correctness | character_consistency | style_fidelity | coherence | scenario_adaptation | naturalness | 安全 | 关键点命中 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | 2616 | **3.608** | 3.917 | 3.436 | 3.546 | 3.735 | 3.86 | 3.104 | 3.773 | 3.672 | 0.577 |
| qwen3.5-4b | 2616 | **4.266** | 4.515 | 4.141 | 4.245 | 4.258 | 4.498 | 3.917 | 4.325 | 4.517 | 0.737 |

## 红旗率

| 模型 | ai_self_disclosure | break_character | prompt_leak | accept_wrong_identity | accept_false_background | safety_violation | repetition | irrelevant | hallucination |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-2b | 0.1% | 4.5% | 0.3% | 8.6% | 8.4% | 0.5% | 10.1% | 2.9% | 12.5% |
| qwen3.5-4b | 0.0% | 0.8% | 0.0% | 2.5% | 4.2% | 0.2% | 2.0% | 0.3% | 6.0% |
