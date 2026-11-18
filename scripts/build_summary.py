# -*- coding: utf-8 -*-
"""Build the final results/summary.md for RoleCraft-ZH (greedy v0.1 + official-decoding v0.2).
Usage: python scripts/build_summary.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEXT_DIMS = ["relevance", "correctness", "character_consistency", "style_fidelity",
             "coherence", "scenario_adaptation", "naturalness"]
FLAGS = ["ai_self_disclosure", "break_character", "prompt_leak", "accept_wrong_identity",
         "accept_false_background", "safety_violation", "repetition", "irrelevant", "hallucination"]

TEXT_ORDER = [
    ("qwen3.5-2b", "Qwen3.5-2B（greedy）"),
    ("qwen3.5-2b-official", "Qwen3.5-2B（官方采样）"),
    ("qwen3.5-4b", "Qwen3.5-4B（greedy）"),
    ("qwen3.5-4b-official", "Qwen3.5-4B（官方采样）"),
    ("minicpm5-2b-greedy", "MiniCPM5-2B（greedy）"),
    ("minicpm5-2b-official", "MiniCPM5-2B（官方采样）"),
    ("edgeinstant-omni-2b-text", "EdgeInstant-Omni-2B（自家·greedy）"),
]
AUDIO_ORDER = [
    ("audio_pipeline_qwen3.5-2b", "B1: ASR+2B+TTS（greedy）"),
    ("audio_pipeline_qwen3.5-2b-official", "B1: ASR+2B+TTS（官方采样）"),
    ("audio_pipeline_qwen3.5-4b", "B2: ASR+4B+TTS（greedy）"),
    ("audio_pipeline_qwen3.5-4b-official", "B2: ASR+4B+TTS（官方采样）"),
    ("audio_pipeline_minicpm5-2b-greedy", "B4: ASR+MiniCPM5-2B+TTS（greedy）"),
    ("audio_pipeline_minicpm5-2b-official", "B4: ASR+MiniCPM5-2B+TTS（官方采样）"),
    ("audio_edgeinstant", "B3: EdgeInstant-Omni-2B（原生音频·greedy）"),
]

def load(name):
    p = RES / name / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def main():
    sums = {m: load(m) for m, _ in TEXT_ORDER + AUDIO_ORDER}
    aud = json.loads((RES / "audio" / "report.json").read_text(encoding="utf-8")) if (RES / "audio" / "report.json").exists() else None
    bad = json.loads((RES / "badcases" / "summary.json").read_text(encoding="utf-8")) if (RES / "badcases" / "summary.json").exists() else None
    L = ["# RoleCraft-ZH v0.3 结果汇总", "",
         "评测集：50 张角色卡 / 1500 场景（2616 轮）文字 track + 500 场景（784 轮）音频输入 track。",
         "判官：DeepSeek-V4.1-Flash（文本维度，temperature 0）+ Qwen3.8-Omni-Flash（语音维度，reasoning_effort=none）。", "",
         "两种解码配置并存：",
         "- **greedy（v0.1）**：temperature=0，无 repetition/presence penalty；可复现优先。",
         "- **官方采样（v0.2）**：Qwen3.5 用 `temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0, seed=42`；MiniCPM5-2B 用其 README 推荐 `temperature=1.0, top_p=0.95, min_p=0.0, repetition_penalty=1.05, seed=42`。",
         "- EdgeInstant-Omni-2B 保持其部署默认的 greedy（transformers generate 不支持 presence_penalty）。", "",
         "## 1. 文字 track（文本输入 → 文本输出，2616 轮）", "",
         "| 模型 | 轮数 | 总分 | " + " | ".join(TEXT_DIMS) + " | 关键点命中 |",
         "|---|---|---|" + "---|" * len(TEXT_DIMS) + "---|"]
    for slug, label in TEXT_ORDER:
        s = sums.get(slug)
        if not s:
            continue
        L.append(f"| {label} | {s['n_turns']} | **{s['overall']}** | " +
                 " | ".join(str(s["dims"].get(d)) for d in TEXT_DIMS) +
                 f" | {s['keypoint_hit_rate']} |")
    L += ["", "### 1.1 红旗率", "", "| 模型 | " + " | ".join(FLAGS) + " |", "|---|" + "---|" * len(FLAGS)]
    for slug, label in TEXT_ORDER:
        s = sums.get(slug)
        if not s:
            continue
        L.append(f"| {label} | " + " | ".join(f"{s['flags'][f]['rate']:.1%}" for f in FLAGS) + " |")
    L += ["", "## 2. 音频输入 track（语音输入，500 场景 / 784 轮）", ""]
    if aud:
        a = aud.get("asr", {})
        L.append(f"ASR：Qwen3-ASR-0.6B，字符 WER 均值 **{a.get('wer_mean')}**，中位数 {a.get('wer_median')}，"
                 f"完全一致率 {a.get('exact_rate')}，平均延迟 {a.get('latency_mean_s')}s。")
        L.append("")
        L += ["### 2.1 文本维度（与文字 track 同判官/口径）", "",
              "| 链路 | 轮数 | 总分 | 相关性 | 正确性 | 角色一致性 | 场景适配 | 关键点命中 |",
              "|---|---|---|---|---|---|---|---|"]
        for slug, label in AUDIO_ORDER:
            r = aud.get("pipelines", {}).get(slug)
            if not r:
                continue
            tj = r.get("text_judge") or {}
            if not tj:
                continue
            t = tj
            L.append(f"| {label} | {t['n_turns']} | **{t['overall']}** | {t['dims'].get('relevance')} | "
                     f"{t['dims'].get('correctness')} | {t['dims'].get('character_consistency')} | "
                     f"{t['dims'].get('scenario_adaptation')} | {sums.get(slug, {}).get('keypoint_hit_rate')} |")
        L += ["", "### 2.2 语音维度（Qwen3.8-Omni-Flash 听音频）", "",
              "| 链路 | 音频 turn | 相关性 | 正确性 | 角色一致性 | 情感匹配 | 语音清晰度 |",
              "|---|---|---|---|---|---|---|"]
        for slug, label in AUDIO_ORDER:
            r = aud.get("pipelines", {}).get(slug)
            if not r or not r.get("audio_judge"):
                continue
            aj = r["audio_judge"]; d = aj.get("dims", {})
            L.append(f"| {label} | {aj.get('n_audio_scored')}/{aj.get('n_turns')} | {d.get('relevance')} | "
                     f"{d.get('correctness')} | {d.get('character_consistency')} | {d.get('emotion_match')} | {d.get('audio_clarity')} |")
        L += ["", "### 2.3 同一批 500 场景：文字输入 vs 音频输入（文本判官总分）", "",
              "| 链路 | 文字输入 | 音频输入 | 差值 |", "|---|---|---|---|"]
        for slug, label in AUDIO_ORDER:
            r = aud.get("pipelines", {}).get(slug)
            if not r:
                continue
            tb = r.get("text_baseline_same_scenes"); tj = r.get("text_judge")
            if tb and tj:
                L.append(f"| {label} | {tb['overall']} | {tj['overall']} | {round(tj['overall'] - tb['overall'], 3)} |")
    L += ["", "## 3. 复读 / 死循环 badcase（程序化 + 判官辅助）", ""]
    if bad:
        B = bad.get("summary", {})
        L += ["| 模型 | 总轮数 | severe | moderate | suspected | 判官 repetition 率 |",
              "|---|---|---|---|---|---|"]
        for m, s in B.items():
            L.append(f"| {s.get('label', m)} | {s.get('n_turns')} | {s.get('severe')} | {s.get('moderate')} | "
                     f"{s.get('suspected')} | {s.get('judge_repetition_rate'):.1%} |")
        L += ["", "明细（同一测试项跨模型对比）见 `results/badcases/report.md`；解码配置对比见 `results/badcases/decoding_ablation.md`。"]
    L += ["", "## 4. 说明", "",
          "- B1/B2 回复音频：本地 Qwen3-TTS-0.6B 仅 20 条链路验证样本（RTF≈2）；其余 1548 条/配置用云端 qwen-audio-3.1-tts-next 渲染，只用于语音维度评分。",
          "- B3 音频 track 为 784/784 原生音频（12,614.8s，RTF 0.1503）；58.7% 回复文本含括号动作描述，原生 talker 忠实朗读（`stage_direction_read` flag），不扣清晰度分。",
          "- greedy 与官方采样两套结果并存，便于区分「模型能力」与「解码配置」的影响；EdgeInstant 只有 greedy 一套（其部署默认）。"]
    (RES / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("written", RES / "summary.md")

if __name__ == "__main__":
    main()
