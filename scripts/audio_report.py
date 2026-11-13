# -*- coding: utf-8 -*-
"""Aggregate audio-input track: ASR WER + text-dim judge + audio-dim judge, compared with text-input baseline on the same 500 scenes.
Usage: python scripts/audio_report.py
"""
import json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE = ROOT / "baselines"
RES = ROOT / "results"
OUT = RES / "audio"

TEXT_DIMS = ["relevance", "correctness", "character_consistency", "style_fidelity",
             "coherence", "scenario_adaptation", "naturalness"]
WEIGHTS = {"relevance": .15, "correctness": .20, "character_consistency": .25, "style_fidelity": .10,
           "coherence": .10, "scenario_adaptation": .10, "naturalness": .10}
AUDIO_DIMS = ["relevance", "correctness", "character_consistency", "emotion_match", "audio_clarity"]

PIPELINES = [
    {"slug": "audio_pipeline_qwen3.5-2b", "label": "ASR+Qwen3.5-2B+TTS", "text_baseline": "qwen3.5-2b"},
    {"slug": "audio_pipeline_qwen3.5-4b", "label": "ASR+Qwen3.5-4B+TTS", "text_baseline": "qwen3.5-4b"},
    {"slug": "audio_edgeinstant", "label": "EdgeInstant-Omni-2B", "text_baseline": None},
]

def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None

def load_text_judge(slug):
    d = RES / slug / "judge"
    turns = []
    if not d.exists(): return None
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            rec = json.loads(line)
            for t in rec.get("turns", []):
                t["_sid"] = rec.get("scene_id"); t["_cat"] = rec.get("category")
                turns.append(t)
    if not turns: return None
    for t in turns:
        s = t.get("scores", {})
        num = sum(s.get(k) * WEIGHTS[k] for k in TEXT_DIMS if s.get(k) is not None)
        den = sum(WEIGHTS[k] for k in TEXT_DIMS if s.get(k) is not None)
        t["_overall"] = round(num / den, 3) if den else None
    return turns

def load_audio_judge(slug):
    d = RES / "audio" / slug / "judge_audio"
    turns = []
    if not d.exists(): return None
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            rec = json.loads(line)
            for t in rec.get("turns", []):
                turns.append(t)
    if not turns: return None
    for t in turns:
        s = t.get("scores") or {}
        num = sum(s.get(k) * WEIGHTS[k] for k in ("relevance", "correctness", "character_consistency") if s.get(k) is not None)
        den = sum(WEIGHTS[k] for k in ("relevance", "correctness", "character_consistency") if s.get(k) is not None)
        t["_text_overall"] = round(num / den, 3) if den else None
        num = sum(s.get(k) * WEIGHTS[k] for k in ("relevance", "correctness", "character_consistency") if s.get(k) is not None)
    return turns

def load_voice_scene_ids():
    sel = json.loads((DATA / "voice" / "selection.json").read_text(encoding="utf-8"))
    return {sid for it in sel["items"] for sid in it["scene_ids"]}

def text_baseline_on_voice(slug, voice_ids):
    """Text-input baseline scores restricted to the same 500 voice scenes."""
    d = RES / slug / "judge"
    vals, dims = [], {k: [] for k in TEXT_DIMS}
    if not d.exists(): return None
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            rec = json.loads(line)
            if rec.get("scene_id") not in voice_ids: continue
            for t in rec.get("turns", []):
                s = t.get("scores", {})
                num = sum(s.get(k) * WEIGHTS[k] for k in TEXT_DIMS if s.get(k) is not None)
                den = sum(WEIGHTS[k] for k in TEXT_DIMS if s.get(k) is not None)
                if den: vals.append(num / den)
                for k in TEXT_DIMS: dims[k].append(s.get(k))
    if not vals: return None
    return {"n_turns": len(vals), "overall": mean(vals), "dims": {k: mean(v) for k, v in dims.items()}}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    voice_ids = load_voice_scene_ids()
    # ASR
    asr = [json.loads(l) for l in (DATA / "audio_track" / "asr_transcripts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] if (DATA / "audio_track" / "asr_transcripts.jsonl").exists() else []
    asr_stats = {"n": len(asr), "wer_mean": mean([r.get("wer") for r in asr]),
                 "wer_median": round(statistics.median([r["wer"] for r in asr if r.get("wer") is not None]), 3) if asr else None,
                 "exact_rate": round(sum(1 for r in asr if (r.get("wer") or 0) == 0) / len(asr), 3) if asr else None,
                 "latency_mean_s": mean([r.get("latency_s") for r in asr])}
    report = {"asr": asr_stats, "pipelines": {}, "text_baseline_on_voice": {}}
    for p in PIPELINES:
        slug = p["slug"]
        tj = load_text_judge(slug)
        aj = load_audio_judge(slug)
        rep = {"label": p["label"], "text_judge": None, "audio_judge": None}
        if tj:
            rep["text_judge"] = {
                "n_turns": len(tj), "overall": mean([t["_overall"] for t in tj]),
                "dims": {k: mean([t["scores"].get(k) for t in tj]) for k in TEXT_DIMS},
                "flags": {f: round(sum(1 for t in tj if f in (t.get("flags") or [])) / len(tj), 4)
                          for f in ["ai_self_disclosure", "break_character", "prompt_leak", "accept_wrong_identity",
                                    "accept_false_background", "safety_violation", "repetition", "irrelevant", "hallucination"]},
            }
        if aj:
            rep["audio_judge"] = {
                "n_turns": len(aj), "n_audio_scored": sum(1 for t in aj if t.get("audio_scored")),
                "dims": {k: mean([(t.get("scores") or {}).get(k) for t in aj]) for k in AUDIO_DIMS},
            }
        if p["text_baseline"]:
            rep["text_baseline_same_scenes"] = text_baseline_on_voice(p["text_baseline"], voice_ids)
        report["pipelines"][slug] = rep
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# 音频输入 track 结果（v0.1）", "",
             f"ASR（Qwen3-ASR-0.6B，{asr_stats['n']} 条用户语音）：字符 WER 均值 **{asr_stats['wer_mean']}**，完全一致率 {asr_stats['exact_rate']}，平均延迟 {asr_stats['latency_mean_s']}s", "",
             "| 链路 | 文本判官总分 | 相关性 | 正确性 | 角色一致性 | 场景适配 | 音频判官-相关性 | 音频-正确性 | 音频-角色 | 情感匹配 | 清晰度 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in PIPELINES:
        r = report["pipelines"][p["slug"]]
        tj = r.get("text_judge") or {}; aj = r.get("audio_judge") or {}
        td = tj.get("dims", {}); ad = aj.get("dims", {})
        lines.append(f"| {p['label']} | {tj.get('overall')} | {td.get('relevance')} | {td.get('correctness')} | "
                     f"{td.get('character_consistency')} | {td.get('scenario_adaptation')} | "
                     f"{ad.get('relevance')} | {ad.get('correctness')} | {ad.get('character_consistency')} | "
                     f"{ad.get('emotion_match')} | {ad.get('audio_clarity')} |")
    lines += ["", "## 同 500 语音场景：文字输入 vs 音频输入（文本判官）", "",
              "| 场景 | 文字输入 | 音频输入 | 差值 |", "|---|---|---|---|"]
    for p in PIPELINES:
        r = report["pipelines"][p["slug"]]
        tb = r.get("text_baseline_same_scenes"); tj = r.get("text_judge")
        if tb and tj:
            diff = round((tj["overall"] or 0) - (tb["overall"] or 0), 3)
            lines.append(f"| {p['label']} | {tb['overall']} | {tj['overall']} | {diff} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
