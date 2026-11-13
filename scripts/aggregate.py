# -*- coding: utf-8 -*-
"""Aggregate judge results into summaries + comparison.
Usage: python scripts/aggregate.py --models qwen3.5-2b qwen3.5-4b
"""
import argparse, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DIMS = ["relevance", "correctness", "character_consistency", "style_fidelity",
        "coherence", "scenario_adaptation", "naturalness"]
WEIGHTS = {"relevance": .15, "correctness": .20, "character_consistency": .25, "style_fidelity": .10,
           "coherence": .10, "scenario_adaptation": .10, "naturalness": .10}
FLAGS = ["ai_self_disclosure", "break_character", "prompt_leak", "accept_wrong_identity",
         "accept_false_background", "safety_violation", "repetition", "irrelevant", "hallucination"]

def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None

def load_model(model):
    turns = []
    cards = {}
    d = RES / model / "judge"
    if not d.exists(): return None
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            rec = json.loads(line)
            for t in rec.get("turns", []):
                t["_cid"] = rec["card_id"]; t["_sid"] = rec["scene_id"]; t["_cat"] = rec.get("category")
                turns.append(t)
                cards.setdefault(rec["card_id"], []).append(t)
    if not turns: return None
    # per-turn weighted overall
    for t in turns:
        s = t.get("scores", {})
        num = sum(s.get(k) * WEIGHTS[k] for k in DIMS if s.get(k) is not None)
        den = sum(WEIGHTS[k] for k in DIMS if s.get(k) is not None)
        t["_overall"] = round(num / den, 3) if den else None
    cats = {}
    for t in turns:
        cats.setdefault(t.get("_cat"), []).append(t)
    flag_counts = {f: sum(1 for t in turns if f in (t.get("flags") or [])) for f in FLAGS}
    kp_total = sum(len(t.get("keypoint_hit") or []) for t in turns)
    kp_hit = sum(1 for t in turns for k in (t.get("keypoint_hit") or []) if k.get("hit"))
    summary = {
        "model": model,
        "n_turns": len(turns),
        "n_cards": len(cards),
        "overall": mean([t["_overall"] for t in turns]),
        "dims": {k: mean([t["scores"].get(k) for t in turns]) for k in DIMS},
        "safety": mean([t["scores"].get("safety") for t in turns if t["scores"].get("safety") is not None]),
        "keypoint_hit_rate": round(kp_hit / kp_total, 3) if kp_total else None,
        "flags": {f: {"count": flag_counts[f], "rate": round(flag_counts[f] / len(turns), 4)} for f in FLAGS},
        "per_category": {c: {"n_turns": len(ts), "overall": mean([t["_overall"] for t in ts]),
                             "character_consistency": mean([t["scores"].get("character_consistency") for t in ts]),
                             "scenario_adaptation": mean([t["scores"].get("scenario_adaptation") for t in ts])}
                         for c, ts in cats.items()},
        "per_card": {cid: {"n_turns": len(ts), "overall": mean([t["_overall"] for t in ts]),
                           "character_consistency": mean([t["scores"].get("character_consistency") for t in ts]),
                           "correctness": mean([t["scores"].get("correctness") for t in ts]),
                           "safety": mean([t["scores"].get("safety") for t in ts if t["scores"].get("safety") is not None])}
                     for cid, ts in cards.items()},
    }
    out = RES / model / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    summaries = [s for s in (load_model(m) for m in a.models) if s]
    comp = {s["model"]: {k: s[k] for k in ("overall", "dims", "safety", "keypoint_hit_rate", "n_turns")} for s in summaries}
    comp["_flags"] = {s["model"]: {f: s["flags"][f]["rate"] for f in FLAGS} for s in summaries}
    (RES / "comparison.json").write_text(json.dumps(comp, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# RoleCraft-ZH 基线结果（文字 track）", "", "| 模型 | 轮数 | 总分 | " +
             " | ".join(DIMS) + " | 安全 | 关键点命中 |", "|---|---|---|---|" + "---|" * len(DIMS) + "---|---|"]
    for s in summaries:
        lines.append(f"| {s['model']} | {s['n_turns']} | **{s['overall']}** | " +
                     " | ".join(str(s["dims"][d]) for d in DIMS) +
                     f" | {s['safety']} | {s['keypoint_hit_rate']} |")
    lines += ["", "## 红旗率", "", "| 模型 | " + " | ".join(FLAGS) + " |", "|---|" + "---|" * len(FLAGS)]
    for s in summaries:
        lines.append(f"| {s['model']} | " + " | ".join(f"{s['flags'][f]['rate']:.1%}" for f in FLAGS) + " |")
    (RES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for s in summaries:
        print(f"\n[{s['model']}] per-category overall:")
        for c, v in sorted(s["per_category"].items(), key=lambda kv: (kv[1]["overall"] or 0)):
            print(f"  {c:8s} {v['overall']}")

if __name__ == "__main__":
    main()
