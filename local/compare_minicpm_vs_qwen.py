#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MiniCPM5-2B vs Qwen3.5-2B/4B 同口径对照（文字 track + 音频级联）。

输出: logs/minicpm5/minicpm_vs_qwen.json
用法: python local/compare_minicpm_vs_qwen.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINES = ROOT / "baselines"
LOG_DIR = ROOT / "logs" / "minicpm5"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPEAT_RE = re.compile(r"(.{4,20}?)\1{2,}")
ROWS = [
    # (label, group, dir)
    ("qwen3.5-2b greedy", "text", "qwen3.5-2b"),
    ("qwen3.5-2b official", "text", "qwen3.5-2b-official"),
    ("qwen3.5-4b greedy", "text", "qwen3.5-4b"),
    ("qwen3.5-4b official", "text", "qwen3.5-4b-official"),
    ("minicpm5-2b greedy", "text", "minicpm5-2b-greedy"),
    ("minicpm5-2b official", "text", "minicpm5-2b-official"),
    ("qwen3.5-2b audio greedy (B1)", "audio", "audio_pipeline_qwen3.5-2b"),
    ("qwen3.5-2b audio official (B1)", "audio", "audio_pipeline_qwen3.5-2b-official"),
    ("qwen3.5-4b audio greedy (B2)", "audio", "audio_pipeline_qwen3.5-4b"),
    ("qwen3.5-4b audio official (B2)", "audio", "audio_pipeline_qwen3.5-4b-official"),
    ("minicpm5-2b audio greedy", "audio", "audio_pipeline_minicpm5-2b-greedy"),
    ("minicpm5-2b audio official", "audio", "audio_pipeline_minicpm5-2b-official"),
]


def scan(slug: str) -> dict:
    base = BASELINES / slug
    scenes = turns = 0
    lens = []
    lat = []
    repeats = 0
    errors = 0
    for path in sorted(base.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            scenes += 1
            if rec.get("error"):
                errors += 1
            for t in rec.get("turns") or []:
                turns += 1
                txt = (t.get("assistant_text") or "").strip()
                if txt:
                    lens.append(len(txt))
                    if REPEAT_RE.search(txt):
                        repeats += 1
                if t.get("latency_s") is not None:
                    lat.append(float(t["latency_s"]))
    return {
        "slug": slug,
        "scenes": scenes,
        "scenes_failed": errors,
        "turns": turns,
        "reply_chars_mean": round(sum(lens) / len(lens), 1) if lens else 0.0,
        "reply_chars_max": max(lens) if lens else 0,
        "reply_gt_300": sum(1 for x in lens if x > 300),
        "repetition_rate_pct": round(repeats / len(lens) * 100, 2) if lens else 0.0,
        "turn_latency_mean_s": round(sum(lat) / len(lat), 3) if lat else 0.0,
    }


def main() -> int:
    out = {"rows": [], "generated_at": json.loads(json.dumps(__import__("time").strftime("%Y-%m-%dT%H:%M:%S")))}
    for label, group, slug in ROWS:
        if not (BASELINES / slug).is_dir():
            continue
        s = scan(slug)
        s.update({"label": label, "group": group})
        out["rows"].append(s)
        print(f"[{group}] {label:36s} turns={s['turns']:5d} chars={s['reply_chars_mean']:6.1f} "
              f"max={s['reply_chars_max']:4d} rep={s['repetition_rate_pct']:.2f}% "
              f"lat={s['turn_latency_mean_s']:.3f}s")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    p = LOG_DIR / "minicpm_vs_qwen.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
