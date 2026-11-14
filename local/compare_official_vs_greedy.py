#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 v0.1 greedy 与 v0.2 official 解码的文字/级联基线质量指标。

指标:
  * scenes / turns / 失败
  * 回复长度 mean/max、>=300 字比例（接近 max_tokens 截断的代理）
  * 复读退化率：回复中存在长度>=5 的子串连续重复 >=3 次
  * 平均延迟
输出: logs/audio_track/official_vs_greedy_compare.json（并打印）

用法: python local/compare_official_vs_greedy.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINES = ROOT / "baselines"
LOG_DIR = ROOT / "logs" / "audio_track"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PAIRS = [
    ("text-2b", "qwen3.5-2b", "qwen3.5-2b-official"),
    ("text-4b", "qwen3.5-4b", "qwen3.5-4b-official"),
    ("audio-B1", "audio_pipeline_qwen3.5-2b", "audio_pipeline_qwen3.5-2b-official"),
    ("audio-B2", "audio_pipeline_qwen3.5-4b", "audio_pipeline_qwen3.5-4b-official"),
]

REPEAT_RE = re.compile(r"(.{4,20}?)\1{2,}")


def scan(slug: str) -> dict:
    base = BASELINES / slug
    scenes = turns = turns_ok = err_scenes = 0
    lens = []
    lat = []
    repeats = 0
    for path in sorted(base.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            scenes += 1
            if rec.get("error"):
                err_scenes += 1
            for t in rec.get("turns") or []:
                turns += 1
                txt = (t.get("assistant_text") or "").strip()
                if txt:
                    turns_ok += 1
                    lens.append(len(txt))
                    if REPEAT_RE.search(txt):
                        repeats += 1
                if t.get("latency_s") is not None:
                    lat.append(float(t["latency_s"]))
    return {
        "slug": slug,
        "scenes": scenes,
        "scenes_failed": err_scenes,
        "turns": turns,
        "turns_ok": turns_ok,
        "reply_chars_mean": round(sum(lens) / len(lens), 1) if lens else 0.0,
        "reply_chars_max": max(lens) if lens else 0,
        "reply_gt_300": sum(1 for x in lens if x > 300),
        "reply_gt_300_pct": round(sum(1 for x in lens if x > 300) / len(lens) * 100, 2) if lens else 0.0,
        "repetition_degenerate": repeats,
        "repetition_rate_pct": round(repeats / len(lens) * 100, 2) if lens else 0.0,
        "turn_latency_mean_s": round(sum(lat) / len(lat), 3) if lat else 0.0,
    }


def main() -> int:
    out = {"pairs": [], "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for label, greedy_slug, official_slug in PAIRS:
        g = scan(greedy_slug)
        o = scan(official_slug)
        out["pairs"].append({"label": label, "greedy_v01": g, "official_v02": o})
        print(f"[{label}] greedy: chars_mean={g['reply_chars_mean']} gt300={g['reply_gt_300_pct']}% "
              f"rep={g['repetition_rate_pct']}% turns={g['turns_ok']} | "
              f"official: chars_mean={o['reply_chars_mean']} gt300={o['reply_gt_300_pct']}% "
              f"rep={o['repetition_rate_pct']}% turns={o['turns_ok']}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    p = LOG_DIR / "official_vs_greedy_compare.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("->", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
