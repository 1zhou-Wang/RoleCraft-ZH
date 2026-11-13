#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 baselines/<model> 的异常样例（只读，供报告/人工抽检用）。

输出:
  * 逐模型场景/turn 统计、回答长度分布、延迟分位
  * 关键词命中（AI自曝/系统提示/角色卡/评测等）样例
  * 最短/最长回答样例

用法: python local/anomaly_scan.py --model qwen3.5-2b [--top 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
KEYWORDS = ["AI", "ai", "人工智能", "语言模型", "大模型", "系统提示", "角色卡", "提示词",
            "prompt", "评测", "benchmark", "作为一个", "根据设定", "人设卡"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    base = ROOT / "baselines" / a.model
    rows = []
    for f in sorted(base.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    turns = [(r["scene_id"], t) for r in rows for t in r["turns"]]
    hits = []
    for sid, t in turns:
        text = t["assistant_text"]
        for kw in KEYWORDS:
            if kw in text:
                hits.append({"scene_id": sid, "keyword": kw, "text": text[:200]})
                break
    by_len = sorted(turns, key=lambda x: len(x[1]["assistant_text"]))
    report = {
        "model": a.model,
        "scenes": len(rows),
        "turns": len(turns),
        "keyword_hits": len(hits),
        "keyword_samples": hits[: a.top],
        "shortest": [{"scene_id": sid, "len": len(t["assistant_text"]), "text": t["assistant_text"][:120]}
                     for sid, t in by_len[: a.top]],
        "longest": [{"scene_id": sid, "len": len(t["assistant_text"]), "text": t["assistant_text"][:300]}
                    for sid, t in by_len[-a.top:]],
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
