#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 baselines/<model>/<card_id>.jsonl 是否与 data/ 场景一致（只读）。

检查项:
  * 50 个 card 文件都在、每卡 30 行
  * 每行 (card_id, scene_id, model) 正确；turns 数量/idx/user_text 与场景文件一致
  * assistant_text 非空、无 <think> 泄漏、无 error 字段
  * 统计逐轮延迟与回答长度

用法:
  python local/verify_baseline.py --model qwen3.5-2b
  python local/verify_baseline.py --model qwen3.5-4b --strict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="baselines/<model> 下的模型目录名，如 qwen3.5-2b")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    base = ROOT / "baselines" / a.model
    errors: list[str] = []
    scene_files = sorted((DATA / "scenes").glob("*.json"))
    if not base.is_dir():
        print(f"[ERR] 目录不存在: {base}")
        return 1

    total_scenes = 0
    total_turns = 0
    latencies: list[float] = []
    reply_lens: list[int] = []
    reply_min = 10 ** 9
    reply_max = 0
    think_leaks = 0
    empty_replies = 0

    for sf in scene_files:
        cid = sf.stem
        scenes = json.loads(sf.read_text(encoding="utf-8")).get("scenes") or []
        exp = {s["scene_id"]: s for s in scenes}
        path = base / f"{cid}.jsonl"
        if not path.is_file():
            errors.append(f"{cid}.jsonl 缺失")
            continue
        recs: dict[str, dict] = {}
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{cid}.jsonl:{lineno} JSON 解析失败: {exc}")
                continue
            recs[r.get("scene_id")] = r
        if len(recs) != 30:
            errors.append(f"{cid}.jsonl 场景数 {len(recs)} != 30")
        for sid, scene in exp.items():
            r = recs.get(sid)
            if r is None:
                errors.append(f"{cid}: 缺场景 {sid}")
                continue
            total_scenes += 1
            if r.get("card_id") != cid or r.get("model") != a.model:
                errors.append(f"{sid}: card_id/model 字段错误")
            if "error" in r:
                errors.append(f"{sid}: 有 error 字段: {r['error'][:80]}")
            got = r.get("turns") or []
            if len(got) != len(scene["turns"]):
                errors.append(f"{sid}: turns {len(got)} != {len(scene['turns'])}")
            for gt, st in zip(got, scene["turns"]):
                total_turns += 1
                if gt.get("user_text") != st.get("user_text"):
                    errors.append(f"{sid}: user_text 与数据不一致")
                txt = gt.get("assistant_text") or ""
                if not txt.strip():
                    empty_replies += 1
                    errors.append(f"{sid}: 空 assistant_text (turn {gt.get('idx')})")
                if "<think>" in txt or "</think>" in txt:
                    think_leaks += 1
                    errors.append(f"{sid}: assistant_text 含 think 标签")
                reply_lens.append(len(txt))
                reply_min = min(reply_min, len(txt))
                reply_max = max(reply_max, len(txt))
                if gt.get("latency_s") is not None:
                    latencies.append(float(gt["latency_s"]))

    lat = sorted(latencies)
    report = {
        "model": a.model,
        "scenes": total_scenes,
        "turns": total_turns,
        "empty_replies": empty_replies,
        "think_leaks": think_leaks,
        "turn_latency_s": {
            "mean": round(sum(lat) / len(lat), 3) if lat else 0,
            "p50": round(lat[len(lat) // 2], 3) if lat else 0,
            "p95": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 3) if lat else 0,
            "max": round(lat[-1], 3) if lat else 0,
        },
        "reply_chars": {
            "mean": round(sum(reply_lens) / len(reply_lens), 1) if reply_lens else 0,
            "min": reply_min if reply_lens else 0,
            "max": reply_max,
        },
        "errors": errors[:40],
        "error_count": len(errors),
    }
    print(json.dumps({k: report[k] for k in
                      ("model", "scenes", "turns", "empty_replies", "think_leaks", "turn_latency_s", "reply_chars")},
                     ensure_ascii=False, indent=1))
    for e in errors[:20]:
        print("[ERR]", e)
    print(f"errors={len(errors)}")
    return 0 if not errors else (1 if a.strict else 0)


if __name__ == "__main__":
    sys.exit(main())
