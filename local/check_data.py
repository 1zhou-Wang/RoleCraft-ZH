#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoleCraft-ZH 数据就绪自检（只读，输出到 stdout / local/_data_report.json）。

检查:
  * data/characters/*.json 与 data/scenes/*.json 数量/配对
  * 每卡 30 场景、场景 turn 数 1-5、user_text 非空且 ≤40 字
  * scene_id / card_id 一致性、重复 scene_id、persona_card 是否存在
  * 汇总 turn 数（用于估算基线耗时）

用法: python local/check_data.py [--strict]
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
    ap.add_argument("--strict", action="store_true", help="有 error 时退出码 1")
    a = ap.parse_args()

    errors: list[str] = []
    warns: list[str] = []

    card_files = sorted((DATA / "characters").glob("*.json"))
    scene_files = sorted((DATA / "scenes").glob("*.json"))
    cards = {}
    for f in card_files:
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"characters/{f.name}: JSON 解析失败: {exc}")
            continue
        if not c.get("persona_card"):
            errors.append(f"characters/{f.name}: 缺 persona_card")
        cid = c.get("card_id") or f.stem
        if cid != f.stem:
            warns.append(f"characters/{f.name}: card_id={cid} 与文件名不一致（runner 用文件名 stem）")
        cards[f.stem] = c

    scenes_by_card: dict[str, list[dict]] = {}
    seen_scene_ids: set[str] = set()
    turn_hist: dict[int, int] = {}
    total_turns = 0
    total_scenes = 0
    for f in scene_files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"scenes/{f.name}: JSON 解析失败: {exc}")
            continue
        if f.stem not in cards:
            errors.append(f"scenes/{f.name}: 没有对应的 characters/{f.stem}.json")
        if d.get("card_id") not in (None, f.stem):
            warns.append(f"scenes/{f.name}: card_id={d.get('card_id')} 与文件名不一致")
        scenes = d.get("scenes") or []
        scenes_by_card[f.stem] = scenes
        if len(scenes) != 30:
            errors.append(f"scenes/{f.name}: 场景数 {len(scenes)} != 30")
        total_scenes += len(scenes)
        for s in scenes:
            sid = s.get("scene_id")
            if not sid:
                errors.append(f"scenes/{f.name}: 场景缺 scene_id")
                continue
            if sid in seen_scene_ids:
                errors.append(f"重复 scene_id: {sid}")
            seen_scene_ids.add(sid)
            turns = s.get("turns") or []
            if not (1 <= len(turns) <= 5):
                errors.append(f"{sid}: turn 数 {len(turns)} 不在 1-5")
            turn_hist[len(turns)] = turn_hist.get(len(turns), 0) + 1
            total_turns += len(turns)
            for t in turns:
                ut = str(t.get("user_text") or "").strip()
                if not ut:
                    errors.append(f"{sid}: 空 user_text")
                elif len(ut) > 40:
                    errors.append(f"{sid}: user_text 长度 {len(ut)} > 40")
                if "assistant_text" in t:
                    errors.append(f"{sid}: 场景文件不应包含 assistant_text")

    missing_scenes = sorted(set(cards) - set(scenes_by_card))
    if missing_scenes:
        errors.append(f"缺 scenes 文件的卡: {', '.join(missing_scenes)}")
    if len(card_files) != 50:
        warns.append(f"characters 数量 {len(card_files)} != 50")
    if len(scene_files) != 50:
        warns.append(f"scenes 文件数 {len(scene_files)} != 50")

    report = {
        "characters": len(card_files),
        "scene_files": len(scene_files),
        "scenes": total_scenes,
        "turns": total_turns,
        "turns_per_scene_hist": {str(k): turn_hist[k] for k in sorted(turn_hist)},
        "avg_turns_per_scene": round(total_turns / total_scenes, 3) if total_scenes else 0,
        "errors": errors[:50],
        "warnings": warns[:20],
        "error_count": len(errors),
        "warning_count": len(warns),
    }
    out = ROOT / "local" / "_data_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in
                      ("characters", "scene_files", "scenes", "turns", "avg_turns_per_scene", "turns_per_scene_hist")},
                     ensure_ascii=False, indent=1))
    for e in errors[:20]:
        print("[ERR]", e)
    for w in warns[:10]:
        print("[WARN]", w)
    print(f"errors={len(errors)} warnings={len(warns)} report={out}")
    return 0 if not errors else (1 if a.strict else 0)


if __name__ == "__main__":
    sys.exit(main())
