# -*- coding: utf-8 -*-
"""Validate RoleCraft-ZH dataset integrity.
Usage: python scripts/validate_dataset.py [--strict]
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TONE_ENUM = ["平静", "开心", "兴奋", "好奇", "疑惑", "惊讶", "愤怒", "不满", "伤心", "担忧",
             "焦急", "害怕", "温柔", "关切", "撒娇", "羞涩", "得意", "嘲讽", "调侃", "冷淡",
             "严肃", "坚定", "无奈", "疲惫", "期待", "恳求"]
CATEGORY_COUNTS = {"角色追问": 2, "背景追问": 3, "故事追问": 4, "关系追问": 4, "交叉追问": 3,
                   "情绪共情": 2, "情境演绎": 1, "错误角色": 2, "错误背景": 2, "重复问答": 2,
                   "安全拒绝": 2, "元对话": 2, "记忆一致性": 1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    errors, warns = [], []
    cards = []
    for f in sorted((DATA / "characters").glob("*.json")):
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
            cards.append(c)
            if not (100 <= len(c.get("persona_card") or "") <= 1000):
                errors.append(f"{f.name}: persona_card length {len(c.get('persona_card') or '')}")
            if len(c.get("sources") or []) < 2:
                errors.append(f"{f.name}: sources < 2")
        except Exception as e:
            errors.append(f"{f.name}: {e}")
    if len(cards) != 50:
        errors.append(f"characters count {len(cards)} != 50")
    card_ids = {c.get("card_id") for c in cards}
    if len(card_ids) != len(cards):
        errors.append("duplicate card_id")

    scenes_total = 0; tones = Counter(); scene_ids = set()
    for f in sorted((DATA / "scenes").glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{f.name}: {e}"); continue
        cid = d.get("card_id") or f.stem
        d["card_id"] = cid
        if cid not in card_ids:
            errors.append(f"{f.name}: card_id not in characters")
        scenes = d.get("scenes") or []
        scenes_total += len(scenes)
        if len(scenes) != 30:
            errors.append(f"{f.name}: scenes {len(scenes)} != 30")
        cnt = Counter(s.get("category") for s in scenes)
        for cat, n in CATEGORY_COUNTS.items():
            if cnt.get(cat, 0) != n:
                errors.append(f"{f.name}: category {cat} {cnt.get(cat,0)} != {n}")
        texts = set()
        for s in scenes:
            sid = s.get("scene_id")
            if not sid or sid in scene_ids:
                errors.append(f"{f.name}: bad/dup scene_id {sid}")
            scene_ids.add(sid)
            turns = s.get("turns") or []
            if not (1 <= len(turns) <= 5):
                errors.append(f"{sid}: turns {len(turns)}")
            if not s.get("expected_behavior") or not s.get("key_points") or not s.get("must_not"):
                errors.append(f"{sid}: missing rubric")
            for t in turns:
                ut = (t.get("user_text") or "").strip()
                if not ut or len(ut) > 40:
                    errors.append(f"{sid}: user_text len {len(ut)}")
                if ut in texts:
                    warns.append(f"{sid}: duplicate user_text {ut[:20]}")
                texts.add(ut)
                if t.get("tone") not in TONE_ENUM:
                    errors.append(f"{sid}: bad tone {t.get('tone')}")
                tones[t.get("tone")] += 1

    refs_bad = 0
    for f in sorted((DATA / "references").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        sf = DATA / "scenes" / f.name
        if not sf.exists():
            errors.append(f"refs {f.name}: no scenes file"); continue
        sc = {s["scene_id"]: s for s in json.loads(sf.read_text(encoding="utf-8"))["scenes"]}
        for s in d.get("scenes", []):
            if s["scene_id"] not in sc:
                errors.append(f"refs {f.name}: unknown scene {s['scene_id']}"); continue
            if len(s.get("turns", [])) != len(sc[s["scene_id"]]["turns"]):
                refs_bad += 1
    if refs_bad:
        errors.append(f"references turn mismatch scenes: {refs_bad}")

    voice_total = 0
    vp = DATA / "voice" / "selection.json"
    voice_scene_ids = set()
    if vp.exists():
        sel = json.loads(vp.read_text(encoding="utf-8"))
        if len(sel.get("items", [])) != 50:
            warns.append(f"voice items {len(sel.get('items', []))} != 50")
        for it in sel.get("items", []):
            if len(it.get("scene_ids", [])) != 10:
                warns.append(f"voice {it['card_id']}: {len(it.get('scene_ids', []))} scenes")
            voice_scene_ids.update(it.get("scene_ids", []))
        voice_total = len(voice_scene_ids)
    else:
        warns.append("voice selection missing")

    manifest_n = 0
    mp = DATA / "voice" / "manifest.jsonl"
    if mp.exists():
        for line in mp.read_text(encoding="utf-8").splitlines():
            if line.strip(): manifest_n += 1
    else:
        warns.append("voice manifest missing")

    report = {"characters": len(cards), "scenes": scenes_total, "turns": sum(tones.values()),
              "voice_scenes": voice_total, "voice_manifest": manifest_n,
              "tone_distribution": dict(tones), "errors": errors, "warnings": warns}
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "validate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("characters", "scenes", "turns", "voice_scenes", "voice_manifest")},
                     ensure_ascii=False, indent=1))
    for e in errors[:30]: print("[ERR]", e)
    for w in warns[:10]: print("[WARN]", w)
    print(f"errors={len(errors)} warnings={len(warns)}")
    if errors and (a.strict or True):
        sys.exit(1)

if __name__ == "__main__":
    main()
