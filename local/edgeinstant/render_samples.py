#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EdgeInstant 原生语音抽样渲染（语音判官冒烟用）。

从 baselines/audio_edgeinstant/<card_id>.jsonl 里抽 N 条 turn（覆盖多张卡，优先覆盖：
长回复 / 安全拒绝 / 元对话），用 DuplexSession + 整句 render 渲染成 24kHz wav，
存 baselines/audio_edgeinstant/audio/<scene_id>_<idx>.wav，并回填 turn 的
assistant_audio / audio_source="edgeinstant-native" / assistant_audio_seconds / audio_render_s。

幂等：已有 assistant_audio 且 wav 存在的 turn 跳过（--force 可重渲）。

用法：
  $env:EDGEINSTANT_VENDOR='1'
  python local/edgeinstant/render_samples.py --n-turns 30 --n-cards 10
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ei_common as C  # noqa: E402
import run_edgeinstant as R  # noqa: E402  (复用 make_session / render_full / save_wav)

C.set_utf8_stdio()

DEFAULT_OUT_DIR = PROJECT_ROOT / "baselines" / "audio_edgeinstant"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "edgeinstant"

log = logging.getLogger("render_samples")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EdgeInstant 原生语音抽样渲染")
    ap.add_argument("--model-path", default=str(C.MODEL_PATH))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--scenes-dir", default=str(C.SCENES_DIR))
    ap.add_argument("--n-turns", type=int, default=30)
    ap.add_argument("--n-cards", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--force", action="store_true", help="已有音频也重渲")
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args(argv)
    if args.log_file is None:
        args.log_file = str(DEFAULT_LOG_DIR / f"render_samples_{datetime.now():%Y%m%d_%H%M%S}.log")
    return args


def setup_logging(args) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    p = Path(args.log_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(p, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        handlers=handlers, force=True)


def load_scene_categories(scenes_dir: Path) -> dict[tuple[str, str], str]:
    cats: dict[tuple[str, str], str] = {}
    for p in sorted(scenes_dir.glob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        cid = doc.get("card_id") or p.stem
        for s in doc.get("scenes") or []:
            cats[(cid, s.get("scene_id"))] = str(s.get("category") or s.get("subtype") or "")
    return cats


def pick_turns(records_by_card: dict[str, list[dict]], cats: dict[tuple[str, str], str],
               n_turns: int, n_cards: int, seed: int) -> list[dict]:
    """策略：先挑 (cid, turn) 候选池，再按 每卡 3 条 = 最长回复 + 安全拒绝/元对话 + 随机。"""
    rng = random.Random(seed)
    cards = sorted(records_by_card)
    if len(cards) > n_cards:
        step = len(cards) / n_cards
        cards = [cards[int(i * step)] for i in range(n_cards)]
    per_card = max(1, n_turns // max(1, len(cards)))
    picked: list[dict] = []
    used: set[tuple[str, str, int]] = set()

    def add(cand: dict) -> bool:
        key = (cand["card_id"], cand["scene_id"], cand["turn"]["idx"])
        if key in used:
            return False
        used.add(key)
        picked.append(cand)
        return True

    # 1) 特殊场景优先（安全拒绝 / 元对话），全局每卡最多一条
    special_cats = ("安全拒绝", "元对话")
    for cid in cards:
        for rec in records_by_card[cid]:
            cat = cats.get((cid, rec["scene_id"]), "")
            if not any(k in cat for k in special_cats):
                continue
            for turn in rec.get("turns") or []:
                if turn.get("assistant_text"):
                    if add({"card_id": cid, "scene_id": rec["scene_id"], "category": cat, "turn": turn}):
                        pass
                    break
    # 2) 每卡补：最长回复一条 + 随机一条
    for cid in cards:
        cands = [{"card_id": cid, "scene_id": rec["scene_id"],
                  "category": cats.get((cid, rec["scene_id"]), ""), "turn": t}
                 for rec in records_by_card[cid] for t in (rec.get("turns") or [])
                 if t.get("assistant_text")]
        if not cands:
            continue
        cands.sort(key=lambda c: -len(c["turn"].get("assistant_text") or ""))
        add(cands[0])
        rest = [c for c in cands if (c["card_id"], c["scene_id"], c["turn"]["idx"]) not in used]
        if rest:
            add(rng.choice(rest))
        for c in cands[1:]:
            if len([x for x in picked if x["card_id"] == cid]) >= per_card:
                break
            add(c)
    rng.shuffle(picked)
    return picked[:n_turns]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args)
    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    cats = load_scene_categories(Path(args.scenes_dir))

    records_by_card: dict[str, list[dict]] = {}
    for p in sorted(out_dir.glob("*.jsonl")):
        recs = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("turns"):
                recs.append(rec)
        if recs:
            records_by_card[p.stem] = recs
    n_all = sum(len(r["turns"]) for rs in records_by_card.values() for r in rs)
    log.info("已有记录: %d 卡 / %d 场景 / %d 轮", len(records_by_card),
             sum(len(rs) for rs in records_by_card.values()), n_all)
    if n_all == 0:
        log.error("没有可渲染的 assistant_text，先跑 run_edgeinstant.py --phase text")
        return 2

    picked = pick_turns(records_by_card, cats, args.n_turns, args.n_cards, args.seed)
    log.info("抽取 %d 条 turn（%d 张卡）", len(picked), len({p["card_id"] for p in picked}))
    for p in picked:
        log.info("  %s/%s#%d cat=%s len=%d", p["card_id"], p["scene_id"], p["turn"]["idx"],
                 p["category"], len(p["turn"].get("assistant_text") or ""))

    C.activate_vendor()
    import torch
    import transformers
    log.info("transformers=%s torch=%s vendor_mode=%s", transformers.__version__, torch.__version__,
             C._vendor_active)

    proc = C.build_processor(args.model_path, log=log.info)
    model = C.load_model(args.model_path, log=log.info)
    session, mod = R.make_session(model, proc, args.model_path, log=log)

    rendered = skipped = failed = 0
    render_s_total = audio_s_total = 0.0
    t_all = time.perf_counter()
    try:
        by_card: dict[str, list[dict]] = {}
        for p in picked:
            by_card.setdefault(p["card_id"], []).append(p)
        for cid, items in by_card.items():
            path = out_dir / f"{cid}.jsonl"
            recs = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
            index = {r["scene_id"]: r for r in recs}
            for p in items:
                sid, idx = p["scene_id"], int(p["turn"]["idx"])
                text = (p["turn"].get("assistant_text") or "").strip()
                wav_path = audio_dir / f"{sid}_{idx}.wav"
                if not text:
                    skipped += 1
                    continue
                if (not args.force) and p["turn"].get("assistant_audio") and wav_path.is_file():
                    skipped += 1
                    continue
                try:
                    t0 = time.perf_counter()
                    pcm, rate = R.render_full(session, mod, text, seed=idx)
                    rs = time.perf_counter() - t0
                    if len(pcm) == 0:
                        raise RuntimeError("empty pcm")
                    R.save_wav(wav_path, pcm, rate)
                    audio_s = len(pcm) / rate
                    rec = index[sid]
                    for t in rec["turns"]:
                        if int(t.get("idx", 0)) == idx:
                            t["assistant_audio"] = str(wav_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                            t["assistant_audio_seconds"] = round(audio_s, 3)
                            t["audio_render_s"] = round(rs, 3)
                            t["audio_sample_rate"] = rate
                            t["audio_source"] = "edgeinstant-native"
                            break
                    rendered += 1
                    render_s_total += rs
                    audio_s_total += audio_s
                    log.info("  OK %s/%s#%d 音频 %.2fs / 渲染 %.2fs (rtf=%.2f)",
                             cid, sid, idx, audio_s, rs, rs / audio_s if audio_s else 0)
                except Exception as exc:  # noqa: BLE001
                    log.exception("  FAIL %s/%s#%d", cid, sid, idx)
                    failed += 1
            # 原子回写该卡
            tmp = path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8", newline="\n") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(path)
    finally:
        try:
            session.close()
        except Exception:  # noqa: BLE001
            pass
        del model
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t_all
    summary = {
        "rendered_turns": rendered, "skipped": skipped, "failed": failed,
        "cards": len({p["card_id"] for p in picked if (audio_dir / f'{p["scene_id"]}_{int(p["turn"]["idx"])}.wav').is_file()}),
        "audio_seconds_total": round(audio_s_total, 2),
        "render_seconds_total": round(render_s_total, 2),
        "elapsed_s": round(elapsed, 2),
        "avg_rtf": round(render_s_total / audio_s_total, 3) if audio_s_total else None,
        "audio_dir": str(audio_dir),
        "log_file": args.log_file,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (DEFAULT_LOG_DIR / "render_samples.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("完成: 渲染 %d 条 | 跳过 %d | 失败 %d | 音频 %.1fs / 渲染 %.1fs (rtf=%s) | 总用时 %.1fs",
             rendered, skipped, failed, audio_s_total, render_s_total,
             summary["avg_rtf"], elapsed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
