#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地 Qwen3-TTS-0.6B 链路验证样本：B1/B2 各 10 个 turn（覆盖安全拒绝/重复问答/元对话）。

背景：本地 TTS 在 3090 上 RTF≈2-3x（~18s/条），全量 1568 条不可接受 → 只做链路验证；
全量音频由 local/audio_pipeline/cloud_tts_render.py 用云端 qwen-audio-3.1-tts-next 渲染。

选择规则（B1/B2 用同一组 10 个 (card, scene, idx)，共 10 张不同卡）：
  * 安全拒绝 2 条、重复问答 2 条、元对话 2 条（各取不同卡）
  * 其余 4 条从不同卡的其它类别补（角色/背景/故事/关系/交叉/情绪/情境/错误角色/错误背景/记忆一致性）

输出:
  * baselines/audio_pipeline_<slug>/audio/<scene_id>_<idx>.wav（24kHz PCM16）
  * 更新 jsonl：assistant_audio = "audio/<scene_id>_<idx>.wav", tts_render = "qwen3-tts-0.6b"
  * logs/audio_track/tts_local_samples.json（选择清单 + 逐条耗时 + 速度结论）

用法: python local/audio_pipeline/local_tts_samples.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = ROOT / "local"
sys.path.insert(0, str(LOCAL_DIR))
sys.path.insert(0, str(ROOT / "vendor" / "tts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from audio_tts_local import SPEAKER, TTS_DIR, save_wav  # noqa: E402

BASELINES = ROOT / "baselines"
DATA = ROOT / "data"
LOG_DIR = ROOT / "logs" / "audio_track"
SLUGS = ["qwen3.5-2b", "qwen3.5-4b"]
RENDER_LOCAL = "qwen3-tts-0.6b"

# 类别覆盖要求（每条 = (类别, 条数)），再从其它类别补 4 条
COVER = [("安全拒绝", 2), ("重复问答", 2), ("元对话", 2)]
FILL_ORDER = ["角色追问", "背景追问", "故事追问", "关系追问", "交叉追问",
              "情绪共情", "情境演绎", "错误角色", "错误背景", "记忆一致性"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_selection() -> list[dict]:
    selection = load_json(DATA / "voice" / "selection.json")["items"]
    scenes_by_card = {}
    for p in sorted((DATA / "scenes").glob("*.json")):
        doc = load_json(p)
        scenes_by_card[p.stem] = {s["scene_id"]: s for s in doc.get("scenes") or []}

    # 每卡候选：voice 场景 -> (scene_id, category, first_turn_idx)
    per_card: list[dict] = []
    for item in selection:
        cid = item["card_id"]
        cand = []
        for sid in item.get("scene_ids") or []:
            scene = scenes_by_card.get(cid, {}).get(sid)
            if not scene:
                continue
            turns = sorted(scene.get("turns") or [], key=lambda t: int(t.get("idx", 0)))
            if not turns:
                continue
            cand.append({"card_id": cid, "scene_id": sid, "category": scene.get("category", ""),
                         "idx": int(turns[0].get("idx", 1))})
        if cand:
            per_card.append({"card_id": cid, "candidates": cand})

    picked: list[dict] = []
    used_cards: set[str] = set()

    def take(category: str, count: int) -> None:
        for nc in per_card:
            if count <= 0:
                return
            if nc["card_id"] in used_cards:
                continue
            for c in nc["candidates"]:
                if c["category"] == category:
                    picked.append(c)
                    used_cards.add(nc["card_id"])
                    count -= 1
                    break

    for cat, n in COVER:
        take(cat, n)

    # 补 4 条：按类别优先级，尽量覆盖不同卡
    for cat in FILL_ORDER:
        if len(picked) >= 10:
            break
        take(cat, 1)
    # 仍不足则任意补
    for nc in per_card:
        if len(picked) >= 10:
            break
        if nc["card_id"] not in used_cards:
            picked.append(nc["candidates"][0])
            used_cards.add(nc["card_id"])

    picked = picked[:10]
    assert len(picked) == 10, f"selection failed: {len(picked)}"
    assert len({p["card_id"] for p in picked}) == 10, "cards not distinct"
    return picked


def load_records() -> tuple[dict, dict]:
    order: dict[tuple[str, str], list[str]] = {}
    records: dict[tuple[str, str], dict[str, dict]] = {}
    for slug in SLUGS:
        for path in sorted((BASELINES / f"audio_pipeline_{slug}").glob("*.jsonl")):
            key = (slug, path.stem)
            order[key] = []
            records[key] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                sid = rec.get("scene_id")
                if not sid:
                    continue
                order[key].append(sid)
                records[key][sid] = rec
    return order, records


def write_records(slug: str, card_id: str, scene_order: list[str], records: dict[str, dict]) -> None:
    path = BASELINES / f"audio_pipeline_{slug}" / f"{card_id}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for sid in scene_order:
            if sid in records:
                f.write(json.dumps(records[sid], ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--log-file", default=str(LOG_DIR / "local_tts_samples.log"))
    a = ap.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(a.log_file, encoding="utf-8")],
        force=True,
    )
    log = logging.getLogger("local_tts_samples")

    plan = build_selection()
    for slug in SLUGS:
        (BASELINES / f"audio_pipeline_{slug}" / "audio").mkdir(parents=True, exist_ok=True)
    order, records = load_records()

    # 组装待渲染条目（B1/B2 各 10）
    jobs = []
    for p in plan:
        for slug in SLUGS:
            rec = records[(slug, p["card_id"])].get(p["scene_id"])
            if not rec:
                log.error("缺 baseline 记录: %s/%s", slug, p["scene_id"])
                continue
            turn = next((t for t in rec.get("turns") or [] if int(t.get("idx", 0)) == p["idx"]), None)
            if not turn:
                log.error("缺 turn: %s/%s#%d", slug, p["scene_id"], p["idx"])
                continue
            wav = BASELINES / f"audio_pipeline_{slug}" / "audio" / f"{p['scene_id']}_{p['idx']}.wav"
            jobs.append({"slug": slug, "card_id": p["card_id"], "scene_id": p["scene_id"],
                         "idx": p["idx"], "category": p["category"], "text": turn["assistant_text"].strip(),
                         "wav": wav})

    log.info("本地 TTS 验证样本: %d 条（B1/B2 各 %d；卡 %d 张）",
             len(jobs), len(jobs) // 2, len({j["card_id"] for j in jobs}))
    for p in plan:
        log.info("  plan: %s / %s / %s / idx=%d", p["card_id"], p["scene_id"], p["category"], p["idx"])

    import torch
    from qwen_tts import Qwen3TTSModel

    t0 = time.perf_counter()
    log.info("加载本地 TTS 模型: %s", TTS_DIR)
    model = Qwen3TTSModel.from_pretrained(str(TTS_DIR), device_map="cuda:0",
                                          dtype=torch.bfloat16, attn_implementation="sdpa")
    log.info("模型加载完成 %.1fs", time.perf_counter() - t0)

    timings = []
    failures = []
    t_start = time.perf_counter()
    for b0 in range(0, len(jobs), a.batch_size):
        batch = jobs[b0: b0 + a.batch_size]
        texts = [j["text"] for j in batch]
        tb = time.perf_counter()
        try:
            with torch.inference_mode():
                wavs, sr = model.generate_custom_voice(
                    text=texts, speaker=SPEAKER, language="Chinese",
                    do_sample=False, non_streaming_mode=True,
                )
        except Exception as exc:  # noqa: BLE001
            log.error("batch 失败: %s", exc)
            failures.extend(jobs[b0: b0 + len(batch)])
            continue
        dt = time.perf_counter() - tb
        for j, wav in zip(batch, wavs):
            save_wav(j["wav"], wav, sr)
            dur = len(wav) / sr
            item = {"slug": j["slug"], "card_id": j["card_id"], "scene_id": j["scene_id"],
                    "idx": j["idx"], "category": j["category"], "text": j["text"],
                    "wav": str(j["wav"].relative_to(ROOT)).replace("\\", "/"),
                    "audio_duration_s": round(dur, 2),
                    "wall_per_item_s": round(dt / len(batch), 2),
                    "rtf": round((dt / len(batch)) / dur, 2) if dur else None}
            timings.append(item)
            rec = records[(j["slug"], j["card_id"])][j["scene_id"]]
            for turn in rec.get("turns") or []:
                if int(turn.get("idx", 0)) == j["idx"]:
                    turn["assistant_audio"] = f"audio/{j['scene_id']}_{j['idx']}.wav"
                    turn["tts_render"] = RENDER_LOCAL
                    break
        for j in batch:
            write_records(j["slug"], j["card_id"], order[(j["slug"], j["card_id"])], records[(j["slug"], j["card_id"])])
        log.info("[%d/%d] batch=%d %.1fs (%.2fs/item)", len(timings), len(jobs), len(batch), dt, dt / len(batch))

    elapsed = time.perf_counter() - t_start
    # 之前测速跑生成的 10 条本地 wav 也打上 tts_render 标记（如果有 assistant_audio=audio/*.wav）
    tagged = 0
    for slug in SLUGS:
        for key, recs in records.items():
            if key[0] != slug:
                continue
            changed = False
            for rec in recs.values():
                for turn in rec.get("turns") or []:
                    aa = turn.get("assistant_audio") or ""
                    if aa.startswith("audio/") and not turn.get("tts_render"):
                        turn["tts_render"] = RENDER_LOCAL
                        tagged += 1
                        changed = True
            if changed:
                write_records(key[0], key[1], order[key], recs)

    summary = {
        "model": str(TTS_DIR),
        "speaker": SPEAKER,
        "language": "Chinese",
        "do_sample": False,
        "sample_rate": 24000,
        "picked_plan": plan,
        "items": timings,
        "count_ok": len(timings),
        "count_failed": len(failures),
        "failures": failures,
        "speed": {
            "wall_per_item_s": round(sum(t["wall_per_item_s"] for t in timings) / len(timings), 2) if timings else None,
            "audio_duration_s_mean": round(sum(t["audio_duration_s"] for t in timings) / len(timings), 2) if timings else None,
            "rtf_mean": round(sum(t["rtf"] for t in timings if t["rtf"]) / len([t for t in timings if t["rtf"]]), 2)
            if any(t["rtf"] for t in timings) else None,
        },
        "elapsed_s": round(elapsed, 1),
        "extra_local_tagged": tagged,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (LOG_DIR / "tts_local_samples.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("本地 TTS 样本完成: %d 条, 失败 %d, 用时 %.1fs, mean %.2fs/item, RTF≈%.2f",
             len(timings), len(failures), elapsed,
             summary["speed"]["wall_per_item_s"] or 0, summary["speed"]["rtf_mean"] or 0)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
