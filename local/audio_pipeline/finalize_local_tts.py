#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收尾本地 TTS 样本：给已有本地 wav 的 turn 打 tts_render 标记，并重建本地样本 manifest。

背景：本地 Qwen3-TTS 因 GPU 被 EdgeInstant 占用而变慢（171s/item），按 lead 要求提前结束进程，
只保留已生成的样本。此脚本不加载模型，只做数据层收尾。

用法: python local/audio_pipeline/finalize_local_tts.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "baselines"
LOG_DIR = ROOT / "logs" / "audio_track"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RENDER_LOCAL = "qwen3-tts-0.6b"
SLUGS = ["qwen3.5-2b", "qwen3.5-4b"]


def load_records():
    order, records = {}, {}
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


def write_records(slug: str, card_id: str, scene_order, records) -> None:
    path = BASELINES / f"audio_pipeline_{slug}" / f"{card_id}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for sid in scene_order:
            if sid in records:
                f.write(json.dumps(records[sid], ensure_ascii=False) + "\n")
    tmp.replace(path)


def parse_log_speeds() -> list[dict]:
    """从 local_tts_samples.log 解析每批耗时（进程被杀前）。"""
    p = LOG_DIR / "local_tts_samples.log"
    rows = []
    if not p.is_file():
        return rows
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"\[(\d+)/(\d+)\] batch=(\d+) ([\d.]+)s \(([\d.]+)s/item\)", line)
        if m:
            rows.append({"done": int(m.group(1)), "batch": int(m.group(3)),
                         "batch_s": float(m.group(4)), "per_item_s": float(m.group(5))})
    return rows


def main() -> int:
    order, records = load_records()
    tagged = 0
    audio = []  # 每个本地音频的清单
    for slug in SLUGS:
        base = BASELINES / f"audio_pipeline_{slug}"
        for key, recs in records.items():
            if key[0] != slug:
                continue
            changed = False
            for rec in recs.values():
                for turn in rec.get("turns") or []:
                    aa = (turn.get("assistant_audio") or "").replace("\\", "/")
                    if aa.startswith("audio/") and aa.endswith(".wav"):
                        if turn.get("tts_render") != RENDER_LOCAL:
                            turn["tts_render"] = RENDER_LOCAL
                            tagged += 1
                            changed = True
                        wav = base / aa
                        if wav.is_file():
                            audio.append({"slug": slug, "scene_id": rec.get("scene_id"),
                                          "idx": turn.get("idx"), "wav": str(wav.relative_to(ROOT)).replace("\\", "/"),
                                          "bytes": wav.stat().st_size})
                    elif aa.startswith("audio_cloud/") and turn.get("tts_render") != "qwen-audio-3.1-tts-next":
                        turn["tts_render"] = "qwen-audio-3.1-tts-next"
                        changed = True
            if changed:
                write_records(key[0], key[1], order[key], recs)

    batch_speeds = parse_log_speeds()
    per_item = [r["per_item_s"] for r in batch_speeds]
    speeds = {
        "batch_samples": batch_speeds,
        "per_item_s_mean_uncontended": round(sum(per_item[:4]) / 4, 2) if len(per_item) >= 4 else None,
        "per_item_s_all": round(sum(per_item) / len(per_item), 2) if per_item else None,
        "note": "后两批 >70s/item 是因为 EdgeInstant 开始占用 GPU（llama-server 停止后），按 lead 要求提前结束进程",
    }
    summary = {
        "model": r"E:\RBM\models\Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "speaker": "Vivian",
        "language": "Chinese",
        "do_sample": False,
        "sample_rate": 24000,
        "count_ok": len(audio),
        "files": audio,
        "speed": speeds,
        "tagged_turns": tagged,
        "status": "partial - stopped early to release GPU for EdgeInstant",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "tts_local_samples.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count_ok": len(audio), "tagged_turns": tagged, "speeds": speeds,
                      "by_slug": {s: sum(1 for a in audio if a["slug"] == s) for s in SLUGS}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
