#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理异常本地 TTS 样本（GPU 竞争下产生的静音/超长文件），清空其 audio 字段以便云端重渲。

判定异常（满足任一）：
  * wav 缺失 / 峰值 < 0.02 / RMS < 0.004
  * 时长 > 3.0 × 期望时长（期望时长按中文 4.2 字/秒、英文 2.8 字/秒估算，括号动作也计入）
  * 时长 > 60s
异常样本：删除 wav，清空 assistant_audio / tts_render，写入 logs/audio_track/local_tts_cleaned.json

用法: python local/audio_pipeline/clean_bad_local_tts.py [--dry-run]
"""
from __future__ import annotations

import argparse
import array
import json
import math
import re
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "baselines"
LOG_DIR = ROOT / "logs" / "audio_track"
SLUGS = ["qwen3.5-2b", "qwen3.5-4b"]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def estimate_seconds(text: str) -> float:
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[A-Za-z]+", text))
    return max(1.5, cn / 4.2 + en * 0.36)


def wav_stats(path: Path):
    with wave.open(str(path), "rb") as w:
        n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    a = array.array("h")
    a.frombytes(raw[: len(raw) - len(raw) % 2])
    if not a:
        return n / sr, 0.0, 0.0
    peak = max(abs(x) for x in a) / 32768.0
    rms = math.sqrt(sum(x * x for x in a) / len(a)) / 32768.0
    return n / sr, peak, rms


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    order, records = load_records()
    cleaned, kept = [], []
    for slug in SLUGS:
        base = BASELINES / f"audio_pipeline_{slug}"
        for key, recs in records.items():
            if key[0] != slug:
                continue
            changed = False
            for rec in recs.values():
                for turn in rec.get("turns") or []:
                    aa = (turn.get("assistant_audio") or "").replace("\\", "/")
                    if not (aa.startswith("audio/") and aa.endswith(".wav")):
                        continue
                    wav = base / aa
                    text = turn.get("assistant_text") or ""
                    exp = estimate_seconds(text)
                    if not wav.is_file():
                        dur, peak, rms = 0.0, 0.0, 0.0
                        bad = True
                        reason = "missing"
                    else:
                        dur, peak, rms = wav_stats(wav)
                        reason = None
                        if peak < 0.02 or rms < 0.004:
                            reason = f"silent(peak={peak:.3f},rms={rms:.4f})"
                        elif dur > 3.0 * exp:
                            reason = f"too_long(dur={dur:.1f}s,exp={exp:.1f}s)"
                        elif dur > 60:
                            reason = f"too_long_abs(dur={dur:.1f}s)"
                        bad = reason is not None
                    item = {"slug": slug, "scene_id": rec.get("scene_id"), "idx": turn.get("idx"),
                            "wav": str(wav.relative_to(ROOT)).replace("\\", "/"),
                            "duration_s": round(dur, 2), "expected_s": round(exp, 1),
                            "peak": round(peak, 3), "rms": round(rms, 4)}
                    if bad:
                        item["reason"] = reason
                        cleaned.append(item)
                        turn.pop("assistant_audio", None)
                        turn.pop("tts_render", None)
                        changed = True
                        if not a.dry_run and wav.is_file():
                            wav.unlink()
                    else:
                        kept.append(item)
            if changed and not a.dry_run:
                write_records(key[0], key[1], order[key], recs)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report = {"cleaned": cleaned, "kept": kept,
              "kept_count": len(kept), "cleaned_count": len(cleaned),
              "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not a.dry_run:
        (LOG_DIR / "local_tts_cleaned.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
    print(json.dumps({"kept_count": len(kept), "cleaned_count": len(cleaned), "cleaned": cleaned},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
