#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端 TTS 音频完整性抽检：时长 vs 文本期望、文件大小、采样率（ffprobe）。

用法: python local/audio_pipeline/check_cloud_audio.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "baselines"
SLUGS = ["qwen3.5-2b", "qwen3.5-4b"]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def estimate_seconds(text: str) -> float:
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[A-Za-z]+", text))
    return max(1.5, cn / 4.2 + en * 0.36)


def ffprobe_duration(path: Path) -> float | None:
    ff = shutil.which("ffprobe")
    if not ff:
        return None
    r = subprocess.run([ff, "-v", "error", "-print_format", "json", "-show_format", str(path)],
                       capture_output=True, timeout=60)
    if r.returncode != 0:
        return None
    try:
        return float(json.loads(r.stdout).get("format", {}).get("duration") or 0)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    rows = []
    for slug in SLUGS:
        base = BASELINES / f"audio_pipeline_{slug}"
        for path in sorted((base / "audio_cloud").glob("*.mp3")):
            sid, idx = path.stem.rsplit("_", 1)
            rows.append({"slug": slug, "scene_id": sid, "idx": int(idx), "file": path,
                         "bytes": path.stat().st_size})
    # 映射 turn 文本
    texts = {}
    for slug in SLUGS:
        for p in (BASELINES / f"audio_pipeline_{slug}").glob("*.jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                for t in rec.get("turns") or []:
                    if t.get("tts_render") == "qwen-audio-3.1-tts-next":
                        texts[(slug, rec["scene_id"], int(t["idx"]))] = t.get("assistant_text") or ""
    anomalies = []
    durations = []
    sizes = []
    for r in rows:
        sizes.append(r["bytes"])
        text = texts.get((r["slug"], r["scene_id"], r["idx"]))
        if text is None:
            anomalies.append({**{k: str(v) for k, v in r.items() if k != "file"}, "reason": "no text mapping"})
            continue
        exp = estimate_seconds(text)
        dur = ffprobe_duration(r["file"])
        if dur is None:
            anomalies.append({"slug": r["slug"], "scene_id": r["scene_id"], "idx": r["idx"],
                              "reason": "ffprobe failed"})
            continue
        durations.append(dur)
        if dur > 3.2 * exp or dur < max(1.0, 0.45 * exp):
            anomalies.append({"slug": r["slug"], "scene_id": r["scene_id"], "idx": r["idx"],
                              "duration_s": round(dur, 2), "expected_s": round(exp, 1),
                              "chars": len(text), "bytes": r["bytes"]})
    report = {
        "mp3_count": len(rows),
        "sizes_bytes": {"min": min(sizes), "max": max(sizes),
                        "mean": round(sum(sizes) / len(sizes), 0)} if sizes else None,
        "duration_s": {"mean": round(sum(durations) / len(durations), 2),
                       "min": round(min(durations), 2), "max": round(max(durations), 2),
                       "total": round(sum(durations), 1)} if durations else None,
        "anomalies": anomalies[:20],
        "anomaly_count": len(anomalies),
    }
    out = ROOT / "logs" / "audio_track" / "cloud_audio_check.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("report ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
