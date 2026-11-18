#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MiniCPM5-2B 链路结果汇总 → baselines/audio_pipeline_summary_minicpm5.json。

覆盖:
  * 文字 track: baselines/minicpm5-2b-greedy / minicpm5-2b-official（1500 场景/2616 turn 各）
  * 音频级联: baselines/audio_pipeline_minicpm5-2b-greedy / -official（500 场景/784 turn 各）
  * 云端 TTS（qwen-audio-3.1-tts-next）：条数/失败/耗时/速率（logs/minicpm5/cloud_tts*.log）
  * ASR（复用 v0.1 转写）
含复读退化率与回复长度，便于与 Qwen3.5-2B 文字/音频 track 同口径对照。

用法: python local/audio_pipeline/update_audio_summary_minicpm5.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "baselines"
DATA = ROOT / "data"
LOG_DIR = ROOT / "logs" / "minicpm5"
OUT = BASELINES / "audio_pipeline_summary_minicpm5.json"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPEAT_RE = re.compile(r"(.{4,20}?)\1{2,}")
TEXT_PIPELINES = [
    {"slug": "minicpm5-2b-greedy", "mode": "greedy"},
    {"slug": "minicpm5-2b-official", "mode": "official"},
]
AUDIO_PIPELINES = [
    {"slug": "audio_pipeline_minicpm5-2b-greedy", "pipeline": "minicpm5-greedy",
     "model": "minicpm5-2b-greedy", "base_url": "http://127.0.0.1:8090"},
    {"slug": "audio_pipeline_minicpm5-2b-official", "pipeline": "minicpm5-official",
     "model": "minicpm5-2b-official", "base_url": "http://127.0.0.1:8091"},
]


def load_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def pct(values, p):
    if not values:
        return 0.0
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round(p / 100.0 * (len(v) - 1))))], 3)


def scan_track(slug: str) -> dict:
    base = BASELINES / slug
    scenes = turns = turns_ok = err = 0
    lat = []
    lens = []
    repeats = 0
    files = sorted(base.glob("*.jsonl"))
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            scenes += 1
            if rec.get("error"):
                err += 1
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
    manifest = load_json(base / "_manifest.json", {})
    return {
        "slug": slug,
        "files": len(files),
        "scenes": scenes,
        "scenes_failed": err,
        "turns": turns,
        "turns_ok": turns_ok,
        "turn_latency_s": {"mean": round(sum(lat) / len(lat), 3) if lat else 0.0,
                           "p50": pct(lat, 50), "p95": pct(lat, 95)},
        "elapsed_s": manifest.get("elapsed_s"),
        "sampling": manifest.get("sampling"),
        "generation": manifest.get("generation"),
        "reply_chars": {"mean": round(sum(lens) / len(lens), 1) if lens else 0.0,
                        "max": max(lens) if lens else 0,
                        "gt_300": sum(1 for x in lens if x > 300)},
        "repetition_degenerate": repeats,
        "repetition_rate_pct": round(repeats / len(lens) * 100, 2) if lens else 0.0,
        "errors": manifest.get("cards_errors") or [],
    }


def scan_audio(spec: dict) -> dict:
    slug = spec["slug"]
    base = BASELINES / slug
    scenes = turns = turns_ok = 0
    errors = []
    turn_lat, scene_lat = [], []
    tts = {"qwen-audio-3.1-tts-next": 0, "qwen3-tts-0.6b": 0, "none": 0, "other": 0}
    files_ok = files_missing = 0
    lens = []
    repeats = 0
    for path in sorted(base.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            scenes += 1
            if rec.get("latency_s") is not None:
                scene_lat.append(float(rec["latency_s"]))
            trs = rec.get("turns") or []
            ok = [t for t in trs if (t.get("assistant_text") or "").strip()]
            if rec.get("error") or len(ok) < len(trs):
                errors.append({"scene_id": rec.get("scene_id"), "error": rec.get("error") or "incomplete"})
            for t in trs:
                turns += 1
                txt = (t.get("assistant_text") or "").strip()
                if txt:
                    turns_ok += 1
                    lens.append(len(txt))
                    if REPEAT_RE.search(txt):
                        repeats += 1
                if t.get("latency_s") is not None:
                    turn_lat.append(float(t["latency_s"]))
                aa = t.get("assistant_audio") or ""
                render = t.get("tts_render")
                if not aa:
                    tts["none"] += 1
                elif render == "qwen-audio-3.1-tts-next":
                    tts["qwen-audio-3.1-tts-next"] += 1
                elif render == "qwen3-tts-0.6b":
                    tts["qwen3-tts-0.6b"] += 1
                else:
                    tts["other"] += 1
                if aa:
                    cands = [ROOT / aa, base / aa, base.parent / aa]
                    if any(c.is_file() and c.stat().st_size > 1024 for c in cands):
                        files_ok += 1
                    else:
                        files_missing += 1
    manifest = load_json(base / "_manifest.json", {})
    return {
        "pipeline": spec["pipeline"],
        "slug": slug,
        "model": spec["model"],
        "base_url": spec["base_url"],
        "scenes": scenes,
        "scenes_failed": len(errors),
        "turns": turns,
        "turns_ok": turns_ok,
        "turn_latency_s": {"mean": round(sum(turn_lat) / len(turn_lat), 3) if turn_lat else 0.0,
                           "p50": pct(turn_lat, 50), "p95": pct(turn_lat, 95)},
        "scene_latency_s": {"mean": round(sum(scene_lat) / len(scene_lat), 3) if scene_lat else 0.0,
                            "p50": pct(scene_lat, 50), "p95": pct(scene_lat, 95)},
        "cascade_elapsed_s": manifest.get("elapsed_s"),
        "sampling": manifest.get("sampling"),
        "generation": manifest.get("generation"),
        "tts": tts,
        "audio_files_ok": files_ok,
        "audio_files_missing": files_missing,
        "reply_chars": {"mean": round(sum(lens) / len(lens), 1) if lens else 0.0,
                        "gt_300": sum(1 for x in lens if x > 300)},
        "repetition_rate_pct": round(repeats / len(lens) * 100, 2) if lens else 0.0,
        "errors": errors[:10],
        "error_count": len(errors),
    }


def cloud_tts_stats() -> dict:
    pat = re.compile(r"云端渲染完成: (\d+)/(\d+) 条, 失败 (\d+), 用时 ([\d.]+)s \(([\d.]+) 条/分钟\)")
    runs = []
    for name in ("cloud_tts.log", "cloud_tts_retry.log"):
        p = LOG_DIR / name
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pat.search(line)
            if m:
                runs.append({"log": name, "rendered": int(m.group(1)), "total": int(m.group(2)),
                             "failures": int(m.group(3)), "elapsed_s": float(m.group(4)),
                             "per_min": float(m.group(5))})
    failures = load_json(LOG_DIR / "cloud_tts_failures.json", [])
    rendered = sum(r["rendered"] for r in runs)
    fail_count = sum(r["failures"] for r in runs)
    elapsed = round(sum(r["elapsed_s"] for r in runs), 1)
    return {
        "render": "qwen-audio-3.1-tts-next",
        "runs": runs,
        "rendered_total": rendered,
        "failure_runs": fail_count,
        "unresolved_failures": failures if isinstance(failures, list) else [],
        "elapsed_s": elapsed,
        "throughput_per_min": round(rendered / elapsed * 60, 1) if elapsed else None,
        "workers": 24,
        "rpm_limit": 180,
    }


def main() -> int:
    asr = load_json(DATA / "audio_track" / "asr_summary.json", {})
    result = {
        "version": "minicpm5-2b-audio-link",
        "model": "MiniCPM5-2B",
        "deployment": {
            "llama_cpp": "b11552",
            "arch": "LlamaForCausalLM（标准 GGUF）",
            "greedy": "http://127.0.0.1:8090（请求 temperature=0）",
            "official": "http://127.0.0.1:8091（server flags: --temp 1.0 --top-p 0.95 --min-p 0.0 --seed 42；"
                        "README 特别要求 --min-p 0.0，llama.cpp 默认 0.05 会加重读）",
            "thinking": "关闭（--reasoning off + chat_template_kwargs enable_thinking=false，输出无 <think> 泄漏）",
        },
        "asr": {"model": asr.get("model"), "transcribed": asr.get("transcribed"),
                "wer_mean": asr.get("wer_mean"), "wer_median": asr.get("wer_median"),
                "wer_p90": asr.get("wer_p90"), "exact_match_rate": asr.get("exact_match_rate"),
                "elapsed_s": (asr.get("info") or {}).get("elapsed_s"), "reused_from": "v0.1"},
        "text_track": [scan_track(p["slug"]) for p in TEXT_PIPELINES],
        "audio_pipelines": [scan_audio(p) for p in AUDIO_PIPELINES],
        "tts_cloud": cloud_tts_stats(),
    }
    tt, ap = result["text_track"], result["audio_pipelines"]
    result["totals"] = {
        "text_scenes": sum(x["scenes"] for x in tt),
        "text_turns": sum(x["turns"] for x in tt),
        "text_elapsed_s": round(sum(x["elapsed_s"] or 0 for x in tt), 1),
        "audio_scenes": sum(x["scenes"] for x in ap),
        "audio_turns": sum(x["turns"] for x in ap),
        "audio_turns_with_audio": sum(x["tts"]["qwen-audio-3.1-tts-next"] + x["tts"]["qwen3-tts-0.6b"]
                                      + x["tts"]["other"] for x in ap),
        "audio_missing_audio": sum(x["tts"]["none"] for x in ap),
        "tts_rendered": result["tts_cloud"]["rendered_total"],
    }
    result["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"version": result["version"], "totals": result["totals"],
                      "tts_cloud": {k: result["tts_cloud"][k] for k in
                                    ("rendered_total", "failure_runs", "elapsed_s", "throughput_per_min")}},
                     ensure_ascii=False, indent=2))
    for x in tt:
        print("text", x["slug"], x["scenes"], x["turns"], x["elapsed_s"],
              x["reply_chars"], "rep%", x["repetition_rate_pct"])
    for x in ap:
        print("audio", x["slug"], x["scenes"], x["turns"], x["cascade_elapsed_s"], x["tts"],
              "rep%", x["repetition_rate_pct"])
    print("summary ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
