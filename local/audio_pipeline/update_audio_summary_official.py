#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v0.2 官方解码配置（Plan A）结果汇总 → baselines/audio_pipeline_summary_official.json。

覆盖:
  * 文字 track v0.2: baselines/qwen3.5-{2b,4b}-official（1500 场景/2616 turn）
  * 音频级联 v0.2: baselines/audio_pipeline_qwen3.5-{2b,4b}-official（500 场景/784 turn）
  * TTS 云端渲染（qwen-audio-3.1-tts-next）：turns/文件数/失败/耗时/速率（解析 logs/audio_track/cloud_tts_official.log）
  * ASR（复用 v0.1 转写，不重跑）

用法: python local/audio_pipeline/update_audio_summary_official.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "baselines"
DATA = ROOT / "data"
LOG_DIR = ROOT / "logs" / "audio_track"
OUT = BASELINES / "audio_pipeline_summary_official.json"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEXT_PIPELINES = [
    {"model": "qwen3.5-2b", "slug": "qwen3.5-2b-official"},
    {"model": "qwen3.5-4b", "slug": "qwen3.5-4b-official"},
]
AUDIO_PIPELINES = [
    {"pipeline": "B1", "model": "qwen3.5-2b", "slug": "audio_pipeline_qwen3.5-2b-official",
     "base_url": "http://127.0.0.1:8088"},
    {"pipeline": "B2", "model": "qwen3.5-4b", "slug": "audio_pipeline_qwen3.5-4b-official",
     "base_url": "http://127.0.0.1:8089"},
]
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 2.0, "seed": 42, "max_tokens": 512}


def load_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def pct(values, p):
    if not values:
        return 0.0
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round(p / 100.0 * (len(v) - 1))))], 3)


def scan_text_track(slug: str) -> dict:
    base = BASELINES / slug
    scenes = turns = turns_ok = 0
    errors = 0
    lat = []
    reply_chars = []
    long_replies = 0
    files = sorted(base.glob("*.jsonl"))
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            scenes += 1
            if rec.get("error"):
                errors += 1
            for t in rec.get("turns") or []:
                turns += 1
                txt = (t.get("assistant_text") or "").strip()
                if txt:
                    turns_ok += 1
                    reply_chars.append(len(txt))
                    if len(txt) > 300:
                        long_replies += 1
                if t.get("latency_s") is not None:
                    lat.append(float(t["latency_s"]))
    manifest = load_json(base / "_manifest.json", {})
    return {
        "slug": slug,
        "files": len(files),
        "scenes": scenes,
        "scenes_failed": errors,
        "turns": turns,
        "turns_ok": turns_ok,
        "turn_latency_s": {"mean": round(sum(lat) / len(lat), 3) if lat else 0.0,
                           "p50": pct(lat, 50), "p95": pct(lat, 95)},
        "elapsed_s": manifest.get("elapsed_s"),
        "sampling": manifest.get("sampling"),
        "generation": manifest.get("generation"),
        "reply_chars": {"mean": round(sum(reply_chars) / len(reply_chars), 1) if reply_chars else 0.0,
                        "max": max(reply_chars) if reply_chars else 0,
                        "gt_300": long_replies},
        "errors": manifest.get("cards_errors") or [],
    }


def scan_audio_pipeline(spec: dict) -> dict:
    slug = spec["slug"]
    base = BASELINES / slug
    scenes = turns = turns_ok = 0
    errors = []
    turn_lat = []
    scene_lat = []
    tts = {"qwen3-tts-0.6b": 0, "qwen-audio-3.1-tts-next": 0, "none": 0, "other": 0}
    files_ok = files_missing = 0
    reply_chars = []
    long_replies = 0
    for path in sorted(base.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            scenes += 1
            if rec.get("latency_s") is not None:
                scene_lat.append(float(rec["latency_s"]))
            turns_rec = rec.get("turns") or []
            ok = [t for t in turns_rec if (t.get("assistant_text") or "").strip()]
            if rec.get("error") or len(ok) < len(turns_rec):
                errors.append({"scene_id": rec.get("scene_id"), "error": rec.get("error") or "incomplete"})
            for t in turns_rec:
                turns += 1
                txt = (t.get("assistant_text") or "").strip()
                if txt:
                    turns_ok += 1
                    reply_chars.append(len(txt))
                    if len(txt) > 300:
                        long_replies += 1
                if t.get("latency_s") is not None:
                    turn_lat.append(float(t["latency_s"]))
                render = t.get("tts_render")
                aa = t.get("assistant_audio") or ""
                if not aa:
                    tts["none"] += 1
                elif render == "qwen3-tts-0.6b":
                    tts["qwen3-tts-0.6b"] += 1
                elif render == "qwen-audio-3.1-tts-next":
                    tts["qwen-audio-3.1-tts-next"] += 1
                else:
                    tts["other"] += 1
                if aa:
                    candidates = [ROOT / aa, base / aa, base.parent / aa]
                    if any(c.is_file() and c.stat().st_size > 1024 for c in candidates):
                        files_ok += 1
                    else:
                        files_missing += 1
    manifest = load_json(base / "_manifest.json", {})
    return {
        "pipeline": spec["pipeline"],
        "model": spec["model"],
        "slug": slug,
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
        "reply_chars": {"mean": round(sum(reply_chars) / len(reply_chars), 1) if reply_chars else 0.0,
                        "gt_300": long_replies},
        "errors": errors[:10],
        "error_count": len(errors),
    }


def cloud_tts_stats() -> dict:
    runs = []
    logs = [LOG_DIR / "cloud_tts_official.log", LOG_DIR / "cloud_tts_official_retry.log"]
    pat = re.compile(r"云端渲染完成: (\d+)/(\d+) 条, 失败 (\d+), 用时 ([\d.]+)s \(([\d.]+) 条/分钟\)")
    for log_path in logs:
        if not log_path.is_file():
            continue
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pat.search(line)
            if m:
                runs.append({"log": log_path.name,
                             "rendered": int(m.group(1)), "total": int(m.group(2)),
                             "failures": int(m.group(3)), "elapsed_s": float(m.group(4)),
                             "per_min": float(m.group(5))})
    rendered = sum(r["rendered"] for r in runs)
    failures = sum(r["failures"] for r in runs)
    elapsed = round(sum(r["elapsed_s"] for r in runs), 1)
    return {
        "render": "qwen-audio-3.1-tts-next",
        "runs": runs,
        "rendered_total": rendered,
        "failures": failures,
        "elapsed_s": elapsed,
        "throughput_per_min": round(rendered / elapsed * 60, 1) if elapsed else None,
        "workers": 24,
        "rpm_limit": 180,
        "note": "1 条回复（B2 rc29_s03#2）被云端 TTS 内容审核拒绝（HTTP 400 DataInspectionFailed），"
                "换 seed=43 重试成功；最终 1568/1568 条均有云端 mp3",
    }


def main() -> int:
    asr = load_json(DATA / "audio_track" / "asr_summary.json", {})
    result = {
        "version": "v0.2-official-decoding",
        "sampling": SAMPLING,
        "sampling_note": "Qwen3.5 README 非思考文本推荐：temperature=1.0, top_p=1.0, top_k=20, "
                         "min_p=0.0, presence_penalty=2.0, seed=42（presence_penalty 不是 repetition_penalty）；"
                         "llama-server 端 flag 与请求字段同值，/props 已验证默认采样生效",
        "asr": {"model": asr.get("model"), "transcribed": asr.get("transcribed"),
                "wer_mean": asr.get("wer_mean"), "wer_median": asr.get("wer_median"),
                "wer_p90": asr.get("wer_p90"), "exact_match_rate": asr.get("exact_match_rate"),
                "elapsed_s": (asr.get("info") or {}).get("elapsed_s"), "reused_from": "v0.1"},
        "text_track": [scan_text_track(p["slug"]) for p in TEXT_PIPELINES],
        "audio_pipelines": [scan_audio_pipeline(p) for p in AUDIO_PIPELINES],
        "tts_cloud": cloud_tts_stats(),
        "totals": {},
    }
    tt = result["text_track"]
    ap = result["audio_pipelines"]
    result["totals"] = {
        "text_scenes": sum(x["scenes"] for x in tt),
        "text_turns": sum(x["turns"] for x in tt),
        "text_elapsed_s": round(sum(x["elapsed_s"] or 0 for x in tt), 1),
        "audio_scenes": sum(x["scenes"] for x in ap),
        "audio_turns": sum(x["turns"] for x in ap),
        "audio_turns_with_audio": sum(x["tts"]["qwen3-tts-0.6b"] + x["tts"]["qwen-audio-3.1-tts-next"]
                                      + x["tts"]["other"] for x in ap),
        "audio_missing_audio": sum(x["tts"]["none"] for x in ap),
    }
    result["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("version", "totals", "tts_cloud")}, ensure_ascii=False, indent=2))
    print("text:", json.dumps([{k: x[k] for k in ("slug", "scenes", "turns", "elapsed_s", "reply_chars")}
                               for x in tt], ensure_ascii=False))
    print("audio:", json.dumps([{k: x[k] for k in ("slug", "scenes", "turns", "tts", "cascade_elapsed_s")}
                                for x in ap], ensure_ascii=False))
    print(f"summary -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
