#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总音频 track B1/B2 结果 → baselines/audio_pipeline_summary.json。

统计口径（全部以文件为准，避免 runner 续跑计数口径问题）：
  * ASR: data/audio_track/asr_summary.json（WER/延迟）
  * 每个 pipeline: 场景数/turn 数/成功数/延迟分位/错误
  * TTS: 本地 wav（tts_render=qwen3-tts-0.6b）与云端 mp3（qwen-audio-3.1-tts-next）条数、缺失/损坏数
  * 与文字 track 同模型的 turn 延迟对比（可选）

用法: python local/audio_pipeline/update_audio_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "baselines"
DATA = ROOT / "data"
LOG_DIR = ROOT / "logs" / "audio_track"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PIPELINES = [
    {"key": "B1", "model": "qwen3.5-2b", "slug": "audio_pipeline_qwen3.5-2b"},
    {"key": "B2", "model": "qwen3.5-4b", "slug": "audio_pipeline_qwen3.5-4b"},
]


def pct(values, p):
    if not values:
        return 0.0
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round(p / 100.0 * (len(v) - 1))))], 3)


def load_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def scan_pipeline(slug: str) -> dict:
    base = BASELINES / slug
    scenes_total = 0
    turns_total = 0
    turns_ok = 0
    scenes_complete = 0
    scenes_with_error = 0
    turn_lat = []
    scene_lat = []
    tts_count = {"qwen3-tts-0.6b": 0, "qwen-audio-3.1-tts-next": 0, "other": 0, "none": 0}
    audio_files_missing = 0
    audio_files_ok = 0
    errors = []
    for path in sorted(base.glob("*.jsonl")):
        recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        for rec in recs:
            scenes_total += 1
            turns = rec.get("turns") or []
            ok_turns = [t for t in turns if (t.get("assistant_text") or "").strip()]
            turns_total += len(turns)
            turns_ok += len(ok_turns)
            if rec.get("error") or len(ok_turns) < len(turns):
                scenes_with_error += 1
                errors.append({"scene_id": rec.get("scene_id"), "error": rec.get("error") or "incomplete turns"})
            else:
                scenes_complete += 1
            if rec.get("latency_s") is not None:
                scene_lat.append(float(rec["latency_s"]))
            for t in ok_turns:
                if t.get("latency_s") is not None:
                    turn_lat.append(float(t["latency_s"]))
                aa = t.get("assistant_audio") or ""
                render = t.get("tts_render")
                if not aa:
                    tts_count["none"] += 1
                    continue
                if render == "qwen3-tts-0.6b":
                    tts_count["qwen3-tts-0.6b"] += 1
                elif render == "qwen-audio-3.1-tts-next":
                    tts_count["qwen-audio-3.1-tts-next"] += 1
                else:
                    tts_count["other"] += 1
                candidates = [ROOT / aa, base / aa, base.parent / aa]
                found = next((c for c in candidates if c.is_file() and c.stat().st_size > 1024), None)
                if found:
                    audio_files_ok += 1
                else:
                    audio_files_missing += 1
    # 校验 total（以 data/scenes 选中场景为准）
    sel = load_json(DATA / "voice" / "selection.json", {"items": []})["items"]
    expected_scenes = sum(len(it.get("scene_ids") or []) for it in sel)
    expected_turns = 0
    for it in sel:
        cid = it["card_id"]
        doc = load_json(DATA / "scenes" / f"{cid}.json", {"scenes": []})
        smap = {s["scene_id"]: s for s in doc.get("scenes") or []}
        for sid in it.get("scene_ids") or []:
            expected_turns += len((smap.get(sid) or {}).get("turns") or [])
    return {
        "slug": slug,
        "scenes_total_expected": expected_scenes,
        "scenes_total_file": scenes_total,
        "scenes_complete": scenes_complete,
        "scenes_with_error": scenes_with_error,
        "turns_total_expected": expected_turns,
        "turns_total_file": turns_total,
        "turns_ok": turns_ok,
        "turn_latency_s": {"mean": round(sum(turn_lat) / len(turn_lat), 3) if turn_lat else 0.0,
                           "p50": pct(turn_lat, 50), "p95": pct(turn_lat, 95),
                           "max": round(max(turn_lat), 3) if turn_lat else 0.0},
        "scene_latency_s": {"mean": round(sum(scene_lat) / len(scene_lat), 3) if scene_lat else 0.0,
                            "p50": pct(scene_lat, 50), "p95": pct(scene_lat, 95)},
        "tts": tts_count,
        "audio_files_ok": audio_files_ok,
        "audio_files_missing": audio_files_missing,
        "errors": errors[:20],
        "error_count": len(errors),
    }


def main() -> int:
    asr = load_json(DATA / "audio_track" / "asr_summary.json", {})
    local = load_json(LOG_DIR / "tts_local_samples.json", {})
    result = {
        "asr": {
            "model": asr.get("model"),
            "transcribed": asr.get("transcribed"),
            "wer_mean": asr.get("wer_mean"),
            "wer_median": asr.get("wer_median"),
            "wer_p90": asr.get("wer_p90"),
            "exact_match_rate": asr.get("exact_match_rate"),
            "latency_s": asr.get("latency_s"),
        },
        "pipelines": [],
        "tts_local": {
            "model": local.get("model"),
            "count_ok": local.get("count_ok"),
            "count_failed": local.get("count_failed"),
            "speed": local.get("speed"),
            "elapsed_s": local.get("elapsed_s"),
            "plan": local.get("picked_plan"),
        },
        "tts_cloud": {},
    }
    for spec in PIPELINES:
        scan = scan_pipeline(spec["slug"])
        manifest = load_json(BASELINES / spec["slug"] / "_manifest.json", {})
        cloud = load_json(BASELINES / spec["slug"] / "audio_cloud" / "_render_manifest.json", {})
        scan.update({
            "pipeline": spec["key"],
            "model": spec["model"],
            "cascade_elapsed_s": manifest.get("elapsed_s"),
            "cascade_turn_latency_s": manifest.get("turn_latency_s"),
        })
        result["pipelines"].append(scan)
        if cloud:
            result["tts_cloud"][spec["model"]] = cloud
        # 修正 runner 续跑计数口径：用文件实测值回写 manifest
        if manifest:
            manifest["scenes_ok"] = scan["scenes_complete"]
            manifest["scenes_failed"] = scan["scenes_with_error"]
            manifest["turns_ok"] = scan["turns_ok"]
            manifest["turns_total"] = scan["turns_total_file"]
            manifest["verified_from_files_at"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%S")
            (BASELINES / spec["slug"] / "_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # 文本 track 对照
    result["text_baseline_turn_latency_s"] = {}
    for spec in PIPELINES:
        m = load_json(BASELINES / spec["model"] / "_manifest.json", {})
        if m:
            result["text_baseline_turn_latency_s"][spec["model"]] = m.get("turn_latency_s")

    # TTS 汇总（本地样本 + 云端全量）
    local_tts = load_json(LOG_DIR / "tts_local_samples.json", {})
    cleaned = load_json(LOG_DIR / "local_tts_cleaned.json", {})
    cloud_failures = load_json(LOG_DIR / "cloud_tts_failures.json", [])
    local_total = sum(p["tts"]["qwen3-tts-0.6b"] for p in result["pipelines"])
    cloud_total = sum(p["tts"]["qwen-audio-3.1-tts-next"] for p in result["pipelines"])
    other_total = sum(p["tts"]["other"] for p in result["pipelines"])
    no_audio = sum(p["tts"]["none"] for p in result["pipelines"])
    turns_total = sum(p["turns_total_file"] for p in result["pipelines"])
    turns_ok = sum(p["turns_ok"] for p in result["pipelines"])
    coverage = {}
    for p in result["pipelines"]:
        with_audio = p["turns_ok"] - p["tts"]["none"]
        coverage[p["model"]] = {
            "turns_total": p["turns_total_file"],
            "turns_ok": p["turns_ok"],
            "turns_with_audio": with_audio,
            "missing_audio": p["tts"]["none"],
            "audio_files_ok": p["audio_files_ok"],
            "audio_files_missing": p["audio_files_missing"],
        }
    kept = cleaned.get("kept") or local_tts.get("files") or []
    kept_durs = [k["duration_s"] for k in kept if k.get("duration_s")]
    speed = local_tts.get("speed") or {}
    per_item = speed.get("per_item_s_mean_uncontended") or speed.get("per_item_s_all")
    mean_dur = round(sum(kept_durs) / len(kept_durs), 2) if kept_durs else None
    # 云端渲染累计统计：解析 cloud_tts.log 的每次运行汇总（最后一次增量 run 会覆盖 _render_manifest.json，
    # 所以以日志累计为准；每个 slug 的 manifest 只反映最后一次增量运行）
    import re
    cloud_runs = []
    log_path = LOG_DIR / "cloud_tts.log"
    if log_path.is_file():
        pat = re.compile(r"云端渲染完成: (\d+)/(\d+) 条, 失败 (\d+), 用时 ([\d.]+)s \(([\d.]+) 条/分钟\)")
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pat.search(line)
            if m:
                cloud_runs.append({"rendered": int(m.group(1)), "failures": int(m.group(3)),
                                   "elapsed_s": float(m.group(4)), "per_min": float(m.group(5))})
    cloud_elapsed = round(sum(r["elapsed_s"] for r in cloud_runs), 1)
    cloud_rendered = sum(r["rendered"] for r in cloud_runs)
    cloud_fail_runs = sum(r["failures"] for r in cloud_runs)
    result["tts"] = {
        "local": {"render": "qwen3-tts-0.6b", "count": local_total, "count_total_wav": len(kept),
                  "speed": speed,
                  "mean_audio_duration_s": mean_dur,
                  "rtf_estimate": round(per_item / mean_dur, 2) if per_item and mean_dur else None,
                  "cleaned_bad": cleaned.get("cleaned_count", 0),
                  "cleaned_bad_samples": cleaned.get("cleaned", []),
                  "model": local_tts.get("model")},
        "cloud": {"render": "qwen-audio-3.1-tts-next", "count": cloud_total,
                  "runs": cloud_runs,
                  "rendered_total": cloud_rendered,
                  "failures": cloud_fail_runs + (len(cloud_failures) if isinstance(cloud_failures, list) else 0),
                  "elapsed_s": cloud_elapsed,
                  "throughput_per_min": round(cloud_rendered / cloud_elapsed * 60, 1) if cloud_elapsed else None,
                  "workers": 24, "rpm_limit": 180,
                  "last_incremental_manifests": result["tts_cloud"]},
        "other": other_total,
        "note": ("本地 Qwen3-TTS-0.6B 在这台 RTX 3090 上 RTF≈2-3（无 GPU 竞争时 ~20-30 s/条音频；"
                 "EdgeInstant 占用 GPU 后 70-170 s/条），全量 1568 条需 ~8 h，故仅保留 "
                 f"{local_total} 条本地链路验证样本（另 2 条 GPU 竞争下产生的异常样本已清理并改用云端），"
                 "全量音频判官用云端 qwen-audio-3.1-tts-next 渲染。"),
    }
    result["coverage"] = coverage
    result["asr"]["elapsed_s"] = (asr.get("info") or {}).get("elapsed_s")
    result["totals"] = {
        "turns": turns_total,
        "turns_ok": turns_ok,
        "turns_with_audio": local_total + cloud_total + other_total,
        "no_audio": no_audio,
        "tts_local": local_total,
        "tts_cloud": cloud_total,
    }

    import time
    result["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = BASELINES / "audio_pipeline_summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nsummary -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
