#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端 TTS 内容审核拒绝（DataInspectionFailed）条目的本地兜底渲染。

背景：official 链路云端渲染中个别回复文本被云端 TTS 内容审核拒绝（HTTP 400
DataInspectionFailed，例如涉及「死亡/灵魂/没有痛苦」等主题的回复），无法用云端渲染。
兜底：用本地 Qwen3-TTS-0.6B 对这些 turn 合成 wav（无内容审核），写回
assistant_audio="audio/<scene_id>_<idx>.wav" + tts_render="qwen3-tts-0.6b"。

用法:
  python local/audio_pipeline/local_tts_fallback.py --slugs qwen3.5-2b-official,qwen3.5-4b-official
  python local/audio_pipeline/local_tts_fallback.py --log logs/audio_track/cloud_tts_official.log --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
LOG_DIR = ROOT / "logs" / "audio_track"
RENDER_LOCAL = "qwen3-tts-0.6b"
FAIL_RE = re.compile(r"ERROR 失败 ([A-Za-z0-9_]+)_(\d+): http_400")


def load_records(slug: str):
    order, records = {}, {}
    for path in sorted((BASELINES / f"audio_pipeline_{slug}").glob("*.jsonl")):
        key = path.stem
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


def parse_failures(log_path: Path) -> list[dict]:
    """优先读 cloud_tts_failures.json（含 slug），否则从日志解析 (scene_id, idx)。"""
    fail_json = LOG_DIR / "cloud_tts_failures.json"
    if fail_json.is_file():
        try:
            data = json.loads(fail_json.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return [{"slug": d.get("slug"), "scene_id": d["scene_id"], "idx": int(d["idx"])} for d in data]
        except Exception:  # noqa: BLE001
            pass
    seen: list[dict] = []
    if not log_path.is_file():
        return seen
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = FAIL_RE.search(line)
        if m:
            sid, idx = m.group(1), int(m.group(2))
            if not any(x["scene_id"] == sid and x["idx"] == idx for x in seen):
                seen.append({"slug": None, "scene_id": sid, "idx": idx})
    return seen


def find_item(slugs: list[str], slug_hint: str | None, sid: str, idx: int):
    """按 slug_hint（失败记录里的 slug）优先定位，避免同 scene/idx 在两个模型间串号。"""
    candidates = [slug_hint] if slug_hint else list(slugs)
    for slug in candidates:
        if slug not in slugs:
            continue
        order, records = load_records(slug)
        for card_id, recs in records.items():
            rec = recs.get(sid)
            if not rec:
                continue
            turn = next((t for t in rec.get("turns") or [] if int(t.get("idx", 0)) == idx), None)
            if turn and (turn.get("assistant_text") or "").strip():
                # 已经渲染过的（有 assistant_audio）且内容未变则跳过
                return slug, card_id, order, records, rec, turn
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", default="qwen3.5-2b-official,qwen3.5-4b-official")
    ap.add_argument("--log", default=str(LOG_DIR / "cloud_tts_official.log"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    slugs = [s.strip() for s in a.slugs.split(",") if s.strip()]
    fails = parse_failures(Path(a.log))
    print(f"失败条目 {len(fails)} 个: {fails}")
    if not fails:
        return 0
    if a.dry_run:
        for f in fails:
            item = find_item(slugs, f.get("slug"), f["scene_id"], f["idx"])
            print(f["scene_id"], f["idx"], "->",
                  None if item is None else (item[0], item[1], item[5]["assistant_text"][:60]))
        return 0

    import torch
    from qwen_tts import Qwen3TTSModel

    t0 = time.perf_counter()
    print(f"加载本地 TTS 模型: {TTS_DIR}")
    model = Qwen3TTSModel.from_pretrained(str(TTS_DIR), device_map="cuda:0",
                                          dtype=torch.bfloat16, attn_implementation="sdpa")
    print(f"模型加载完成 {time.perf_counter()-t0:.1f}s")

    results = []
    for f in fails:
        sid, idx = f["scene_id"], int(f["idx"])
        item = find_item(slugs, f.get("slug"), sid, idx)
        if item is None:
            results.append({"scene_id": sid, "idx": idx, "ok": False, "error": "turn not found"})
            continue
        slug, card_id, order, records, rec, turn = item
        text = turn["assistant_text"].strip()
        wav = BASELINES / f"audio_pipeline_{slug}" / "audio" / f"{sid}_{idx}.wav"
        t1 = time.perf_counter()
        try:
            with torch.inference_mode():
                wavs, sr = model.generate_custom_voice(text=[text], speaker=SPEAKER, language="Chinese",
                                                       do_sample=False, non_streaming_mode=True)
            save_wav(wav, wavs[0], sr)
            turn["assistant_audio"] = f"audio/{sid}_{idx}.wav"
            turn["tts_render"] = RENDER_LOCAL
            write_records(slug, card_id, order[card_id], records[card_id])
            results.append({"slug": slug, "scene_id": sid, "idx": idx, "ok": True,
                            "wav": str(wav.relative_to(ROOT)).replace("\\", "/"),
                            "elapsed_s": round(time.perf_counter() - t1, 2),
                            "audio_duration_s": round(len(wavs[0]) / sr, 2)})
            print(f"OK {slug}/{sid}#{idx} -> {wav.name} ({len(wavs[0])/sr:.1f}s audio)")
        except Exception as exc:  # noqa: BLE001
            results.append({"slug": slug, "scene_id": sid, "idx": idx, "ok": False, "error": str(exc)[:300]})
            print(f"FAIL {slug}/{sid}#{idx}: {exc}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"items": results, "ok": sum(1 for r in results if r["ok"]),
               "failed": sum(1 for r in results if not r["ok"]),
               "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (LOG_DIR / "local_tts_fallback_official.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"fallback 完成: ok={summary['ok']} failed={summary['failed']}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
