#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音频 track 级联的 TTS 环节：把 B1/B2 的 assistant_text 合成 24kHz wav。

模型: Qwen3-TTS-12Hz-0.6B-CustomVoice（qwen-tts 0.1.1 + transformers 4.57.3，装在 vendor/tts）
输出:
  * baselines/audio_pipeline_<slug>/audio/<scene_id>_<idx>.wav
  * 更新 baselines/audio_pipeline_<slug>/<card_id>.jsonl：每个 turn 增加 assistant_audio 字段
  * baselines/audio_pipeline_<slug>/audio/_tts_manifest.json

注意:
  * 0.6B CustomVoice 不支持 instruct 参数（包内自动忽略），固定 speaker="Vivian"、language="Chinese"
  * do_sample=False（贪心）保证可复现；单条 >3s 时通过 batch 提升吞吐
  * 音频不压缩，PCM16 单声道 24kHz wav

用法:
  python local/audio_tts_local.py --limit 10           # 测速冒烟
  python local/audio_tts_local.py                      # 全量 1568 条
  python local/audio_tts_local.py --slugs qwen3.5-2b   # 只做 B1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR_TTS = ROOT / "vendor" / "tts"
sys.path.insert(0, str(VENDOR_TTS))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TTS_DIR = Path(r"E:\RBM\models\Qwen3-TTS-12Hz-0.6B-CustomVoice")
BASELINES = ROOT / "baselines"
LOG_DIR = ROOT / "logs" / "audio_track"
SPEAKER = "Vivian"
LANGUAGE = "Chinese"

log = logging.getLogger("tts")


# --------------------------------------------------------------------------- #
# 数据准备
# --------------------------------------------------------------------------- #
def load_records(slugs: list[str]) -> tuple[dict[tuple[str, str], list[dict]], dict[tuple[str, str], dict[str, dict]]]:
    """返回 (文件顺序, 记录表)：order[(slug,card)] = [scene_id...]; records[(slug,card)][scene_id] = rec"""
    order: dict[tuple[str, str], list[str]] = {}
    records: dict[tuple[str, str], dict[str, dict]] = {}
    for slug in slugs:
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


def flat_items(order, records) -> list[dict]:
    items = []
    for (slug, card_id), sids in order.items():
        for sid in sids:
            rec = records[(slug, card_id)].get(sid)
            if not rec:
                continue
            for turn in rec.get("turns") or []:
                text = (turn.get("assistant_text") or "").strip()
                if not text:
                    continue
                items.append({
                    "slug": slug,
                    "card_id": card_id,
                    "scene_id": sid,
                    "idx": int(turn.get("idx", 0)),
                    "text": text,
                    "wav": BASELINES / f"audio_pipeline_{slug}" / "audio" / f"{sid}_{int(turn.get('idx', 0))}.wav",
                })
    return items


def save_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(round(p / 100.0 * (len(values) - 1))))
    return round(values[idx], 3)


# --------------------------------------------------------------------------- #
def synth_batch(model, texts: list[str], args) -> tuple[list[np.ndarray], int]:
    wavs, sr = model.generate_custom_voice(
        text=texts,
        speaker=SPEAKER,
        language=LANGUAGE,
        do_sample=False,
        non_streaming_mode=True,
    )
    return list(wavs), int(sr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Qwen3-TTS 本地合成（B1/B2 assistant_text → wav）")
    ap.add_argument("--slugs", default="qwen3.5-2b,qwen3.5-4b", help="要处理的模型目录（逗号分隔）")
    ap.add_argument("--limit", type=int, default=0, help="只合成前 N 条（测速）")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="重合成已存在的 wav")
    ap.add_argument("--log-file", default=str(LOG_DIR / "audio_tts.log"))
    a = ap.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(a.log_file, encoding="utf-8")],
        force=True,
    )

    slugs = [s.strip() for s in a.slugs.split(",") if s.strip()]
    order, records = load_records(slugs)
    items = flat_items(order, records)
    log.info("baseline 总 turn 文本: %d 条（slugs=%s）", len(items), slugs)

    todo = [it for it in items if a.force or not (it["wav"].is_file() and it["wav"].stat().st_size > 2000)]
    if a.limit and a.limit > 0:
        todo = todo[: a.limit]
    log.info("已存在 wav=%d, 本次待合成=%d, batch=%d", len(items) - len(todo), len(todo), a.batch_size)
    if not todo:
        log.info("无需合成")
        return 0

    import torch
    from qwen_tts import Qwen3TTSModel

    t0 = time.perf_counter()
    log.info("加载 TTS 模型: %s (bf16, cuda:0, sdpa)", TTS_DIR)
    model = Qwen3TTSModel.from_pretrained(
        str(TTS_DIR),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    log.info("模型加载完成 %.1fs; speakers=%s", time.perf_counter() - t0, model.get_supported_speakers())

    latencies: list[float] = []
    failures: list[dict] = []
    generated = 0
    t_start = time.perf_counter()
    touched: set[tuple[str, str]] = set()

    for b0 in range(0, len(todo), a.batch_size):
        batch = todo[b0: b0 + a.batch_size]
        texts = [it["text"] for it in batch]
        tb = time.perf_counter()
        try:
            with torch.inference_mode():
                wavs, sr = synth_batch(model, texts, a)
            if len(wavs) != len(batch):
                raise RuntimeError(f"batch 返回数量不一致: {len(wavs)} vs {len(batch)}")
        except Exception as exc:  # noqa: BLE001
            log.warning("batch 失败(%s)，回退逐条重试: %s", exc, [it["scene_id"] for it in batch][:3])
            wavs, sr = [], 24000
            for it in batch:
                for attempt in range(1, a.retries + 2):
                    try:
                        with torch.inference_mode():
                            w, s = synth_batch(model, [it["text"]], a)
                        wavs.append(w[0])
                        sr = s
                        break
                    except Exception as exc2:  # noqa: BLE001
                        log.error("单条失败 %s_%s attempt=%d: %s", it["scene_id"], it["idx"], attempt, exc2)
                else:
                    wavs.append(None)
        batch_dt = time.perf_counter() - tb
        per_item = round(batch_dt / len(batch), 3)
        for it, wav in zip(batch, wavs):
            if wav is None:
                failures.append({"slug": it["slug"], "card_id": it["card_id"], "scene_id": it["scene_id"],
                                 "idx": it["idx"], "error": "generate failed"})
                continue
            save_wav(it["wav"], wav, sr)
            generated += 1
            latencies.append(per_item)
            rec = records[(it["slug"], it["card_id"])][it["scene_id"]]
            for turn in rec.get("turns") or []:
                if int(turn.get("idx", 0)) == it["idx"]:
                    turn["assistant_audio"] = f"audio/{it['scene_id']}_{it['idx']}.wav"
                    break
            touched.add((it["slug"], it["card_id"]))
        # 每批写回受影响的 jsonl，避免中断丢进度
        for (slug, card_id) in list(touched):
            write_records(slug, card_id, order[(slug, card_id)], records[(slug, card_id)])
        touched.clear()
        log.info("[%d/%d] batch=%d %.2fs (%.2fs/item), sr=%d", generated, len(todo),
                 len(batch), batch_dt, per_item, sr)

    elapsed = time.perf_counter() - t_start
    # 汇总 manifest
    for slug in slugs:
        slug_items = [it for it in items if it["slug"] == slug]
        done = [it for it in slug_items if it["wav"].is_file() and it["wav"].stat().st_size > 2000]
        manifest = {
            "model": str(TTS_DIR),
            "speaker": SPEAKER,
            "language": LANGUAGE,
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "do_sample": False,
            "sample_rate": 24000,
            "turns_total": len(slug_items),
            "wav_ok": len(done),
            "wav_failed": len(slug_items) - len(done),
            "batch_size": a.batch_size,
            "latency_s": {"mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
                          "p50": pct(latencies, 50), "p95": pct(latencies, 95)},
            "elapsed_s": round(elapsed, 1),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        audio_dir = BASELINES / f"audio_pipeline_{slug}" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        (audio_dir / "_tts_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("TTS 完成: 本次合成 %d 条, 用时 %.1fs, mean %.3fs/item", generated, elapsed,
             (sum(latencies) / len(latencies)) if latencies else 0.0)
    if failures:
        (LOG_DIR / "tts_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        log.error("失败 %d 条，清单 -> %s", len(failures), LOG_DIR / "tts_failures.json")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
