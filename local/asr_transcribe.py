#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B1/B2 级联链路的 ASR 环节：Qwen3-ASR-0.6B 转写 784 条用户 TTS 音频。

输入 : data/voice/manifest.jsonl（逐 turn：card_id/scene_id/idx/user_text/file）
输出 : data/audio_track/asr_transcripts.jsonl（逐 turn 一行，含 char-level WER）
       data/audio_track/asr_summary.json

环境:
  * sys.path 前置 vendor/asr（qwen-asr 0.0.6 + transformers 4.57.6，均装在项目 vendor 目录，
    不影响 torch venv 的 transformers 5.19）
  * torch 用 E:\\RBM\\algorithms\\draft\\qwen3_5_infra_test\\qwen_bench_win\\.venv 的 CUDA 版（cu130）
  * mp3 解码用 ffmpeg（不走 librosa/av）
  * ASR 模型 bf16 占用 ~2GB 显存；跑完进程退出即释放

用法:
  python local/asr_transcribe.py --limit 3          # 冒烟
  python local/asr_transcribe.py                    # 全量 784 条
  python local/asr_transcribe.py --rebuild-summary  # 仅重算 summary
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR_ASR = ROOT / "vendor" / "asr"
sys.path.insert(0, str(VENDOR_ASR))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ASR_DIR = Path(r"E:\RBM\models\Qwen3-ASR-0.6B")
VOICE_DIR = ROOT / "data" / "voice"
OUT_DIR = ROOT / "data" / "audio_track"
OUT_JSONL = OUT_DIR / "asr_transcripts.jsonl"
OUT_SUMMARY = OUT_DIR / "asr_summary.json"
LOG_DIR = ROOT / "logs" / "audio_track"
SAMPLE_RATE = 16000

log = logging.getLogger("asr")


# --------------------------------------------------------------------------- #
# 文本归一化 / WER
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """中文按字符、英文按词：去掉空白与标点，英文小写。"""
    import re
    text = text.strip().lower()
    # 保留 CJK 字符、英文字母、数字
    text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbfa-z0-9]", "", text)
    return text


def edit_distance(ref: str, hyp: str) -> int:
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1]


def char_wer(ref: str, hyp: str) -> float:
    r, h = normalize_text(ref), normalize_text(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return round(edit_distance(r, h) / len(r), 4)


# --------------------------------------------------------------------------- #
# 音频解码
# --------------------------------------------------------------------------- #
_FFMPEG: str | None = None


def find_ffmpeg() -> str:
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    exe = shutil.which("ffmpeg")
    if not exe:
        fallback = Path(r"E:\Apps\FFmPeg\ffmpeg-2026-03-12-git-9dc44b43b2-full_build\bin\ffmpeg.exe")
        if fallback.is_file():
            exe = str(fallback)
    if not exe:
        raise FileNotFoundError("ffmpeg not found")
    _FFMPEG = exe
    return exe


def decode_audio(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    proc = subprocess.run(
        [find_ffmpeg(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def read_manifest() -> list[dict]:
    items = []
    with (VOICE_DIR / "manifest.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def read_existing() -> dict[tuple[str, str, int], dict]:
    done: dict[tuple[str, str, int], dict] = {}
    if not OUT_JSONL.is_file():
        return done
    with OUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[(rec["card_id"], rec["scene_id"], int(rec["idx"]))] = rec
    return done


def write_summary(model_info: dict) -> dict:
    records = []
    if OUT_JSONL.is_file():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    wers = sorted(r["wer"] for r in records)
    lat = sorted(r.get("latency_s") or 0.0 for r in records)
    summary = {
        "model": str(ASR_DIR),
        "manifest_items": len(read_manifest()),
        "transcribed": len(records),
        "wer_mean": round(sum(wers) / len(wers), 4) if wers else None,
        "wer_median": round(wers[len(wers) // 2], 4) if wers else None,
        "wer_p90": round(wers[min(len(wers) - 1, int(len(wers) * 0.9))], 4) if wers else None,
        "wer_max": round(wers[-1], 4) if wers else None,
        "exact_match_rate": round(sum(1 for w in wers if w == 0) / len(wers), 4) if wers else None,
        "latency_s": {
            "mean": round(sum(lat) / len(lat), 3) if lat else None,
            "p50": round(lat[len(lat) // 2], 3) if lat else None,
            "p95": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 3) if lat else None,
        },
        "info": model_info,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟）")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--rebuild-summary", action="store_true", help="只重算 summary，不转写")
    ap.add_argument("--log-file", default=str(LOG_DIR / "asr_transcribe.log"))
    a = ap.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(a.log_file, encoding="utf-8")],
        force=True,
    )

    if a.rebuild_summary:
        summary = write_summary({"rebuild": True})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    import torch

    # 仅做中文 ASR，不需要 forced aligner（它依赖 nagisa/dynet 且仅日文分词用）。
    # 在导入 qwen_asr 前把 forced_aligner 模块替换成占位模块，避免拉起 nagisa→dynet 依赖。
    import types

    fa_mod = types.ModuleType("qwen_asr.inference.qwen3_forced_aligner")

    class _DisabledForcedAligner:  # pragma: no cover - 占位
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("forced aligner disabled in this environment")

    fa_mod.Qwen3ForcedAligner = _DisabledForcedAligner
    sys.modules["qwen_asr.inference.qwen3_forced_aligner"] = fa_mod

    from qwen_asr import Qwen3ASRModel

    items = read_manifest()
    done = read_existing()
    todo = [it for it in items if (it["card_id"], it["scene_id"], int(it["idx"])) not in done]
    if a.limit and a.limit > 0:
        todo = todo[: a.limit]
    log.info("manifest=%d, 已转写=%d, 本次待转写=%d", len(items), len(done), len(todo))
    if not todo:
        write_summary({"skipped": "all done"})
        log.info("无需转写，summary 已刷新")
        return 0

    t_load = time.perf_counter()
    log.info("加载 ASR 模型: %s (bf16, cuda:0, sdpa)", ASR_DIR)
    model = Qwen3ASRModel.from_pretrained(
        str(ASR_DIR),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        max_inference_batch_size=a.batch_size,
        max_new_tokens=a.max_new_tokens,
    )
    log.info("模型加载完成 %.1fs", time.perf_counter() - t_load)

    written = 0
    t_start = time.perf_counter()
    out_f = OUT_JSONL.open("a", encoding="utf-8")
    try:
        for b0 in range(0, len(todo), a.batch_size):
            batch = todo[b0: b0 + a.batch_size]
            audios = []
            for it in batch:
                audios.append((decode_audio(ROOT / it["file"]), SAMPLE_RATE))
            t0 = time.perf_counter()
            results = model.transcribe(audio=audios, language=["Chinese"] * len(audios))
            dt = time.perf_counter() - t0
            per_item_lat = round(dt / len(batch), 3)
            for it, res in zip(batch, results):
                asr_text = (res.text or "").strip()
                wer = char_wer(it["user_text"], asr_text)
                rec = {
                    "card_id": it["card_id"],
                    "scene_id": it["scene_id"],
                    "idx": int(it["idx"]),
                    "audio": it["file"],
                    "asr_text": asr_text,
                    "user_text_gt": it["user_text"],
                    "wer": wer,
                    "latency_s": per_item_lat,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            out_f.flush()
            log.info(
                "[%d/%d] batch=%d %.2fs (%.2fs/item) 累计 WER 样例: %r -> %r (wer=%.3f)",
                written, len(todo), len(batch), dt, per_item_lat,
                batch[0]["user_text"], results[0].text, char_wer(batch[0]["user_text"], results[0].text or ""),
            )
    finally:
        out_f.close()

    elapsed = time.perf_counter() - t_start
    summary = write_summary({"limit": a.limit, "batch_size": a.batch_size,
                             "requests_this_run": len(todo), "elapsed_s": round(elapsed, 1)})
    log.info("ASR 完成: %d 条, 用时 %.1fs | mean WER=%.4f exact=%.1f%%",
             written, elapsed, summary["wer_mean"] or 0, (summary["exact_match_rate"] or 0) * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
