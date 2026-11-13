#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASR 环境探针：验证 transformers 5.19 原生 qwen3_asr 能否加载并对真实 mp3 转写。

只做 1 条音频的端到端测试，打印加载/转写耗时、输出文本。
用法: python local/audio_pipeline/probe_asr.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
ASR_DIR = Path(r"E:\RBM\models\Qwen3-ASR-0.6B")


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    fallback = Path(r"E:\Apps\FFmPeg\ffmpeg-2026-03-12-git-9dc44b43b2-full_build\bin\ffmpeg.exe")
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError("ffmpeg not found")


def decode_audio(path: Path, sr: int = 16000) -> np.ndarray:
    ff = find_ffmpeg()
    proc = subprocess.run(
        [ff, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def main() -> int:
    import torch
    from transformers import AutoProcessor
    from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration

    t0 = time.perf_counter()
    print("[1] loading processor ...", flush=True)
    processor = AutoProcessor.from_pretrained(str(ASR_DIR))
    print(f"    processor={type(processor).__name__} ({time.perf_counter()-t0:.1f}s)")

    t1 = time.perf_counter()
    print("[2] loading model (bf16, cuda:0, sdpa) ...", flush=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        str(ASR_DIR), dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
    )
    model.eval()
    print(f"    model loaded ({time.perf_counter()-t1:.1f}s)")

    manifest = ROOT / "data" / "voice" / "manifest.jsonl"
    items = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()][:1]
    item = items[0]
    audio_path = ROOT / item["file"]
    print(f"[3] audio={audio_path.name} duration={item['duration']}s gt={item['user_text']!r}")

    t2 = time.perf_counter()
    audio = decode_audio(audio_path)
    print(f"    ffmpeg decode: {len(audio)} samples ({len(audio)/16000:.2f}s) in {time.perf_counter()-t2:.2f}s")

    t3 = time.perf_counter()
    inputs = processor.apply_transcription_request(audio=(audio, 16000), language="Chinese")
    print(f"    processor: keys={list(inputs.keys())} ({time.perf_counter()-t3:.2f}s)")
    inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)

    t4 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    print(f"    generate: {out.shape} ({time.perf_counter()-t4:.2f}s)")
    text = processor.decode(out[0], return_format="transcription_only")
    print(f"[4] ASR text = {text!r}")
    print(f"    GT       = {item['user_text']!r}")
    print(f"    match    = {text.strip() == item['user_text'].strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
