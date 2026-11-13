#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TTS 性能剖析：拆分 talker 生成 vs speech tokenizer 解码的耗时，定位瓶颈。

用法: python local/audio_pipeline/profile_tts.py [--compile]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "tts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


def main() -> int:
    import torch
    from qwen_tts import Qwen3TTSModel

    tts_dir = r"E:\RBM\models\Qwen3-TTS-12Hz-0.6B-CustomVoice"
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(tts_dir, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    print(f"load: {time.perf_counter()-t0:.1f}s")
    m = model.model
    st = m.speech_tokenizer
    print("main device:", next(m.parameters()).device, "attn impl:", getattr(m.config, "_attn_implementation", None))
    print("tokenizer device:", next(st.model.parameters()).device, "sr_out:", st.get_output_sample_rate())

    texts = [
        "我是罗小黑呀，一只黑猫妖。你问我是谁？",
        "唔！我最爱那枚捡来的蓝色玻璃弹珠！每次摸它，心里就暖洋洋的。墨叔那家伙虽然啰嗦，但总给我讲有趣的故事。",
        "（轻声笑了笑）你说的这些我都记着呢。人类和妖精一起生活，只要彼此不伤害，就一定能找到办法。",
    ]
    for text in texts:
        input_ids = m._tokenize_texts([m._build_assistant_text(text)])
        t1 = time.perf_counter()
        with torch.inference_mode():
            codes, extra = m.generate(
                input_ids=input_ids, instruct_ids=[None], languages=["Chinese"],
                speakers=["Vivian"], non_streaming_mode=True, do_sample=False,
            )
        t_gen = time.perf_counter() - t1
        c0 = codes[0]
        shape = tuple(c0.shape) if hasattr(c0, "shape") else "?"
        t2 = time.perf_counter()
        with torch.inference_mode():
            wavs, fs = st.decode([{"audio_codes": c} for c in codes])
        t_dec = time.perf_counter() - t2
        dur = len(wavs[0]) / fs
        ntok = int(np.prod(shape)) if hasattr(c0, "shape") else 0
        print(f"text={len(text)}字 gen={t_gen:.2f}s decode={t_dec:.2f}s total={t_gen+t_dec:.2f}s "
              f"audio={dur:.2f}s codes={shape} tok/s={(ntok/t_gen if t_gen else 0):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
