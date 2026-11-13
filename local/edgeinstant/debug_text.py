#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文字 track 5.19 路径调试：定位 'dict' object has no attribute 'shape'。"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ei_common as C  # noqa: E402

C.set_utf8_stdio()


def main() -> int:
    import torch
    import transformers

    print("transformers", transformers.__version__, flush=True)
    proc = C.build_processor(log=print)
    model = C.load_model(log=print)
    device = next(model.parameters()).device

    sys_prompt = C.build_system_prompt("rc01")
    inputs = proc(text="你是谁呀？", system_prompt=sys_prompt, history=[], enable_thinking=False)
    print("inputs type:", type(inputs), "keys:", list(inputs.keys()), flush=True)
    for k, v in inputs.items():
        print(f"  {k}: {type(v)}", getattr(v, "shape", None), getattr(v, "dtype", None), flush=True)
    prompt_len = int(inputs["input_ids"].shape[1])
    print("prompt_len", prompt_len, flush=True)

    moved = inputs.to(device)
    print("moved type:", type(moved), "keys:", list(moved.keys()) if hasattr(moved, "keys") else "NO .keys", flush=True)
    print("moved input_ids type:", type(moved["input_ids"]), getattr(moved["input_ids"], "device", None), flush=True)

    try:
        with torch.inference_mode():
            out = model.generate(**moved, max_new_tokens=32, do_sample=False)
        print("generate OK; out type:", type(out), flush=True)
        if isinstance(out, dict):
            print("dict keys:", list(out.keys()), flush=True)
        else:
            print("out shape:", getattr(out, "shape", None), flush=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        # 进一步尝试直接 thinker.generate
        try:
            print("--- retry via model.thinker.generate ---", flush=True)
            with torch.inference_mode():
                out2 = model.thinker.generate(input_ids=moved["input_ids"],
                                              attention_mask=moved["attention_mask"],
                                              max_new_tokens=32, do_sample=False)
            print("thinker.generate OK type:", type(out2), getattr(out2, "shape", None), flush=True)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
