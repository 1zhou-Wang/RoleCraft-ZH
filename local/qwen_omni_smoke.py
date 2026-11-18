# -*- coding: utf-8 -*-
"""Smoke test Qwen2.5-Omni-3B role-play: text-in and audio-in single turns.
Usage: python local/qwen_omni_smoke.py [--processor-only]
"""
import sys, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local"))
import qwen_omni_common as C  # noqa: E402

def main():
    cards, scenes = C.load_card("rc01"), C.scene_map()
    sid = "rc01_s01"
    scene = scenes[sid]
    turn = scene["turns"][0]
    conv_t = C.text_conversation(cards, [], turn["user_text"])
    conv_a = C.audio_conversation(cards, [], ROOT / "data" / "voice" / "audio" / "rc01_s01_1.mp3")
    if "--processor-only" in sys.argv:
        from transformers import Qwen2_5OmniProcessor
        p = Qwen2_5OmniProcessor.from_pretrained(str(C.MODEL_DIR))
        print("== text template ==")
        print(p.apply_chat_template(conv_t, add_generation_prompt=True, tokenize=False)[:600])
        print("== audio template ==")
        print(p.apply_chat_template(conv_a, add_generation_prompt=True, tokenize=False)[:600])
        return
    t0 = time.time(); model, processor = C.load_model(); print("load_model", round(time.time()-t0,1), "s", flush=True)
    r, dt, n = C._generate(model, processor, conv_t, audio_mode=False)
    print("TEXT:", r, "|", dt, "s | prompt", n, flush=True)
    r2, dt2, n2 = C._generate(model, processor, conv_a, audio_mode=True)
    print("AUDIO:", r2, "|", dt2, "s | prompt", n2, flush=True)
    import torch
    print("VRAM used:", round(torch.cuda.memory_allocated()/1e9, 2), "GB allocated /", round(torch.cuda.memory_reserved()/1e9,2), "GB reserved")

if __name__ == "__main__":
    main()
