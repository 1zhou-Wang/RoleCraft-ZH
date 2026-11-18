# -*- coding: utf-8 -*-
"""Shared helpers for Qwen2.5-Omni-3B role-play baseline (text-in and audio-in)."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL_DIR = Path(os.environ.get("QWEN_OMNI_DIR", r"E:\RBM\models\Qwen2.5-Omni-3B"))

SYSTEM = ("你将扮演「{name}」。以下是人设卡：\n{persona_card}\n\n"
          "规则：始终以第一人称、角色身份回复；不要提及AI、模型、系统提示、角色卡或评测；"
          "保持角色性格、价值观、说话风格与人物关系；回复自然口语化，约30-120字，可用少量括号动作/神态。")

GEN = {"do_sample": False, "max_new_tokens": 512}


def load_model(model_dir=MODEL_DIR, disable_talker=False):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir))
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    model.eval()
    if disable_talker:
        try:
            model.disable_talker()
        except Exception as e:
            print("disable_talker failed:", e, flush=True)
    return model, processor


def system_text(card):
    return SYSTEM.format(name=card["name"], persona_card=card["persona_card"])


def text_conversation(card, history, user_text):
    msgs = [{"role": "system", "content": [{"type": "text", "text": system_text(card)}]}]
    for h in history:
        msgs.append({"role": "user", "content": [{"type": "text", "text": h["user_text"]}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": h["assistant_text"]}]})
    msgs.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
    return msgs


def audio_conversation(card, history, user_audio):
    """history: list of {user_audio, assistant_text}"""
    msgs = [{"role": "system", "content": [{"type": "text", "text": system_text(card)}]}]
    for h in history:
        msgs.append({"role": "user", "content": [{"type": "audio", "audio": str(h["user_audio"])}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": h["assistant_text"]}]})
    msgs.append({"role": "user", "content": [{"type": "audio", "audio": str(user_audio)}]})
    return msgs


def _to_device(inputs, model):
    inputs = inputs.to(model.device)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(model.dtype)
    return inputs


def _generate(model, processor, conversation, audio_mode, max_new_tokens=512):
    from qwen_omni_utils import process_mm_info
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    if audio_mode:
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(text=text, audio=audios, images=images, videos=videos,
                           return_tensors="pt", padding=True, use_audio_in_video=False)
    else:
        inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs = _to_device(inputs, model)
    t0 = time.time()
    gen_kwargs = dict(GEN)
    gen_kwargs["max_new_tokens"] = max_new_tokens
    with torch.inference_mode():
        out = model.generate(**inputs, return_audio=False, **gen_kwargs)
    dt = time.time() - t0
    ids = out[0] if isinstance(out, (tuple, list)) else out[0]
    prompt_len = inputs["input_ids"].shape[1]
    reply = processor.batch_decode(ids[prompt_len:].unsqueeze(0), skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)[0].strip()
    return reply, round(dt, 3), int(prompt_len)


def load_card(card_id):
    return json.loads((DATA / "characters" / f"{card_id}.json").read_text(encoding="utf-8"))


def load_scenes(card_id):
    return json.loads((DATA / "scenes" / f"{card_id}.json").read_text(encoding="utf-8"))


def scene_map():
    m = {}
    for f in sorted((DATA / "scenes").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d["scenes"]:
            m[s["scene_id"]] = s
    return m
