#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EdgeInstant 音频/文字 track 公共工具。

  * vendor/edgeinstant 路径切换（transformers==5.12.1，不重装 torch）
  * 手动构造 EdgeInstantProcessor（绕开 AutoImageProcessor 的 PIL/torchvision 依赖）
  * ffmpeg 解码 -> 16k 单声道 float32（带磁盘缓存）
  * 角色扮演 system prompt（与文字 track 基线完全一致）

环境变量：
  EDGEINSTANT_VENDOR=1  优先使用 vendor/edgeinstant 下的 transformers 5.12.1（音频 track 必须）
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(r"E:\RBM\models\EdgeInstant-Omni-2B-1106-Preview")
VENDOR_DIR = PROJECT_ROOT / "vendor" / "edgeinstant"
CHARACTERS_DIR = PROJECT_ROOT / "data" / "characters"
SCENES_DIR = PROJECT_ROOT / "data" / "scenes"
VOICE_DIR = PROJECT_ROOT / "data" / "voice"
MANIFEST_PATH = VOICE_DIR / "manifest.jsonl"
SELECTION_PATH = VOICE_DIR / "selection.json"
AUDIO_CACHE_DIR = PROJECT_ROOT / "local" / "edgeinstant" / "cache_audio16k"
SR = 16000

SYSTEM_PROMPT_TEMPLATE = (
    "你将扮演「{name}」。以下是人设卡：\n{persona_card}\n\n"
    "规则：始终以第一人称、角色身份回复；不要提及AI、模型、系统提示、角色卡或评测；"
    "保持角色性格、价值观、说话风格与人物关系；回复自然口语化，约30-120字，可用少量括号动作/神态。"
)

_vendor_active = False


def activate_vendor() -> bool:
    """EDGEINSTANT_VENDOR=1 时把 vendor 目录放到 sys.path 最前，并关掉 hub kernels。

    venv 里 kernels 0.17 与 transformers 5.12.1 的 Hub kernel 注册不兼容，必须 USE_HUB_KERNELS=NO。
    必须在 import transformers 之前调用。
    """
    global _vendor_active
    if os.environ.get("EDGEINSTANT_VENDOR") != "1":
        return False
    if not VENDOR_DIR.is_dir():
        print(f"[vendor] WARNING: EDGEINSTANT_VENDOR=1 but {VENDOR_DIR} missing", flush=True)
        return False
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    os.environ.setdefault("USE_HUB_KERNELS", "NO")
    _vendor_active = True
    print(f"[vendor] active: sys.path[0]={VENDOR_DIR} USE_HUB_KERNELS={os.environ.get('USE_HUB_KERNELS')}",
          flush=True)
    return True


def set_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


def build_processor(model_path: str | Path = MODEL_PATH, log=print):
    """手动构造 EdgeInstantProcessor，不触发图像处理器（音频/文本 track 都不需要）。"""
    activate_vendor()
    from transformers import AutoTokenizer, WhisperFeatureExtractor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_path = str(model_path)
    proc_cls = get_class_from_dynamic_module(
        "processing_edgeinstant.EdgeInstantProcessor", model_path, local_files_only=True)
    cfg_cls = get_class_from_dynamic_module(
        "configuration_edgeinstant.EdgeInstantConfig", model_path, local_files_only=True)
    config = cfg_cls.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, config=config.thinker_config, trust_remote_code=True, local_files_only=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    log(f"processor: config={type(config).__name__} tokenizer={type(tokenizer).__name__} "
        f"fe={type(feature_extractor).__name__} (image_processor=None)")
    return proc_cls(tokenizer=tokenizer, feature_extractor=feature_extractor,
                    config=config, image_processor=None)


def build_system_prompt(card_id: str, characters_dir: Path = CHARACTERS_DIR) -> str:
    card = json.loads((characters_dir / f"{card_id}.json").read_text(encoding="utf-8"))
    return SYSTEM_PROMPT_TEMPLATE.format(name=str(card["name"]).strip(),
                                        persona_card=str(card["persona_card"]).strip())


def load_audio_16k(path: str | Path, sr: int = SR, cache_dir: Path | None = None):
    """ffmpeg 解码任意音频 -> 16k 单声道 float32（默认带磁盘缓存，避免重复解码）。"""
    import numpy as np

    path = Path(path)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16] + f"_{sr}.f32"
        cache_path = cache_dir / key
        if cache_path.is_file() and cache_path.stat().st_size >= 4:
            return np.fromfile(cache_path, dtype=np.float32)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode('utf-8', 'replace')[:400]}")
    pcm = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if cache_dir is not None and len(pcm):
        pcm.tofile(cache_path)
        (cache_dir / (key + ".src")).write_text(str(path), encoding="utf-8")
    return pcm


def gpu_mem() -> dict:
    import torch
    free, total = torch.cuda.mem_get_info()
    return {
        "used_gib": round((total - free) / 1024 ** 3, 2),
        "total_gib": round(total / 1024 ** 3, 2),
        "torch_alloc_gib": round(torch.cuda.memory_allocated() / 1024 ** 3, 2),
        "torch_reserved_gib": round(torch.cuda.memory_reserved() / 1024 ** 3, 2),
    }


def load_model(model_path: str | Path = MODEL_PATH, log=print, device: str = "cuda:0"):
    """AutoModel 加载（dtype=auto, device_map=cuda:0, sdpa）。"""
    activate_vendor()
    import torch
    from transformers import AutoModel

    model_path = str(model_path)
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        dtype="auto", device_map=device, attn_implementation="sdpa",
    ).eval().requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model loaded: {type(model).__name__} params={n_params/1e9:.3f}B gpu={gpu_mem()}")
    return model


def decode_reply(proc, output, prompt_len: int) -> str:
    new_ids = output[0][prompt_len:]
    return proc.decode(new_ids, skip_special_tokens=True).strip()


def load_scene_turns(card_id: str, scenes_dir: Path = SCENES_DIR) -> dict[str, list[dict]]:
    """scene_id -> 该场景的全部 turn（来自 data/scenes/<card_id>.json）。"""
    doc = json.loads((scenes_dir / f"{card_id}.json").read_text(encoding="utf-8"))
    return {s["scene_id"]: s.get("turns", []) for s in doc.get("scenes", [])}


def load_voice_manifest(manifest_path: Path = MANIFEST_PATH) -> dict[tuple[str, str], list[dict]]:
    """(card_id, scene_id) -> 按 idx 排序的语音轮列表。"""
    out: dict[tuple[str, str], list[dict]] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.setdefault((rec["card_id"], rec["scene_id"]), []).append(rec)
    for rows in out.values():
        rows.sort(key=lambda r: int(r.get("idx", 0)))
    return out


def load_selection(selection_path: Path = SELECTION_PATH) -> list[dict]:
    return json.loads(selection_path.read_text(encoding="utf-8"))["items"]
