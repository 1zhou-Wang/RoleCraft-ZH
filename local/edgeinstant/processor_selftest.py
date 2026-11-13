#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构造 EdgeInstantProcessor（绕开 AutoImageProcessor 的 PIL/torchvision 依赖）+ 输入接口自检。

背景：AutoProcessor.from_pretrained 会因为 processor_config.json 里 image_processor_subfolder
而去加载图像处理器；音频/文本 track 都不需要它。
本脚本手动构造 EdgeInstantProcessor(tokenizer, feature_extractor, config, image_processor=None)，
并在两种 transformers 版本（venv 5.19 / vendor 5.12.1）下验证：
  * proc(audio=..., task='qa', system_prompt=..., history=...) 能产出 input_features
  * proc(text=..., system_prompt=..., history=...) 能产出 input_ids（无 input_features）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path(r"E:\RBM\models\EdgeInstant-Omni-2B-1106-Preview")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_processor(model_path: str, log=print):
    """手动构造 EdgeInstantProcessor，不触发图像处理器加载。"""
    from transformers import AutoTokenizer, WhisperFeatureExtractor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    proc_cls = get_class_from_dynamic_module(
        "processing_edgeinstant.EdgeInstantProcessor", model_path, local_files_only=True)
    cfg_cls = get_class_from_dynamic_module(
        "configuration_edgeinstant.EdgeInstantConfig", model_path, local_files_only=True)
    config = cfg_cls.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, config=config.thinker_config, trust_remote_code=True, local_files_only=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    log(f"config={type(config).__name__} tokenizer={type(tokenizer).__name__} "
        f"feature_extractor={type(feature_extractor).__name__}")
    return proc_cls(tokenizer=tokenizer, feature_extractor=feature_extractor,
                    config=config, image_processor=None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL))
    args = ap.parse_args()

    import numpy as np
    import transformers
    print(f"transformers={transformers.__version__} file={transformers.__file__}", flush=True)
    proc = build_processor(args.model_path)

    sys_prompt = "你将扮演「测试角色」。以下是人设卡：\n这是一个自检用的人设卡。\n\n规则：始终以第一人称回复。"
    history = [{"role": "user", "content": "你好。"}, {"role": "assistant", "content": "你好呀。"}]

    # 文本分支
    text_in = proc(text="今天天气怎么样？", system_prompt=sys_prompt, history=history, enable_thinking=False)
    print("TEXT keys:", sorted(text_in.keys()), "input_ids", tuple(text_in["input_ids"].shape), flush=True)
    print("TEXT has input_features:", "input_features" in text_in, flush=True)
    toks = proc.tokenizer.convert_ids_to_tokens(text_in["input_ids"][0][:40].tolist())
    print("TEXT first tokens:", toks, flush=True)

    # 音频分支
    wav = (np.random.default_rng(0).standard_normal(16000 * 2) * 0.05).astype(np.float32)
    audio_in = proc(audio=wav, sampling_rate=16000, task="qa", system_prompt=sys_prompt, history=history)
    print("AUDIO keys:", sorted(audio_in.keys()), "input_ids", tuple(audio_in["input_ids"].shape),
          "input_features", tuple(audio_in["input_features"].shape), flush=True)
    print("AUDIO token count:", int((audio_in["input_ids"][0] == proc.config.audio_pad_id).sum()), flush=True)

    # ASR 分支（无 system prompt）
    asr_in = proc(audio=wav, sampling_rate=16000, task="asr")
    print("ASR keys:", sorted(asr_in.keys()), "input_ids", tuple(asr_in["input_ids"].shape), flush=True)
    print("SELFTEST_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
