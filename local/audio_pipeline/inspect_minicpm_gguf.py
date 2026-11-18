#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 MiniCPM5-2B GGUF 元数据与 chat template（判断 thinking 默认行为）。

用法: python local/audio_pipeline/inspect_minicpm_gguf.py [gguf_path]
输出: 打印 architecture / tokenizer pre / chat_template（含 enable_thinking 相关片段）；存 logs/minicpm5/minicpm_gguf_meta.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLAMA_PY = Path(r"E:\RBM\algorithms\draft\qwen3_5_infra_test\llama_src\llama.cpp-b11552\gguf-py")
sys.path.insert(0, str(LLAMA_PY))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import gguf  # noqa: E402


def main() -> int:
    candidates = [
        ROOT / "work" / "minicpm5" / "MiniCPM5-2B-Q8_0.gguf",
        Path(r"E:\RBM\models\minicpm-gguf\MiniCPM5-2B-Q8_0.gguf"),
    ]
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else next((p for p in candidates if p.is_file()), None)
    if path is None or not path.is_file():
        print("GGUF not found:", candidates)
        return 2
    print(f"GGUF: {path} ({path.stat().st_size/1e9:.2f} GB)")
    reader = gguf.GGUFReader(str(path))
    meta = {k: v for k, v in reader.fields.items() if not k.startswith("GGUF.")}
    arch = None
    template = None
    for f in reader.fields.values():
        if f.name.endswith(".architecture"):
            arch = str(f.parts[-1])
        if f.name.endswith(".chat_template"):
            try:
                template = f.contents()
            except Exception:
                template = str(f.parts)
    print("architecture:", arch)
    print("n_tensors:", len(reader.tensors))
    print("n_kv:", len(meta))
    if template:
        t = template if isinstance(template, str) else str(template)
        print(f"chat_template ({len(t)} chars), 包含 enable_thinking: {'enable_thinking' in t}, "
              f"包含 <think>: {'<think>' in t}, role/vars: "
              f"{[x for x in ('add_generation_prompt', 'think', 'thinking') if x in t]}")
        # 提取模板尾部（生成提示部分）与 thinking 相关片段
        print("---- template tail ----")
        print(t[-1500:])
    else:
        print("chat_template: NOT FOUND")
    out = ROOT / "logs" / "minicpm5" / "minicpm_gguf_meta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "path": str(path), "size": path.stat().st_size, "architecture": arch,
        "n_tensors": len(reader.tensors), "n_kv": len(meta),
        "has_chat_template": bool(template),
        "has_enable_thinking": bool(template and "enable_thinking" in str(template)),
        "template_tail": (str(template)[-3000:] if template else None),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("meta ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
