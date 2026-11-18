#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MiniCPM5-2B 双 server 验证：thinking 泄漏 / 采样生效 / greedy vs official / seed 可复现。

服务: greedy http://127.0.0.1:8090（请求 temperature=0）
      official http://127.0.0.1:8091（请求 temperature=1.0，server 端 flag）
输出: logs/minicpm5/server_probe.json

用法: python local/audio_pipeline/minicpm_probe.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from run_baseline import SYSTEM_PROMPT_TEMPLATE, build_system_prompt, http_json  # noqa: E402

LOG_DIR = ROOT / "logs" / "minicpm5"
GREEDY = "http://127.0.0.1:8090"
OFFICIAL = "http://127.0.0.1:8091"


def call(base: str, messages, temperature: float, max_tokens: int = 512) -> dict:
    payload = {"model": "minicpm5-2b", "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, "stream": False}
    data = http_json(f"{base}/v1/chat/completions", payload, timeout=180.0)
    msg = data["choices"][0].get("message") or {}
    return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content"),
            "finish_reason": data["choices"][0].get("finish_reason"),
            "completion_tokens": (data.get("usage") or {}).get("completion_tokens")}


def main() -> int:
    card = json.loads((ROOT / "data" / "characters" / "rc01.json").read_text(encoding="utf-8"))
    scene_doc = json.loads((ROOT / "data" / "scenes" / "rc01.json").read_text(encoding="utf-8"))
    user = scene_doc["scenes"][0]["turns"][0]["user_text"]
    system_prompt = build_system_prompt(card)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]

    report = {"user": user, "servers": {}, "checks": {}}
    for name, base, temp in [("greedy", GREEDY, 0.0), ("official", OFFICIAL, 1.0)]:
        try:
            props = http_json(f"{base}/props", timeout=10.0)
            default = (props.get("default_generation_settings") or {}).get("params") or {}
            r1 = call(base, messages, temp)
            r2 = call(base, messages, temp)
            entry = {"base_url": base, "request_temperature": temp,
                     "default_sampling": {k: default.get(k) for k in
                                          ("seed", "temperature", "top_k", "top_p", "min_p",
                                           "presence_penalty", "repeat_penalty")},
                     "run1": r1, "run2": r2,
                     "deterministic_same_request": r1["content"] == r2["content"],
                     "has_think_tag": ("<think>" in r1["content"]) or ("</think>" in r1["content"]),
                     "content_empty": not r1["content"].strip(),
                     "reasoning_len": len(str(r1["reasoning"] or "")),
                     "reply_chars": len(r1["content"])}
            report["servers"][name] = entry
            print(f"[{name}] {base} temp={temp} deterministic={entry['deterministic_same_request']} "
                  f"think_tag={entry['has_think_tag']} empty={entry['content_empty']} "
                  f"reasoning_len={entry['reasoning_len']} chars={entry['reply_chars']}")
            print(f"  reply: {r1['content'][:160]}")
            print(f"  default sampling: {entry['default_sampling']}")
        except Exception as exc:  # noqa: BLE001
            report["servers"][name] = {"error": str(exc)}
            print(f"[{name}] ERROR: {exc}")

    g = report["servers"].get("greedy") or {}
    o = report["servers"].get("official") or {}
    report["checks"] = {
        "greedy_deterministic": g.get("deterministic_same_request"),
        "official_deterministic_with_fixed_seed": o.get("deterministic_same_request"),
        "greedy_vs_official_different": (g.get("run1", {}).get("content") != o.get("run1", {}).get("content")),
        "greedy_no_think_tag": not g.get("has_think_tag", True),
        "official_no_think_tag": not o.get("has_think_tag", True),
        "both_non_empty": (not g.get("content_empty", True)) and (not o.get("content_empty", True)),
        "official_min_p_zero": (o.get("default_sampling") or {}).get("min_p") == 0.0,
        "official_temp_one": (o.get("default_sampling") or {}).get("temperature") == 1.0,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "server_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nchecks:", json.dumps(report["checks"], ensure_ascii=False, indent=1))
    print("report ->", LOG_DIR / "server_probe.json")
    ok = all(v for k, v in report["checks"].items() if v is not None)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
