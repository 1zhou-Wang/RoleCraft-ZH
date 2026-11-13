# -*- coding: utf-8 -*-
"""API smoke test for RoleCraft-ZH data pipeline: DeepSeek chat/search/fetch, DashScope omni, Qwen TTS."""
import json, os, sys, time, base64
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]

def load_env():
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env

ENV = load_env()

def ok(label, **kw):
    print(f"[OK] {label}", json.dumps(kw, ensure_ascii=False)[:600])

def fail(label, e):
    print(f"[FAIL] {label}: {e}")

# ---------- 1. DeepSeek chat ----------
def test_ds_chat():
    from openai import OpenAI
    c = OpenAI(api_key=ENV["DEEPSEEK_API_KEY"], base_url=ENV["DEEPSEEK_BASE"])
    t0 = time.time()
    r = c.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "system", "content": "只输出 json。"},
                  {"role": "user", "content": '返回 json：{"ok": true, "msg": "你好"}'}],
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=100,
    )
    dt = time.time() - t0
    ok("ds_chat", content=r.choices[0].message.content, sec=round(dt, 2), usage=str(r.usage))

# ---------- 2. DeepSeek search ----------
def test_ds_search():
    r = requests.post(f"{ENV['DEEPSEEK_BASE']}/v1/tools/search",
                      headers={"Authorization": f"Bearer {ENV['DEEPSEEK_API_KEY']}"},
                      json={"query": "罗小黑 角色介绍 性格 经历", "count": 5, "need_content": False}, timeout=60)
    r.raise_for_status()
    d = r.json()
    ok("ds_search", n=len(d.get("results", [])),
       first={k: d["results"][0].get(k) for k in ("title", "url", "rerank_score", "authority", "site_name")} if d.get("results") else None)
    return d.get("results", [])

# ---------- 3. DeepSeek fetch ----------
def test_ds_fetch(url):
    r = requests.post(f"{ENV['DEEPSEEK_BASE']}/v1/tools/fetch",
                      headers={"Authorization": f"Bearer {ENV['DEEPSEEK_API_KEY']}"},
                      json={"url": url}, timeout=120)
    r.raise_for_status()
    d = r.json()
    ok("ds_fetch", title=d.get("title"), chars=len(d.get("content") or ""), head=(d.get("content") or "")[:120])

# ---------- 4. DashScope omni (text) ----------
def test_omni_text():
    from openai import OpenAI
    c = OpenAI(api_key=ENV["DASHSCOPE_API_KEY"], base_url=ENV["DASHSCOPE_OPENAI_BASE"])
    r = c.chat.completions.create(
        model="qwen3.8-omni-flash",
        messages=[{"role": "user", "content": "这是一次连通性测试，请只回复：OK"}],
        extra_body={"reasoning_effort": "none"},
        max_tokens=20, timeout=120,
    )
    ok("omni_text", content=r.choices[0].message.content, usage=str(r.usage))

# ---------- 5. Qwen TTS ----------
TTS_PATHS = [
    ENV["DASHSCOPE_WS_BASE"] + "/services/audio/tts/SpeechSynthesizer",
    ENV["DASHSCOPE_NATIVE_BASE"] + "/services/audio/tts/SpeechSynthesizer",
]
def test_tts():
    body = {"model": "qwen-audio-3.1-tts-next",
            "input": {"text_prompt": "一位二十多岁的年轻男性，声音清亮、语速自然，用好奇的语气问道：“你是谁？”",
                      "format": "wav", "sample_rate": 24000, "channels": 1}}
    for ep in TTS_PATHS:
        try:
            r = requests.post(ep, headers={"Authorization": f"Bearer {ENV['DASHSCOPE_API_KEY']}",
                                           "Content-Type": "application/json"}, json=body, timeout=180)
            print("  TTS endpoint:", ep, "->", r.status_code)
            if r.status_code == 200:
                d = r.json()
                audio = d.get("output", {}).get("audio", {})
                url = audio.get("url")
                ok("tts", endpoint=ep.split('/api/')[0], duration=audio.get("duration"), url_ok=bool(url),
                   request_id=d.get("request_id"))
                if url:
                    out = ROOT / "work" / "smoke_tts.wav"
                    out.parent.mkdir(exist_ok=True)
                    a = requests.get(url, timeout=120)
                    out.write_bytes(a.content)
                    ok("tts_audio_saved", path=str(out), bytes=len(a.content))
                return True
            else:
                print("   body:", r.text[:300])
        except Exception as e:
            print("  TTS exception:", ep, e)
    return False

if __name__ == "__main__":
    print("== DeepSeek chat =="); test_ds_chat()
    print("== DeepSeek search ==")
    try:
        res = test_ds_search()
    except Exception as e:
        fail("ds_search", e); res = []
    print("== DeepSeek fetch ==")
    if res:
        try: test_ds_fetch(res[0]["url"])
        except Exception as e: fail("ds_fetch", e)
    print("== DashScope omni ==")
    try: test_omni_text()
    except Exception as e: fail("omni_text", e)
    print("== Qwen TTS ==")
    try: test_tts()
    except Exception as e: fail("tts", e)
