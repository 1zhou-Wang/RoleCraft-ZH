#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EdgeInstant-Omni-2B load smoke（音频 track 门禁）。

测试：
  1. 手动构造 EdgeInstantProcessor + AutoModel 加载（当前 venv 或 vendor/edgeinstant 5.12.1）
  2. 一条真实用户音频 + 角色扮演 system prompt -> model.generate 中文文本回复
  3. DuplexSession 初始化（warmup 含 TTS）后，用原生 talker/codec 渲染 24kHz 语音

用法：
  # 音频 track 必须用 vendor 里的 transformers 5.12.1
  $env:EDGEINSTANT_VENDOR='1'; python local/edgeinstant/load_smoke.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ei_common as C  # noqa: E402

C.set_utf8_stdio()
DEFAULT_MODEL = str(C.MODEL_PATH)
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "edgeinstant"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--audio", default=str(C.VOICE_DIR / "audio" / "rc01_s01_1.mp3"))
    ap.add_argument("--card-id", default="rc01")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--skip-session", action="store_true", help="跳过 DuplexSession / 原生语音测试")
    ap.add_argument("--render-text", default="你好呀，我是罗小黑，一只猫妖。今天想聊点什么？")
    ap.add_argument("--out-json", default=str(DEFAULT_LOG_DIR / "load_smoke.json"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    report: dict = {
        "model_path": args.model_path,
        "python": sys.version.split()[0],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "vendor_mode": bool(C._vendor_active),
        "phases": {},
        "ok": False,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # vendor 必须在 import transformers 之前生效（否则 5.19 先进 sys.modules）
    C.activate_vendor()
    import numpy as np
    import torch
    import transformers

    report["vendor_mode"] = bool(C._vendor_active)
    report["transformers"] = transformers.__version__
    report["torch"] = torch.__version__
    report["cuda"] = torch.version.cuda
    log(f"python={report['python']} transformers={report['transformers']} torch={report['torch']} cuda={report['cuda']}")
    log(f"gpu before load: {C.gpu_mem()}")

    t0 = time.perf_counter()
    try:
        proc = C.build_processor(args.model_path, log=log)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        report["phases"]["processor"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                         "traceback": traceback.format_exc()[-4000:]}
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    report["phases"]["processor"] = {"ok": True, "s": round(time.perf_counter() - t0, 2),
                                     "class": type(proc).__name__}
    log(f"processor ready ({type(proc).__name__}) {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    try:
        model = C.load_model(args.model_path, log=log)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        report["phases"]["model"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                     "traceback": traceback.format_exc()[-8000:]}
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    report["phases"]["model"] = {"ok": True, "s": round(time.perf_counter() - t0, 2),
                                 "class": type(model).__name__,
                                 "params": sum(p.numel() for p in model.parameters())}

    # ---- 1) 单条音频 -> 文本 ----
    try:
        sys_prompt = C.build_system_prompt(args.card_id)
        wav = C.load_audio_16k(Path(args.audio))
        report["audio"] = {"path": args.audio, "samples": int(len(wav)), "seconds": round(len(wav) / C.SR, 2)}
        log(f"audio {Path(args.audio).name}: {len(wav)} samples ({len(wav)/C.SR:.2f}s)")
        inputs = proc(audio=wav, sampling_rate=C.SR, task="qa", system_prompt=sys_prompt, history=[])
        prompt_len = int(inputs["input_ids"].shape[1])
        device = next(model.parameters()).device
        inputs = inputs.to(device)   # device_map 加载不会自动搬移，官方 API 要求显式 .to(device)
        report["prompt_tokens"] = prompt_len
        log(f"prompt tokens: {prompt_len} (audio_pad={int((inputs['input_ids'][0]==proc.config.audio_pad_id).sum())})")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen_s = time.perf_counter() - t0
        new_ids = out[0][prompt_len:]
        text = C.decode_reply(proc, out, prompt_len)
        report["generate"] = {
            "ok": bool(text), "s": round(gen_s, 2), "new_tokens": int(len(new_ids)), "text": text,
            "tok_per_s": round(len(new_ids) / gen_s, 1) if gen_s > 0 else None,
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
            "gpu_after": C.gpu_mem(),
        }
        log(f"generate {len(new_ids)} tok in {gen_s:.2f}s ({len(new_ids)/gen_s:.1f} tok/s) -> {text!r}")
        log(f"gpu after generate: {C.gpu_mem()}")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        report["generate"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                              "traceback": traceback.format_exc()[-8000:]}
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    # ---- 2) DuplexSession + 原生语音渲染 ----
    if not args.skip_session:
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            t0 = time.perf_counter()
            Session = get_class_from_dynamic_module("duplex_edgeinstant.DuplexSession", args.model_path,
                                                    local_files_only=True)
            session = Session(model, proc, trace=False)
            t_init = time.perf_counter() - t0
            mod = sys.modules[type(session).__module__]
            report["session"] = {"ok": True, "init_s": round(t_init, 2),
                                 "class_module": type(session).__module__, "gpu_after_init": C.gpu_mem()}
            log(f"DuplexSession init {t_init:.2f}s gpu={C.gpu_mem()}")

            def render_full(text: str):
                """整句原生语音：_speak_all 的完整版（_render 会截断到 0.7s，仅适合附和音）。"""
                import re
                language = "chinese" if re.search(r"[一-鿿]", text) else "english"
                body = session.tok.encode(text, add_special_tokens=False)
                rate = session.m.codec.sample_rate
                st = session._new_state(language)
                trim, parts, sent = mod.SilenceTrim(rate), [], 0
                session._read(body, language)
                for j in range(len(body)):
                    st.push_token(j)
                st.end_input()
                while not st.ended and len(st.frames) <= mod.FRAMES_PER_TOKEN * len(body) + mod.FRAMES_SLACK:
                    st.advance(max_new_frames=mod.CHUNK_FRAMES)
                    if len(st.frames) > sent:
                        parts.append(trim.feed(session._decode(st, sent).reshape(-1)))
                        sent = len(st.frames)
                parts.append(trim.end())
                return np.concatenate(parts).astype(np.float32), rate

            t0 = time.perf_counter()
            with torch.inference_mode():
                pcm, rate = render_full(args.render_text)
            render_s = time.perf_counter() - t0
            audio_s = len(pcm) / rate if len(pcm) else 0.0
            out_wav = DEFAULT_LOG_DIR / "smoke_render.wav"
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            if len(pcm):
                import wave as wave_mod
                pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
                with wave_mod.open(str(out_wav), "wb") as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(rate)
                    f.writeframes(pcm16.tobytes())
            report["render"] = {
                "ok": bool(len(pcm)), "text": args.render_text, "rate": rate,
                "seconds": round(audio_s, 2), "render_s": round(render_s, 2),
                "rtf": round(render_s / audio_s, 3) if audio_s else None,
                "wav": str(out_wav) if len(pcm) else None, "samples": int(len(pcm)),
                "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
            }
            log(f"render {len(pcm)} samples @{rate}Hz = {audio_s:.2f}s audio in {render_s:.2f}s "
                f"(rtf={render_s/audio_s if audio_s else float('nan'):.2f}) -> {out_wav}")

            t0 = time.perf_counter()
            with torch.inference_mode():
                clip = session._render("对")
            report["session"]["_render_official"] = {"ok": bool(len(clip)), "samples": int(len(clip)),
                                                     "s": round(time.perf_counter() - t0, 2)}
            log(f"session._render('对') official path -> {len(clip)} samples")
            session.close()
            report["session"]["closed"] = True
            log(f"gpu after session: {C.gpu_mem()}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            report["session"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()[-8000:]}
            report["render"] = {"ok": False, "error": "session init failed"}
    else:
        report["session"] = {"ok": None, "skipped": True}
        report["render"] = {"ok": None, "skipped": True}

    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["ok"] = bool(report.get("generate", {}).get("ok"))
    report["gpu_final"] = C.gpu_mem()
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"WROTE {out_json}")
    log("RESULT " + json.dumps({k: (report.get(k) or {}).get("ok") if isinstance(report.get(k), dict) else None
                                for k in ("phases", "generate", "session", "render")}, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
