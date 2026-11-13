#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EdgeInstant-Omni-2B 音频输入 track runner（B3）。

链路：用户轮 TTS 音频（16k 单声道） -> EdgeInstantDuplexForConditionalGeneration
      -> assistant_text（角色扮演 system prompt + 多轮 history）
      ->（可选）原生 talker/codec 语音，24kHz wav

  * 数据：data/voice/selection.json（50 卡 × 10 场景 = 500 场景）、data/voice/manifest.jsonl（784 轮）
  * 输出：baselines/audio_edgeinstant/<card_id>.jsonl，一行一场景，幂等可续跑
  * 原生语音：baselines/audio_edgeinstant/audio/<card_id>/<scene_id>_<idx>.wav

两个阶段（--phase）：
  text   只生成文本回复（快，先交付）
  render 只对已有记录渲染原生语音（需要 DuplexSession warmup ~1-2min）
  all    先 text 再 render（默认）

用法：
  $env:EDGEINSTANT_VENDOR='1'
  python local/edgeinstant/run_edgeinstant.py --phase text
  python local/edgeinstant/run_edgeinstant.py --phase render
  python local/edgeinstant/run_edgeinstant.py --phase all
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ei_common as C  # noqa: E402

C.set_utf8_stdio()

DEFAULT_OUT_DIR = PROJECT_ROOT / "baselines" / "audio_edgeinstant"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "edgeinstant"
PIPELINE = "audio_edgeinstant"       # SCHEMA.md 音频输入 track 的链路 slug
MODEL_SLUG = "edgeinstant-omni-2b"

log = logging.getLogger("edgeinstant_audio")


# --------------------------------------------------------------------------- #
# native speech (talker/codec) helpers
# --------------------------------------------------------------------------- #
def make_session(model, proc, model_path: str, log=log):
    """初始化 DuplexSession（含 warmup + TTS CUDA graph），返回 (session, duplex_module)。"""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    session_cls = get_class_from_dynamic_module("duplex_edgeinstant.DuplexSession", model_path,
                                                local_files_only=True)
    t0 = time.perf_counter()
    session = session_cls(model, proc, trace=False)
    mod = sys.modules[type(session).__module__]
    log.info("DuplexSession init %.1fs gpu=%s", time.perf_counter() - t0, C.gpu_mem())
    return session, mod


def render_full(session, mod, text: str, seed: int | None = None):
    """整句原生语音（_speak_all 的完整版；官方 _render 会截断到 0.7s，仅用于附和音）。"""
    import numpy as np
    import torch

    language = "chinese" if re.search(r"[一-鿿]", text) else "english"
    body = session.tok.encode(text, add_special_tokens=False)
    rate = session.m.codec.sample_rate
    if seed is not None:
        torch.manual_seed(seed)
    st = session._new_state(language)
    trim, parts, sent = mod.SilenceTrim(rate), [], 0
    session._read(body, language)
    for j in range(len(body)):
        st.push_token(j)
    st.end_input()
    with torch.inference_mode():
        while not st.ended and len(st.frames) <= mod.FRAMES_PER_TOKEN * len(body) + mod.FRAMES_SLACK:
            st.advance(max_new_frames=mod.CHUNK_FRAMES)
            if len(st.frames) > sent:
                parts.append(trim.feed(session._decode(st, sent).reshape(-1)))
                sent = len(st.frames)
        parts.append(trim.end())
    if not parts:
        return np.zeros(0, np.float32), rate
    return np.concatenate(parts).astype(np.float32), rate


def save_wav(path: Path, pcm, rate: int) -> None:
    import wave as wave_mod
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16)
    tmp = path.with_suffix(".wav.tmp")
    with wave_mod.open(str(tmp), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm16.tobytes())
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #
def read_existing(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s:%d 无法解析，跳过", path, lineno)
                continue
            if rec.get("scene_id"):
                records[rec["scene_id"]] = rec
    return records


def write_records(path: Path, ordered: list[str], records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for sid in ordered:
            if sid in records:
                f.write(json.dumps(records[sid], ensure_ascii=False) + "\n")
    tmp.replace(path)


def scene_is_complete(rec: dict | None) -> bool:
    if not rec or rec.get("error"):
        return False
    turns = rec.get("turns") or []
    return bool(turns) and all(t.get("assistant_text") for t in turns)


def scene_audio_complete(rec: dict | None) -> bool:
    if not scene_is_complete(rec):
        return False
    return all(t.get("assistant_audio") for t in (rec.get("turns") or []))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EdgeInstant-Omni-2B 音频输入 track runner")
    ap.add_argument("--model-path", default=str(C.MODEL_PATH))
    ap.add_argument("--characters-dir", default=str(C.CHARACTERS_DIR))
    ap.add_argument("--scenes-dir", default=str(C.SCENES_DIR))
    ap.add_argument("--selection", default=str(C.SELECTION_PATH))
    ap.add_argument("--manifest", default=str(C.MANIFEST_PATH))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--cards", default=None, help="只跑指定卡，逗号分隔")
    ap.add_argument("--limit-cards", type=int, default=0, help="只跑前 N 张卡（0=全部）")
    ap.add_argument("--limit-scenes", type=int, default=0, help="每卡只跑前 N 个场景（0=全部 10 个）")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--phase", default="all", choices=["text", "render", "all"])
    ap.add_argument("--no-native-audio", dest="native_audio", action="store_false", help="不做原生语音渲染")
    ap.add_argument("--no-retry-errors", dest="retry_errors", action="store_false")
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.log_file is None:
        args.log_file = str(DEFAULT_LOG_DIR / f"audio_run_{datetime.now():%Y%m%d_%H%M%S}.log")
    return args


def setup_logging(args: argparse.Namespace) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        handlers=handlers, force=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args)

    characters_dir = Path(args.characters_dir)
    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = C.load_selection(Path(args.selection))
    manifest = C.load_voice_manifest(Path(args.manifest))
    if args.cards:
        wanted = {c.strip() for c in args.cards.split(",") if c.strip()}
        selection = [it for it in selection if it["card_id"] in wanted]
    if args.limit_cards and args.limit_cards > 0:
        selection = selection[: args.limit_cards]
    if not selection:
        log.error("selection 为空: %s", args.selection)
        return 2

    plan: list[tuple[str, list[str]]] = []
    for it in selection:
        cid = it["card_id"]
        sids = list(it["scene_ids"])
        if args.limit_scenes and args.limit_scenes > 0:
            sids = sids[: args.limit_scenes]
        missing = [s for s in sids if (cid, s) not in manifest]
        if missing:
            log.warning("[%s] manifest 缺 %d 个场景，跳过: %s", cid, len(missing), missing[:3])
            sids = [s for s in sids if (cid, s) in manifest]
        plan.append((cid, sids))

    total_scenes = sum(len(s) for _, s in plan)
    total_turns = sum(len(manifest[(cid, sid)]) for cid, sids in plan for sid in sids)
    log.info("计划: 卡 %d 张 | 场景 %d 个 | 轮 %d | phase=%s native_audio=%s max_new_tokens=%d | 输出 %s",
             len(plan), total_scenes, total_turns, args.phase, args.native_audio,
             args.max_new_tokens, out_dir)
    if args.dry_run:
        for cid, sids in plan:
            log.info("[dry-run] %s: %d 场景 -> %s", cid, len(sids), out_dir / f"{cid}.jsonl")
        return 0

    existing_by_card = {cid: read_existing(out_dir / f"{cid}.jsonl") for cid, _ in plan}

    do_text = args.phase in ("text", "all")
    do_render = args.phase in ("render", "all") and args.native_audio
    text_todo = sum(1 for cid, sids in plan for sid in sids if not scene_is_complete(existing_by_card[cid].get(sid)))
    render_todo = sum(1 for cid, sids in plan for sid in sids
                      if scene_is_complete(existing_by_card[cid].get(sid))
                      and not scene_audio_complete(existing_by_card[cid].get(sid)))
    log.info("待跑: text 场景 %d | 待渲染语音场景 %d", text_todo, render_todo)
    if (not do_text or text_todo == 0) and (not do_render or render_todo == 0):
        log.info("没有待办工作，退出。")
        return 0

    C.activate_vendor()
    import torch
    import transformers
    log.info("transformers=%s torch=%s vendor_mode=%s", transformers.__version__, torch.__version__, C._vendor_active)

    t_load = time.perf_counter()
    proc = C.build_processor(args.model_path, log=log.info)
    model = C.load_model(args.model_path, log=log.info)
    device = next(model.parameters()).device
    log.info("模型加载完成 %.1fs gpu=%s device=%s", time.perf_counter() - t_load, C.gpu_mem(), device)

    started = time.perf_counter()
    stats = {
        "scenes_text_ok": 0, "scenes_text_failed": 0, "turns_ok": 0, "turns_failed": 0,
        "scenes_audio_ok": 0, "turns_audio_ok": 0, "turns_audio_failed": 0,
        "turn_latencies": [], "render_latencies": [], "audio_seconds": [],
    }
    cards_errors: list[dict] = []

    # ---------------- phase 1: text ----------------
    todo_turns_planned = sum(len(manifest[(cid, sid)]) for cid, sids in plan for sid in sids
                             if not scene_is_complete(existing_by_card[cid].get(sid)))
    todo_turns_done = 0
    if do_text and text_todo:
        log.info("== phase text 开始 ==")
        for ci, (cid, scene_ids) in enumerate(plan):
            records = existing_by_card[cid]
            ordered = scene_ids
            try:
                sys_prompt = C.build_system_prompt(cid, characters_dir)
            except Exception as exc:  # noqa: BLE001
                log.exception("[%s] 角色卡读取失败", cid)
                cards_errors.append({"card_id": cid, "error": str(exc)})
                continue
            card_t0 = time.perf_counter()
            for sid in scene_ids:
                rec = records.get(sid)
                if scene_is_complete(rec):
                    continue
                turns = manifest[(cid, sid)]
                history: list[dict] = []
                turns_out: list[dict] = []
                scene_error: str | None = None
                scene_t0 = time.perf_counter()
                for turn in turns:
                    idx = int(turn.get("idx", len(turns_out) + 1))
                    user_text = str(turn.get("user_text", ""))
                    try:
                        wav = C.load_audio_16k(PROJECT_ROOT / turn["file"], cache_dir=C.AUDIO_CACHE_DIR)
                        inputs = proc(audio=wav, sampling_rate=C.SR, task="qa",
                                      system_prompt=sys_prompt, history=history)
                        prompt_len = int(inputs["input_ids"].shape[1])
                        inputs = inputs.to(device)
                        t0 = time.perf_counter()
                        with torch.inference_mode():
                            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                        latency = time.perf_counter() - t0
                        reply = C.decode_reply(proc, out, prompt_len)
                        if not reply:
                            raise RuntimeError("empty reply")
                        history.append({"role": "user", "content": user_text})
                        history.append({"role": "assistant", "content": reply})
                        turns_out.append({
                            "idx": idx,
                            "user_audio": turn["file"],
                            "user_text": user_text,
                            "user_text_gt": user_text,
                            "user_asr_text": None,
                            "assistant_text": reply,
                            "latency_s": round(latency, 3),
                            "user_audio_seconds": turn.get("duration"),
                            "prompt_tokens": prompt_len,
                        })
                        stats["turn_latencies"].append(latency)
                        stats["turns_ok"] += 1
                        todo_turns_done += 1
                    except Exception as exc:  # noqa: BLE001
                        scene_error = f"turn {idx} failed: {type(exc).__name__}: {exc}"
                        log.error("[%s] %s %s", cid, sid, scene_error)
                        turns_out.append({"idx": idx, "user_audio": turn.get("file"),
                                          "user_text": user_text, "assistant_text": "",
                                          "latency_s": None, "error": str(exc)})
                        stats["turns_failed"] += 1
                        break
                scene_latency = round(time.perf_counter() - scene_t0, 3)
                record = {
                    "card_id": cid,
                    "scene_id": sid,
                    "pipeline": PIPELINE,
                    "model": MODEL_SLUG,
                    "chain": "edgeinstant-omni-2b-native-audio-in",
                    "mode": "audio-input",
                    "input_modality": "audio",
                    "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                    "turns": turns_out,
                    "latency_s": scene_latency,
                    "native_audio": bool(args.native_audio),
                }
                if scene_error:
                    record["error"] = scene_error
                    stats["scenes_text_failed"] += 1
                else:
                    stats["scenes_text_ok"] += 1
                records[sid] = record
                write_records(out_dir / f"{cid}.jsonl", ordered, records)

            if (ci + 1) % 1 == 0 or ci == len(plan) - 1:
                elapsed = time.perf_counter() - started
                rate = todo_turns_done / elapsed if elapsed > 0 else 0.0
                remaining = max(0, todo_turns_planned - todo_turns_done)
                avg = (sum(stats["turn_latencies"]) / len(stats["turn_latencies"])) if stats["turn_latencies"] else 0
                log.info("[text %d/%d] %s: 本轮 %.1fs | 累计 场景 ok=%d fail=%d 轮 ok=%d fail=%d | "
                         "均延迟 %.2fs | %.1f 轮/min | 剩余 %d 轮 ETA %.1f min",
                         ci + 1, len(plan), cid, time.perf_counter() - card_t0,
                         stats["scenes_text_ok"], stats["scenes_text_failed"],
                         stats["turns_ok"], stats["turns_failed"], avg, rate * 60,
                         remaining, (remaining / rate / 60) if rate > 0 else float("inf"))
        text_elapsed = time.perf_counter() - started
        log.info("== phase text 完成 %.1fs | 场景 ok=%d fail=%d | 轮 ok=%d fail=%d | 均延迟 %.2fs",
                 text_elapsed, stats["scenes_text_ok"], stats["scenes_text_failed"],
                 stats["turns_ok"], stats["turns_failed"],
                 (sum(stats["turn_latencies"]) / len(stats["turn_latencies"])) if stats["turn_latencies"] else 0)
    else:
        log.info("跳过 phase text（已完成或未请求）")

    # ---------------- phase 2: native audio render ----------------
    render_elapsed = 0.0
    session = None
    if do_render:
        # 重新读一遍（text phase 可能刚写入）
        existing_by_card = {cid: read_existing(out_dir / f"{cid}.jsonl") for cid, _ in plan}
        render_todo = sum(1 for cid, sids in plan for sid in sids
                          if scene_is_complete(existing_by_card[cid].get(sid))
                          and not scene_audio_complete(existing_by_card[cid].get(sid)))
        if render_todo == 0:
            log.info("没有待渲染语音的场景。")
        else:
            log.info("== phase render 开始（待渲染场景 %d）==", render_todo)
            render_t0 = time.perf_counter()
            try:
                session, mod = make_session(model, proc, args.model_path, log=log)
                for ci, (cid, scene_ids) in enumerate(plan):
                    records = existing_by_card[cid]
                    card_t0 = time.perf_counter()
                    for sid in scene_ids:
                        rec = records.get(sid)
                        if not scene_is_complete(rec) or scene_audio_complete(rec):
                            continue
                        for turn in rec.get("turns") or []:
                            text = (turn.get("assistant_text") or "").strip()
                            if not text:
                                continue
                            idx = int(turn.get("idx", 0))
                            wav_path = audio_dir / f"{sid}_{idx}.wav"
                            try:
                                t0 = time.perf_counter()
                                pcm, rate = render_full(session, mod, text, seed=idx)
                                render_s = time.perf_counter() - t0
                                if len(pcm) == 0:
                                    raise RuntimeError("empty pcm")
                                save_wav(wav_path, pcm, rate)
                                audio_s = len(pcm) / rate
                                turn["assistant_audio"] = str(wav_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                                turn["assistant_audio_seconds"] = round(audio_s, 3)
                                turn["audio_render_s"] = round(render_s, 3)
                                turn["audio_sample_rate"] = rate
                                turn["audio_source"] = "edgeinstant-native"
                                stats["turns_audio_ok"] += 1
                                stats["render_latencies"].append(render_s)
                                stats["audio_seconds"].append(audio_s)
                            except Exception as exc:  # noqa: BLE001
                                log.error("[%s] %s turn %s 原生语音失败: %s: %s", cid, sid, idx,
                                          type(exc).__name__, exc)
                                turn["audio_error"] = f"{type(exc).__name__}: {exc}"
                                stats["turns_audio_failed"] += 1
                        if scene_audio_complete(rec):
                            stats["scenes_audio_ok"] += 1
                        write_records(out_dir / f"{cid}.jsonl", scene_ids, records)
                    rendered = sum(1 for s in scene_ids if scene_audio_complete(records.get(s)))
                    log.info("[render %d/%d] %s: 本卡已渲染场景 %d/%d | 累计场景 %d | 轮次 ok=%d fail=%d | "
                             "均渲染 %.2fs | 用时 %.1f min",
                             ci + 1, len(plan), cid, rendered, len(scene_ids),
                             stats["scenes_audio_ok"], stats["turns_audio_ok"], stats["turns_audio_failed"],
                             (sum(stats["render_latencies"]) / len(stats["render_latencies"]))
                             if stats["render_latencies"] else 0, (time.perf_counter() - render_t0) / 60)
            except Exception as exc:  # noqa: BLE001
                log.exception("原生语音 session 初始化/渲染失败，音频 phase 中止（文本结果已保留）")
                cards_errors.append({"card_id": "*", "error": f"native audio: {type(exc).__name__}: {exc}"})
            finally:
                if session is not None:
                    try:
                        session.close()
                    except Exception:  # noqa: BLE001
                        pass
            render_elapsed = time.perf_counter() - render_t0
            log.info("== phase render 完成 %.1fs | 场景音频 ok=%d | 轮音频 ok=%d fail=%d | 音频总时长 %.1fs",
                     render_elapsed, stats["scenes_audio_ok"], stats["turns_audio_ok"],
                     stats["turns_audio_failed"], sum(stats["audio_seconds"]))

    # ---------------- summary ----------------
    total_elapsed = time.perf_counter() - started

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        vals = sorted(values)
        return {"mean": round(sum(vals) / len(vals), 3),
                "p50": round(vals[int(round(0.5 * (len(vals) - 1)))], 3),
                "p95": round(vals[int(round(0.95 * (len(vals) - 1)))], 3),
                "max": round(vals[-1], 3)}

    # 全量最终统计（包含历史已完成的记录）
    final_scenes_ok = final_scenes_failed = final_turns = final_turns_failed = final_audio_turns = 0
    for cid, sids in plan:
        recs = read_existing(out_dir / f"{cid}.jsonl")
        for sid in sids:
            rec = recs.get(sid)
            if not rec:
                continue
            if rec.get("error"):
                final_scenes_failed += 1
            else:
                final_scenes_ok += 1
            for t in rec.get("turns") or []:
                if t.get("assistant_text"):
                    final_turns += 1
                elif t.get("error"):
                    final_turns_failed += 1
                if t.get("assistant_audio"):
                    final_audio_turns += 1

    manifest_out = {
        "pipeline": PIPELINE,
        "model": MODEL_SLUG,
        "mode": "audio-input",
        "model_path": args.model_path,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vendor_mode": bool(C._vendor_active),
        "cards": len(plan),
        "scenes_total": total_scenes,
        "turns_total": total_turns,
        "phase": args.phase,
        "native_audio_requested": bool(args.native_audio),
        "scenes_ok": final_scenes_ok,
        "scenes_failed": final_scenes_failed,
        "turns_ok": final_turns,
        "turns_failed": final_turns_failed,
        "turns_with_native_audio": final_audio_turns,
        "elapsed_s": round(total_elapsed, 2),
        "text_elapsed_s": round(text_elapsed, 2) if do_text and text_todo else 0.0,
        "render_elapsed_s": round(render_elapsed, 2),
        "turn_latency_s": _stats(stats["turn_latencies"]),
        "render_latency_s": _stats(stats["render_latencies"]),
        "audio_seconds_total": round(sum(stats["audio_seconds"]), 2),
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "system_prompt_mode": "roleplay-template (same as text track)",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "cards_errors": cards_errors,
        "log_file": args.log_file,
    }
    mpath = out_dir / "_manifest.json"
    mpath.write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.path.insert(0, str(DEFAULT_LOG_DIR))
    (DEFAULT_LOG_DIR / "audio_manifest.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("完成: 场景 ok=%d fail=%d | 轮 ok=%d fail=%d | 原生语音轮 %d | 总用时 %.1fs | "
             "文本延迟 mean=%.2fs p95=%.2fs | manifest: %s",
             final_scenes_ok, final_scenes_failed, final_turns, final_turns_failed, final_audio_turns,
             total_elapsed, manifest_out["turn_latency_s"]["mean"], manifest_out["turn_latency_s"]["p95"], mpath)

    del model
    torch.cuda.empty_cache()
    return 0 if final_turns_failed == 0 and final_scenes_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
