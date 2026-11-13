#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EdgeInstant-Omni-2B 纯文字 track runner（音频输入 track 的对照，用同一底座模型）。

与 Qwen3.5-2B/4B 文字 track 基线严格对齐：
  * system prompt：文字 track 的角色扮演模板（含 persona_card）
  * 逐卡逐场景逐轮，场景内 history 回填前序 user/assistant 文本
  * greedy（do_sample=False，temperature=0），max_new_tokens=512
  * 输出 baselines/edgeinstant-omni-2b-text/<card_id>.jsonl，一行一场景，可幂等续跑

文本接口（processor 的 text 分支，不碰 audio tower，故 transformers 5.19 也可直接跑）：
  inputs = proc(text=user_text, system_prompt=sys_prompt, history=history, enable_thinking=False)
  out = model.generate(**inputs, max_new_tokens=512, do_sample=False)

用法：
  python local/edgeinstant/run_edgeinstant_text.py --limit-cards 1 --limit-scenes 5   # 测速
  python local/edgeinstant/run_edgeinstant_text.py                                    # 全量
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ei_common as C  # noqa: E402

C.set_utf8_stdio()

DEFAULT_OUT_DIR = PROJECT_ROOT / "baselines" / "edgeinstant-omni-2b-text"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "edgeinstant"
PIPELINE = "edgeinstant-omni-2b-text"
MODEL_SLUG = "edgeinstant-omni-2b"

log = logging.getLogger("edgeinstant_text")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EdgeInstant-Omni-2B 纯文字 track runner")
    ap.add_argument("--model-path", default=str(C.MODEL_PATH))
    ap.add_argument("--characters-dir", default=str(C.CHARACTERS_DIR))
    ap.add_argument("--scenes-dir", default=str(C.SCENES_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--cards", default=None, help="只跑指定卡，逗号分隔")
    ap.add_argument("--limit-cards", type=int, default=0, help="只跑前 N 张卡（0=全部）")
    ap.add_argument("--limit-scenes", type=int, default=0, help="每卡只跑前 N 个场景（0=全部）")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--no-retry-errors", dest="retry_errors", action="store_false", help="续跑时不重跑上次失败的场景")
    ap.add_argument("--progress-every", type=int, default=1, help="每 N 张卡打印一次进度")
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.log_file is None:
        args.log_file = str(DEFAULT_LOG_DIR / f"text_run_{datetime.now():%Y%m%d_%H%M%S}.log")
    return args


def setup_logging(args: argparse.Namespace) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        handlers=handlers, force=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args)

    characters_dir, scenes_dir, out_dir = Path(args.characters_dir), Path(args.scenes_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    card_paths = sorted(characters_dir.glob("*.json"))
    if args.cards:
        wanted = {c.strip() for c in args.cards.split(",") if c.strip()}
        card_paths = [p for p in card_paths if p.stem in wanted]
    if args.limit_cards and args.limit_cards > 0:
        card_paths = card_paths[: args.limit_cards]
    if not card_paths:
        log.error("没有可用角色卡: %s", characters_dir)
        return 2

    # 计划：只保留 scenes 就绪的卡
    plan: list[tuple[str, list[str]]] = []
    turns_count: dict[tuple[str, str], int] = {}
    total_scenes = total_turns_planned = 0
    for p in card_paths:
        scenes_path = scenes_dir / f"{p.stem}.json"
        if not scenes_path.is_file():
            log.warning("跳过 %s：scenes 文件不存在", p.stem)
            continue
        scenes = load_json(scenes_path).get("scenes") or []
        if args.limit_scenes and args.limit_scenes > 0:
            scenes = scenes[: args.limit_scenes]
        plan.append((p.stem, [s["scene_id"] for s in scenes]))
        total_scenes += len(scenes)
        for s in scenes:
            n = len(s.get("turns") or [])
            turns_count[(p.stem, s["scene_id"])] = n
            total_turns_planned += n
    log.info("计划: 卡 %d 张 | 场景 %d 个 | 轮次 %d | max_new_tokens=%d greedy | 输出 %s",
             len(plan), total_scenes, total_turns_planned, args.max_new_tokens, out_dir)
    if args.dry_run:
        for cid, sids in plan:
            log.info("[dry-run] %s -> %s (%d 场景)", cid, out_dir / f"{cid}.jsonl", len(sids))
        return 0

    # 已有进度
    existing_by_card = {cid: read_existing(out_dir / f"{cid}.jsonl") for cid, _ in plan}
    todo_scenes = sum(1 for cid, sids in plan for sid in sids
                      if sid not in existing_by_card[cid]
                      or ("error" in existing_by_card[cid][sid] and args.retry_errors))
    todo_turns_planned = sum(turns_count[(cid, sid)] for cid, sids in plan for sid in sids
                             if sid not in existing_by_card[cid]
                             or ("error" in existing_by_card[cid][sid] and args.retry_errors))
    log.info("已完成场景 %d / %d，待跑 %d", total_scenes - todo_scenes, total_scenes, todo_scenes)
    if todo_scenes == 0:
        log.info("全部场景已完成，无需运行。")
        return 0

    import torch
    C.activate_vendor()
    import transformers
    log.info("transformers=%s torch=%s", transformers.__version__, torch.__version__)

    t_load = time.perf_counter()
    proc = C.build_processor(args.model_path, log=log.info)
    model = C.load_model(args.model_path, log=log.info)
    device = next(model.parameters()).device
    log.info("模型加载完成 %.1fs gpu=%s device=%s", time.perf_counter() - t_load, C.gpu_mem(), device)

    started = time.perf_counter()
    n_scenes_done = n_scenes_failed = n_turns_done = n_turns_failed = 0
    turn_latencies: list[float] = []
    scene_latencies: list[float] = []
    cards_errors: list[dict] = []
    todo_turns_done = 0

    for ci, (cid, scene_ids) in enumerate(plan):
        card_path = characters_dir / f"{cid}.json"
        try:
            sys_prompt = C.build_system_prompt(cid, characters_dir)
            scene_turns = C.load_scene_turns(cid, scenes_dir)
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s] 准备失败", cid)
            cards_errors.append({"card_id": cid, "error": str(exc)})
            continue

        records = existing_by_card[cid]
        out_path = out_dir / f"{cid}.jsonl"
        card_t0 = time.perf_counter()
        card_done = card_failed = 0
        for sid in scene_ids:
            turns = scene_turns.get(sid) or []
            rec = records.get(sid)
            if rec is not None and ("error" not in rec or not args.retry_errors):
                continue
            history: list[dict] = []
            turns_out: list[dict] = []
            scene_error: str | None = None
            scene_t0 = time.perf_counter()
            for turn in turns:
                idx = int(turn.get("idx", len(turns_out) + 1))
                user_text = str(turn.get("user_text", ""))
                try:
                    inputs = proc(text=user_text, system_prompt=sys_prompt, history=history,
                                  enable_thinking=False)
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
                    turns_out.append({"idx": idx, "user_text": user_text, "assistant_text": reply,
                                      "latency_s": round(latency, 3)})
                    turn_latencies.append(latency)
                    n_turns_done += 1
                    todo_turns_done += 1
                except Exception as exc:  # noqa: BLE001
                    scene_error = f"turn {idx} failed: {type(exc).__name__}: {exc}"
                    log.error("[%s] %s %s", cid, sid, scene_error)
                    turns_out.append({"idx": idx, "user_text": user_text, "assistant_text": "",
                                      "latency_s": None, "error": str(exc)})
                    n_turns_failed += 1
                    break
            scene_latency = round(time.perf_counter() - scene_t0, 3)
            record = {
                "card_id": cid,
                "scene_id": sid,
                "model": MODEL_SLUG,
                "pipeline": PIPELINE,
                "generation": {"temperature": 0.0, "max_tokens": args.max_new_tokens, "do_sample": False},
                "turns": turns_out,
                "latency_s": scene_latency,
            }
            if scene_error:
                record["error"] = scene_error
                n_scenes_failed += 1
                card_failed += 1
            else:
                n_scenes_done += 1
                card_done += 1
                scene_latencies.append(scene_latency)
            records[sid] = record
            write_records(out_path, scene_ids, records)

        if (ci + 1) % max(1, args.progress_every) == 0 or ci == len(plan) - 1:
            elapsed = time.perf_counter() - started
            rate = todo_turns_done / elapsed if elapsed > 0 else 0.0
            remaining_turns = max(0, todo_turns_planned - todo_turns_done)
            eta_min = (remaining_turns / rate / 60) if rate > 0 else float("inf")
            log.info("[%d/%d] %s: 场景 ok=%d fail=%d | 累计场景 ok=%d fail=%d | 轮次 ok=%d fail=%d | "
                     "本轮 %.1fs | 均延迟 %.2fs | %.1f 轮/min | 剩余约 %d 轮 ETA %.1f min",
                     ci + 1, len(plan), cid, card_done, card_failed, n_scenes_done, n_scenes_failed,
                     n_turns_done, n_turns_failed, time.perf_counter() - card_t0,
                     (sum(turn_latencies) / len(turn_latencies)) if turn_latencies else 0.0,
                     rate * 60, remaining_turns, eta_min)

    elapsed = time.perf_counter() - started
    avg = round(sum(turn_latencies) / len(turn_latencies), 3) if turn_latencies else 0.0
    srt = sorted(turn_latencies)

    def pct(p: float) -> float:
        if not srt:
            return 0.0
        return round(srt[min(len(srt) - 1, int(round(p / 100 * (len(srt) - 1))))], 3)

    manifest = {
        "model": MODEL_SLUG,
        "pipeline": PIPELINE,
        "mode": "text-only",
        "model_path": args.model_path,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cards": len(plan),
        "scenes_total": total_scenes,
        "turns_total_planned": total_turns_planned,
        "scenes_run_ok": n_scenes_done,
        "scenes_run_failed": n_scenes_failed,
        "turns_run_ok": n_turns_done,
        "turns_run_failed": n_turns_failed,
        "elapsed_s": round(elapsed, 2),
        "avg_turn_latency_s": avg,
        "turn_latency_p50_s": pct(50),
        "turn_latency_p95_s": pct(95),
        "turns_per_min": round(n_turns_done / elapsed * 60, 1) if elapsed > 0 else 0.0,
        "generation": {"temperature": 0.0, "max_tokens": args.max_new_tokens, "do_sample": False},
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "cards_errors": cards_errors,
        "log_file": args.log_file,
    }
    mpath = out_dir / "_manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("完成: 场景 ok=%d fail=%d | 轮次 ok=%d fail=%d | 用时 %.1fs | 均延迟 %.2fs (p50=%.2s p95=%.2s) | "
             "%.1f 轮/min | manifest: %s",
             n_scenes_done, n_scenes_failed, n_turns_done, n_turns_failed, elapsed, avg, pct(50), pct(95),
             manifest["turns_per_min"], mpath)
    # 进程退出前主动释放（避免和后续音频 track 抢显存）
    del model
    import torch as _torch
    _torch.cuda.empty_cache()
    return 0 if n_turns_failed == 0 and not cards_errors else 1


if __name__ == "__main__":
    sys.exit(main())
