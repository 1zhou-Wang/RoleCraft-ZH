#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音频 track B1/B2 级联的 LLM 环节：ASR 文本 → Qwen3.5-2B/4B（llama-server）逐轮生成。

输入:
  * data/voice/selection.json        50 卡 × 10 个语音场景
  * data/scenes/<card_id>.json       场景定义（turns 顺序 / user_text_gt / category）
  * data/characters/<card_id>.json   人设卡（system prompt 与文字 track 完全一致）
  * data/audio_track/asr_transcripts.jsonl  ASR 逐 turn 转写（user_text = ASR 文本）
输出:
  * baselines/audio_pipeline_qwen3.5-2b/<card_id>.jsonl   （B1, :8088）
  * baselines/audio_pipeline_qwen3.5-4b/<card_id>.jsonl   （B2, :8089）
  每行一场景：{card_id, scene_id, pipeline, model, generation, turns:[{idx,user_text,user_text_gt,
  assistant_text,latency_s}], latency_s}；副产物 _manifest.json

复用 run_baseline.py 的 system prompt 模板 / HTTP 调用 / 重试逻辑，保证与文字 track 一致。

用法:
  python local/audio_cascade.py --models b1            # 先跑 B1(2B)
  python local/audio_cascade.py --models b1,b2         # 两个都跑
  python local/audio_cascade.py --models b1 --limit-cards 2 --limit-scenes 2   # 冒烟
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from run_baseline import (  # noqa: E402  复用文字 track 的 prompt 模板与 HTTP 工具
    SYSTEM_PROMPT_TEMPLATE,
    build_system_prompt,
    chat_completion,
    http_json,
    wait_for_server,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"
OUT_ROOT = ROOT / "baselines"
LOG_DIR = ROOT / "logs" / "audio_track"

PIPELINES = {
    "b1": {"slug": "qwen3.5-2b", "pipeline": "B1", "base_url": "http://127.0.0.1:8088"},
    "b2": {"slug": "qwen3.5-4b", "pipeline": "B2", "base_url": "http://127.0.0.1:8089"},
}

log = logging.getLogger("audio_cascade")


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def load_selection() -> list[dict]:
    sel = json.loads((DATA / "voice" / "selection.json").read_text(encoding="utf-8"))
    return sel["items"]


def load_all_scenes() -> dict[str, dict[str, dict]]:
    """card_id -> scene_id -> scene"""
    out: dict[str, dict[str, dict]] = {}
    for f in sorted((DATA / "scenes").glob("*.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        out[f.stem] = {s["scene_id"]: s for s in doc.get("scenes") or []}
    return out


def load_transcripts() -> dict[tuple[str, str, int], dict]:
    path = DATA / "audio_track" / "asr_transcripts.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} 不存在，先跑 local/asr_transcribe.py")
    out: dict[tuple[str, str, int], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[(rec["card_id"], rec["scene_id"], int(rec["idx"]))] = rec
    return out


def read_existing(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("scene_id"):
            records[rec["scene_id"]] = rec
    return records


def write_records(path: Path, scene_order: list[str], records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for sid in scene_order:
            if sid in records:
                f.write(json.dumps(records[sid], ensure_ascii=False) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# 单卡任务：10 个语音场景，场景内按 turn 串行带 history
# --------------------------------------------------------------------------- #
def run_one_card(
    args: argparse.Namespace,
    pipeline: dict,
    card_id: str,
    scene_ids: list[str],
    card: dict,
    scenes: dict[str, dict],
    transcripts: dict[tuple[str, str, int], dict],
    stats: dict,
    stats_lock: threading.Lock,
) -> dict:
    out_path = OUT_ROOT / f"audio_pipeline_{pipeline['slug']}" / f"{card_id}.jsonl"
    result = {"card_id": card_id, "scenes_total": 0, "scenes_done": 0, "scenes_failed": 0, "turns_done": 0}
    try:
        system_prompt = build_system_prompt(card)
    except Exception as exc:  # noqa: BLE001
        log.error("[%s] 准备失败: %s", card_id, exc)
        result["error"] = str(exc)
        return result

    existing = read_existing(out_path)
    todo: list[str] = []
    for sid in scene_ids:
        if sid not in scenes:
            log.warning("[%s] 场景 %s 不在 data/scenes 中，跳过", card_id, sid)
            continue
        rec = existing.get(sid)
        if rec is not None and "error" not in rec:
            continue
        todo.append(sid)
    result["scenes_total"] = len(scene_ids)
    log.info("[%s] 语音场景 %d 个，待跑 %d 个", card_id, len(scene_ids), len(todo))

    for sid in todo:
        scene = scenes[sid]
        turns = sorted(scene.get("turns") or [], key=lambda t: int(t.get("idx", 0)))
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        turns_out: list[dict] = []
        scene_started = time.perf_counter()
        scene_error: str | None = None
        for turn in turns:
            idx = int(turn.get("idx", 0))
            tr = transcripts.get((card_id, sid, idx))
            if tr is None:
                scene_error = f"turn {idx} 缺 ASR 转写"
                turns_out.append({"idx": idx, "user_text": "", "user_text_gt": turn.get("user_text", ""),
                                  "assistant_text": "", "latency_s": None, "error": scene_error})
                break
            user_text = tr["asr_text"]
            messages.append({"role": "user", "content": user_text})
            try:
                reply, latency = chat_completion(
                    pipeline["base_url"], pipeline["slug"], messages,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    timeout=args.timeout, retries=args.retries,
                )
                messages.append({"role": "assistant", "content": reply})
                turns_out.append({
                    "idx": idx,
                    "user_text": user_text,
                    "user_text_gt": turn.get("user_text", ""),
                    "assistant_text": reply,
                    "latency_s": latency,
                })
                result["turns_done"] += 1
            except Exception as exc:  # noqa: BLE001
                scene_error = f"turn {idx} failed: {exc}"
                log.error("[%s] 场景 %s %s", card_id, sid, scene_error)
                turns_out.append({"idx": idx, "user_text": user_text,
                                  "user_text_gt": turn.get("user_text", ""),
                                  "assistant_text": "", "latency_s": None, "error": str(exc)})
                break
        scene_latency = round(time.perf_counter() - scene_started, 3)
        record = {
            "card_id": card_id,
            "scene_id": sid,
            "pipeline": pipeline["pipeline"],
            "model": pipeline["slug"],
            "generation": {"temperature": args.temperature, "max_tokens": args.max_tokens},
            "turns": turns_out,
            "latency_s": scene_latency,
        }
        if scene_error:
            record["error"] = scene_error
            result["scenes_failed"] += 1
        else:
            result["scenes_done"] += 1
        existing[sid] = record
        write_records(out_path, scene_ids, existing)
        with stats_lock:
            stats["turn_latencies"].extend(
                t["latency_s"] for t in turns_out if t.get("assistant_text") and t.get("latency_s") is not None
            )
            stats["scene_latencies"].append(scene_latency)
            if scene_error:
                stats["scene_errors"] += 1
    return result


# --------------------------------------------------------------------------- #
# 单 pipeline 主流程
# --------------------------------------------------------------------------- #
def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(round(p / 100.0 * (len(values) - 1))))
    return round(values[idx], 3)


def run_pipeline(args: argparse.Namespace, key: str) -> dict:
    pipeline = PIPELINES[key]
    slug = pipeline["slug"]
    out_dir = OUT_ROOT / f"audio_pipeline_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = load_selection()
    all_scenes = load_all_scenes()
    transcripts = load_transcripts()
    characters = {p.stem: json.loads(p.read_text(encoding="utf-8"))
                  for p in sorted((DATA / "characters").glob("*.json"))}

    tasks = []
    for item in selection:
        cid = item["card_id"]
        if cid not in characters:
            log.warning("缺角色卡 %s，跳过", cid)
            continue
        sids = item.get("scene_ids") or []
        if args.limit_scenes and args.limit_scenes > 0:
            sids = sids[: args.limit_scenes]
        tasks.append((cid, sids))
    if args.limit_cards and args.limit_cards > 0:
        tasks = tasks[: args.limit_cards]
    if args.cards:
        wanted = {c.strip() for c in args.cards.split(",") if c.strip()}
        tasks = [t for t in tasks if t[0] in wanted]

    total_turns = 0
    for cid, sids in tasks:
        for sid in sids:
            total_turns += len(all_scenes.get(cid, {}).get(sid, {}).get("turns") or [])
    log.info("=%s [%s] | 卡 %d | 场景 %d | turn %d | server %s | concurrency=%d",
             slug, pipeline["pipeline"], len(tasks), sum(len(t[1]) for t in tasks), total_turns,
             pipeline["base_url"], args.concurrency)

    if args.dry_run:
        for cid, sids in tasks:
            log.info("[dry-run] %s -> %s (%d scenes)", cid, out_dir / f"{cid}.jsonl", len(sids))
        return {"pipeline": pipeline["pipeline"], "model": slug, "dry_run": True}

    if not wait_for_server(pipeline["base_url"], args.wait_server):
        raise RuntimeError(f"llama-server 未就绪: {pipeline['base_url']}")

    stats = {"turn_latencies": [], "scene_latencies": [], "scene_errors": 0}
    stats_lock = threading.Lock()
    results: list[dict] = []
    started = time.perf_counter()
    started_at = datetime.now()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {
            pool.submit(run_one_card, args, pipeline, cid, sids, characters[cid],
                        all_scenes.get(cid, {}), transcripts, stats, stats_lock): cid
            for cid, sids in tasks
        }
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.exception("[%s] 未捕获异常", cid)
                res = {"card_id": cid, "error": str(exc), "scenes_total": 0, "scenes_done": 0,
                       "scenes_failed": 0, "turns_done": 0}
            results.append(res)
            if res.get("error"):
                with stats_lock:
                    stats.setdefault("card_errors", []).append({"card_id": cid, "error": res["error"]})
            log.info("[%s] done: scenes %d/%d, failed %d", cid, res.get("scenes_done", 0),
                     res.get("scenes_total", 0), res.get("scenes_failed", 0))
    elapsed = time.perf_counter() - started
    tlat, slat = sorted(stats["turn_latencies"]), sorted(stats["scene_latencies"])
    manifest = {
        "pipeline": pipeline["pipeline"],
        "model": slug,
        "base_url": pipeline["base_url"],
        "system_prompt_template": SYSTEM_PROMPT_TEMPLATE,
        "asr_transcripts": str(DATA / "audio_track" / "asr_transcripts.jsonl"),
        "cards": len(tasks),
        "scenes_total": sum(r.get("scenes_total", 0) for r in results),
        "scenes_ok": sum(r.get("scenes_done", 0) for r in results),
        "scenes_failed": sum(r.get("scenes_failed", 0) for r in results),
        "turns_ok": sum(r.get("turns_done", 0) for r in results),
        "elapsed_s": round(elapsed, 2),
        "turn_latency_s": {"mean": round(sum(tlat) / len(tlat), 3) if tlat else 0.0,
                           "p50": pct(tlat, 50), "p95": pct(tlat, 95), "max": tlat[-1] if tlat else 0.0},
        "scene_latency_s": {"mean": round(sum(slat) / len(slat), 3) if slat else 0.0,
                            "p50": pct(slat, 50), "p95": pct(slat, 95)},
        "generation": {"temperature": args.temperature, "max_tokens": args.max_tokens},
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "card_errors": stats.get("card_errors", []),
    }
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("=%s 完成: 场景 %d/%d ok, turn %d, 用时 %.1fs, turn 延迟 mean=%.3fs p95=%.3fs",
             slug, manifest["scenes_ok"], manifest["scenes_total"], manifest["turns_ok"], elapsed,
             manifest["turn_latency_s"]["mean"], manifest["turn_latency_s"]["p95"])
    return manifest


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="音频 track B1/B2 级联 LLM runner")
    ap.add_argument("--models", default="b1,b2", help="跑哪些 pipeline：b1(2B)/b2(4B)，逗号分隔")
    ap.add_argument("--cards", default=None, help="只跑指定卡，逗号分隔")
    ap.add_argument("--limit-cards", type=int, default=0)
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--wait-server", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-file", default=str(LOG_DIR / "audio_cascade.log"))
    a = ap.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(a.log_file, encoding="utf-8")],
        force=True,
    )
    keys = [k.strip().lower() for k in a.models.split(",") if k.strip()]
    for k in keys:
        if k not in PIPELINES:
            log.error("未知 pipeline: %s（可选 b1/b2）", k)
            return 2
    summaries = []
    rc = 0
    for k in keys:
        summary = run_pipeline(a, k)
        summaries.append(summary)
        if not a.dry_run and (summary.get("scenes_failed") or summary.get("card_errors")):
            rc = 1
    if not a.dry_run:
        summary_path = OUT_ROOT / "audio_pipeline_summary.json"
        summary_path.write_text(json.dumps({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "pipelines": summaries,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("summary -> %s", summary_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
