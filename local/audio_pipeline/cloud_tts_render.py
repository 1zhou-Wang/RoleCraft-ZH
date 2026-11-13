#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端 TTS 渲染：把 B1/B2 的 assistant_text 用 qwen-audio-3.1-tts-next 渲染成 mp3。

背景：本地 Qwen3-TTS-0.6B 在 3090 上 RTF≈2-3x（18s/条），全量 1568 条不可接受；
因此全量音频改用与用户语音同一套云端 TTS（DashScope MaaS 原生
/services/audio/tts/SpeechSynthesizer），本地 TTS 只保留少量链路验证样本。

输入: baselines/audio_pipeline_<slug>/*.jsonl（逐 turn assistant_text）
输出:
  * baselines/audio_pipeline_<slug>/audio_cloud/<scene_id>_<idx>.mp3
  * 更新 jsonl：turn.assistant_audio = "audio_cloud/<scene_id>_<idx>.mp3"
                 turn.tts_render = "qwen-audio-3.1-tts-next"
  * baselines/audio_pipeline_<slug>/audio_cloud/_render_manifest.json
失败清单: logs/audio_track/cloud_tts_failures.json

用法:
  python local/audio_pipeline/cloud_tts_render.py --pilot 20             # 测速
  python local/audio_pipeline/cloud_tts_render.py                        # 全量
  python local/audio_pipeline/cloud_tts_render.py --max-items 600        # 只渲染均匀抽样 600 条
  python local/audio_pipeline/cloud_tts_render.py --time-budget-sec 1200 # 20 分钟后优雅停止
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASELINES = ROOT / "baselines"
LOG_DIR = ROOT / "logs" / "audio_track"
MODEL_TTS_DEFAULT = "qwen-audio-3.1-tts-next"
RENDER_TAG = "qwen-audio-3.1-tts-next"

log = logging.getLogger("cloud_tts")


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    p = ROOT / ".env"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float = 4.0):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.ts = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
                self.ts = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                need = (1.0 - self.tokens) / self.rate
            time.sleep(min(need, 0.5))


def build_prompt(text: str, character_name: str | None = None) -> str:
    """让 TTS 用自然、贴合角色的口吻朗读 assistant_text（保留括号动作原文，便于音频判官对齐文本）。"""
    who = f"你是{character_name}。" if character_name else ""
    return f"{who}用自然、符合角色性格与语气的口吻说：“{text}” 要求：纯人声，无背景音乐，无音效。"


def load_cards() -> dict[str, str]:
    out = {}
    for p in sorted((ROOT / "data" / "characters").glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out[p.stem] = d.get("name") or p.stem
        except Exception:  # noqa: BLE001
            out[p.stem] = p.stem
    return out


def load_items(slugs: list[str]) -> tuple[list[dict], dict, dict]:
    """返回 (items, order, records)。items 中跳过已由本地 TTS 渲染的 turn。"""
    order: dict[tuple[str, str], list[str]] = {}
    records: dict[tuple[str, str], dict[str, dict]] = {}
    items: list[dict] = []
    for slug in slugs:
        for path in sorted((BASELINES / f"audio_pipeline_{slug}").glob("*.jsonl")):
            key = (slug, path.stem)
            order[key] = []
            records[key] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                sid = rec.get("scene_id")
                if not sid:
                    continue
                order[key].append(sid)
                records[key][sid] = rec
                for turn in rec.get("turns") or []:
                    idx = int(turn.get("idx", 0))
                    text = (turn.get("assistant_text") or "").strip()
                    if not text:
                        continue
                    if turn.get("assistant_audio") and turn.get("tts_render") != RENDER_TAG:
                        continue  # 已由本地 TTS 渲染，跳过
                    items.append({
                        "slug": slug,
                        "card_id": path.stem,
                        "scene_id": sid,
                        "idx": idx,
                        "text": text,
                        "mp3": BASELINES / f"audio_pipeline_{slug}" / "audio_cloud" / f"{sid}_{idx}.mp3",
                    })
    return items, order, records


def write_records(slug: str, card_id: str, scene_order: list[str], records: dict[str, dict]) -> None:
    path = BASELINES / f"audio_pipeline_{slug}" / f"{card_id}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for sid in scene_order:
            if sid in records:
                f.write(json.dumps(records[sid], ensure_ascii=False) + "\n")
    tmp.replace(path)


def uniform_sample(items: list[dict], n: int) -> list[dict]:
    """按卡均匀抽样 n 条：先在卡内交错，再在全量上等距取点。"""
    if n <= 0 or n >= len(items):
        return items
    step = len(items) / n
    picked = []
    seen = set()
    for i in range(n):
        j = int(i * step)
        while j in seen and j < len(items) - 1:
            j += 1
        seen.add(j)
        picked.append(items[j])
    return picked


def interleave_balanced(items: list[dict]) -> list[dict]:
    """按 (卡, slug) 轮转 + 卡内位置交错，保证前缀覆盖尽量多的卡与两个模型。"""
    groups: dict[tuple[str, str], list[dict]] = {}
    for it in items:
        groups.setdefault((it["card_id"], it["slug"]), []).append(it)
    pos = {k: 0 for k in groups}
    keys = sorted(groups.keys())  # card_id 优先、slug 次之
    out: list[dict] = []
    remaining = len(items)
    while remaining > 0:
        for k in keys:
            i = pos[k]
            if i < len(groups[k]):
                out.append(groups[k][i])
                pos[k] = i + 1
                remaining -= 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="云端 TTS 渲染 B1/B2 assistant_text → mp3")
    ap.add_argument("--slugs", default="qwen3.5-2b,qwen3.5-4b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=float, default=32.0, help="请求速率上限（条/分钟）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pilot", type=int, default=0, help="只渲染前 N 条测速（不写 jsonl？会写）")
    ap.add_argument("--max-items", type=int, default=0, help="均匀抽样最多 N 条（0=全部）")
    ap.add_argument("--time-budget-sec", type=float, default=0, help="超过该时间优雅停止（0=不限）")
    ap.add_argument("--force", action="store_true", help="重渲已存在 mp3")
    ap.add_argument("--balanced", action="store_true", default=True,
                    help="按 (卡,slug) 轮转排序，保证部分完成时按卡均匀（默认开）")
    ap.add_argument("--no-balanced", dest="balanced", action="store_false")
    ap.add_argument("--log-file", default=str(LOG_DIR / "cloud_tts.log"))
    a = ap.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(a.log_file, encoding="utf-8")],
        force=True,
    )

    env = load_env()
    api_key = env.get("DASHSCOPE_API_KEY")
    base = (env.get("DASHSCOPE_WS_BASE") or "").rstrip("/")
    model = env.get("MODEL_TTS", MODEL_TTS_DEFAULT)
    if not api_key or not base:
        log.error("缺少 DASHSCOPE_API_KEY / DASHSCOPE_WS_BASE（.env）")
        return 2
    url = base + "/services/audio/tts/SpeechSynthesizer"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    import requests

    slugs = [s.strip() for s in a.slugs.split(",") if s.strip()]
    cards = load_cards()
    items, order, records = load_items(slugs)
    log.info("待渲染 turn=%d（slugs=%s）", len(items), slugs)

    existing = [it for it in items if it["mp3"].is_file() and it["mp3"].stat().st_size > 1024]
    todo = [it for it in items if a.force or it not in existing]
    if a.pilot:
        todo = todo[: a.pilot]
    elif a.max_items:
        todo = uniform_sample(todo, a.max_items)
    if a.balanced:
        todo = interleave_balanced(todo)
    log.info("已存在 mp3=%d, 本次渲染=%d, workers=%d, rpm=%.0f", len(existing), len(todo), a.workers, a.rpm)
    if not todo:
        log.info("无需渲染")
        return 0

    for slug in slugs:
        (BASELINES / f"audio_pipeline_{slug}" / "audio_cloud").mkdir(parents=True, exist_ok=True)

    bucket = TokenBucket(a.rpm / 60.0)
    lock = threading.Lock()
    failures: list[dict] = []
    latencies: list[float] = []
    durations: list[float] = []
    started = time.perf_counter()
    stop_at = started + a.time_budget_sec if a.time_budget_sec > 0 else None
    touched: set[tuple[str, str]] = set()
    abort = threading.Event()
    done_count = 0

    def flush() -> None:
        with lock:
            keys = list(touched)
            touched.clear()
        for (slug, card_id) in keys:
            write_records(slug, card_id, order[(slug, card_id)], records[(slug, card_id)])

    def render_one(it: dict) -> dict:
        payload = {
            "model": model,
            "input": {
                "text_prompt": build_prompt(it["text"], cards.get(it["card_id"])),
                "format": "mp3",
                "sample_rate": 24000,
                "channels": 1,
                "rate": 1.0,
                "seed": a.seed,
            },
        }
        last_err = None
        for attempt in range(1, 6):
            if abort.is_set():
                raise RuntimeError("aborted by time budget")
            bucket.acquire()
            t0 = time.perf_counter()
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=300)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_err = f"network: {exc}"
                time.sleep(min(30, 2 ** attempt) + random.random())
                continue
            elapsed = time.perf_counter() - t0
            if r.status_code == 200:
                body = r.json()
                audio = ((body.get("output") or {}).get("audio")) or {}
                audio_url = audio.get("url")
                if not audio_url:
                    last_err = f"200 but no audio url: {json.dumps(body, ensure_ascii=False)[:200]}"
                    time.sleep(min(20, 2 ** attempt))
                    continue
                r2 = requests.get(audio_url, timeout=300)
                r2.raise_for_status()
                data = r2.content
                if len(data) < 1024:
                    last_err = f"downloaded too small: {len(data)} bytes"
                    time.sleep(min(20, 2 ** attempt))
                    continue
                it["mp3"].parent.mkdir(parents=True, exist_ok=True)
                tmp = it["mp3"].with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(it["mp3"])
                return {"ok": True, "latency": elapsed,
                        "duration": audio.get("duration"), "bytes": len(data)}
            retryable = r.status_code in (408, 429) or r.status_code >= 500
            try:
                body = r.json()
            except Exception:  # noqa: BLE001
                body = {"raw": r.text[:200]}
            last_err = f"http_{r.status_code}: {json.dumps(body, ensure_ascii=False)[:200]}"
            if not retryable:
                break
            wait = min(15, 0.5 * 2 ** (attempt - 1)) + random.random() * 0.5 if r.status_code == 429 \
                else min(60, 2 ** attempt) + random.random()
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    wait = max(wait, float(ra))
                except ValueError:
                    pass
            time.sleep(wait)
        raise RuntimeError(last_err or "unknown error")

    def work(it: dict) -> dict:
        nonlocal done_count
        if stop_at is not None and time.perf_counter() > stop_at:
            abort.set()
        res = render_one(it)
        rec = records[(it["slug"], it["card_id"])][it["scene_id"]]
        for turn in rec.get("turns") or []:
            if int(turn.get("idx", 0)) == it["idx"]:
                turn["assistant_audio"] = f"audio_cloud/{it['scene_id']}_{it['idx']}.mp3"
                turn["tts_render"] = RENDER_TAG
                break
        with lock:
            touched.add((it["slug"], it["card_id"]))
            latencies.append(res["latency"])
            if res.get("duration"):
                durations.append(float(res["duration"]))
            done_count += 1
            need_flush = done_count % 20 == 0
        if need_flush:
            flush()
        return {"ok": True, "slug": it["slug"], "scene_id": it["scene_id"], "idx": it["idx"]}

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futures = {pool.submit(work, it): it for it in todo}
        for fut in as_completed(futures):
            it = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                if abort.is_set():
                    continue
                failures.append({"slug": it["slug"], "card_id": it["card_id"], "scene_id": it["scene_id"],
                                 "idx": it["idx"], "error": str(exc)[:300]})
                log.error("失败 %s_%s: %s", it["scene_id"], it["idx"], str(exc)[:200])
            if done_count and done_count % 100 == 0:
                el = time.perf_counter() - started
                log.info("进度 %d/%d, 失败 %d, %.0f 条/分钟", done_count, len(todo), len(failures),
                         done_count / el * 60 if el else 0)
    flush()
    elapsed = time.perf_counter() - started

    # manifest
    for slug in slugs:
        slug_items = [it for it in items if it["slug"] == slug]
        ok = [it for it in slug_items if it["mp3"].is_file() and it["mp3"].stat().st_size > 1024]
        manifest = {
            "model": model,
            "render": RENDER_TAG,
            "format": "mp3",
            "sample_rate": 24000,
            "turns_total": len(slug_items),
            "mp3_ok": len(ok),
            "mp3_missing": len(slug_items) - len(ok),
            "workers": a.workers,
            "rpm_limit": a.rpm,
            "elapsed_s": round(elapsed, 1),
            "throughput_per_min": round(done_count / elapsed * 60, 1) if elapsed else 0,
            "request_latency_s": {
                "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
                "p50": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
            },
            "audio_duration_s": {
                "mean": round(sum(durations) / len(durations), 2) if durations else 0.0,
                "total": round(sum(durations), 1),
            },
            "aborted_by_time_budget": bool(abort.is_set()),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        (BASELINES / f"audio_pipeline_{slug}" / "audio_cloud" / "_render_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if failures:
        (LOG_DIR / "cloud_tts_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        log.error("失败 %d 条 -> %s", len(failures), LOG_DIR / "cloud_tts_failures.json")
    log.info("云端渲染完成: %d/%d 条, 失败 %d, 用时 %.1fs (%.1f 条/分钟)",
             done_count, len(todo), len(failures), elapsed, done_count / elapsed * 60 if elapsed else 0)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
