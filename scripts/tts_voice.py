#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RoleCraft-ZH voice track: synthesize per-turn user audio.

Model: qwen-audio-3.1-tts-next (DashScope MaaS, native HTTP endpoint)
Voice: fixed "tester user" reference audio (data/voice/tester_voice_ref.wav) injected
       as input.references and referenced as @voice1 in text_prompt.

Inputs
  data/voice/selection.json      {"items":[{"card_id":..., "scene_ids":[...]}]}
  data/scenes/<card_id>.json     scenes[*].turns[*] = {idx, user_text, tone}
  data/characters/<card_id>.json character name for the prompt
  data/voice/tester_voice_ref.wav fixed reference voice (~12-20 s)

Outputs
  data/voice/audio/<scene_id>_<idx>.mp3
  data/voice/manifest.jsonl      one line per turn (append + dedupe at end)
  data/voice/failures.jsonl      one line per permanently failed turn
  logs/voice/tts_voice_last_run.json  run summary

Features: concurrency (default 3), 180 RPM token bucket, 429/5xx exponential backoff,
          resumable (skips audio that already exists), --mock for offline validation.

Never prints the API key. Only touches data/voice/, logs/voice/ and work/voice/.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

TONE_ENUM = ["平静", "开心", "兴奋", "好奇", "疑惑", "惊讶", "愤怒", "不满", "伤心", "担忧",
             "焦急", "害怕", "温柔", "关切", "撒娇", "羞涩", "得意", "嘲讽", "调侃", "冷淡",
             "严肃", "坚定", "无奈", "疲惫", "期待", "恳求"]
TONE_SET = set(TONE_ENUM)

PROMPT_TEMPLATE = (
    "@voice1 测试用户（青年男性，声音清亮，语速自然）用「{tone}」的语气对「{char}」说：“{text}”"
    " 要求：纯人声，无背景音乐，无音效。"
)

DEFAULTS = dict(
    scenes_dir=ROOT / "data" / "scenes",
    characters_dir=ROOT / "data" / "characters",
    selection=ROOT / "data" / "voice" / "selection.json",
    voice_ref=ROOT / "data" / "voice" / "tester_voice_ref.wav",
    out_dir=ROOT / "data" / "voice" / "audio",
    manifest=ROOT / "data" / "voice" / "manifest.jsonl",
    failures=ROOT / "data" / "voice" / "failures.jsonl",
    log_dir=ROOT / "logs" / "voice",
)


# ---------------------------------------------------------------- env / logging
def load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


class Log:
    """Console + file logger (no API keys ever passed here)."""

    def __init__(self, path: Path | None):
        self.lock = threading.Lock()
        self.fh = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with self.lock:
            print(line, flush=True)
            if self.fh:
                self.fh.write(line + "\n")
                self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


# ---------------------------------------------------------------- rate limiting
class TokenBucket:
    """Thread-safe token bucket enforcing a requests-per-minute ceiling."""

    def __init__(self, rpm: float, burst: int = 3):
        self.capacity = max(1.0, float(burst))
        self.tokens = self.capacity
        self.rate = rpm / 60.0
        self.ts = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
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


# ---------------------------------------------------------------- TTS client
class TTSError(Exception):
    def __init__(self, msg, *, retryable=False, status=None, request_id=None, detail=None):
        super().__init__(msg)
        self.retryable = retryable
        self.status = status
        self.request_id = request_id
        self.detail = detail


class TTSClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 300):
        self.url = base_url.rstrip("/") + "/services/audio/tts/SpeechSynthesizer"
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}",
                                     "Content-Type": "application/json"})

    def synthesize(self, payload: dict, bucket: TokenBucket, log: Log, tag: str,
                   max_attempts: int = 5):
        """POST one TTS request, return (audio_url, duration, request_id)."""
        last = None
        for attempt in range(1, max_attempts + 1):
            bucket.acquire()
            t0 = time.time()
            try:
                r = self.session.post(self.url, json=payload, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last = TTSError(f"{type(e).__name__}: {e}", retryable=True)
                log(f"  [retry {attempt}] {tag} network error: {type(e).__name__}")
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            elapsed = round(time.time() - t0, 2)
            try:
                body = r.json()
            except Exception:  # noqa: BLE001
                body = {"_raw": r.text[:500]}
            rid = body.get("request_id") or r.headers.get("X-Request-Id", "")
            if r.status_code == 200:
                out = body.get("output") or {}
                audio = out.get("audio") or {}
                url = audio.get("url") or out.get("url")
                dur = audio.get("duration")
                if not url:
                    last = TTSError("no_audio_url", status=200, request_id=rid,
                                    detail=body)
                    log(f"  [retry {attempt}] {tag} 200 but no audio url: "
                        f"{json.dumps(body, ensure_ascii=False)[:300]}")
                    time.sleep(min(30, 2 ** attempt))
                    continue
                return url, dur, rid
            retryable = r.status_code == 429 or r.status_code >= 500 or r.status_code in (408,)
            msg = body.get("message") or body.get("code") or r.text[:300]
            last = TTSError(f"http_{r.status_code}: {msg}", retryable=retryable,
                            status=r.status_code, request_id=rid, detail=body)
            if not retryable:
                raise last
            wait = min(60, 2 ** attempt) + random.random()
            if r.status_code == 429:
                # concurrency throttle: short jittered backoff is enough
                wait = min(15, 0.5 * 2 ** (attempt - 1)) + random.random() * 0.5
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    wait = max(wait, float(ra))
                except ValueError:
                    pass
            log(f"  [retry {attempt}] {tag} HTTP {r.status_code} ({elapsed}s), "
                f"backoff {wait:.1f}s")
            time.sleep(wait)
        raise last or TTSError("unknown_error", retryable=True)

    def download(self, url: str, dest: Path, retries: int = 3) -> int:
        last = None
        for attempt in range(1, retries + 1):
            try:
                with requests.get(url, stream=True, timeout=self.timeout) as r:
                    r.raise_for_status()
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    n = 0
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(65536):
                            if chunk:
                                f.write(chunk)
                                n += len(chunk)
                    if n < 1024:
                        raise IOError(f"downloaded file too small ({n} bytes)")
                    os.replace(tmp, dest)
                    return n
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(min(10, 2 ** attempt))
        raise IOError(f"download failed: {last}")


# ---------------------------------------------------------------- job building
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_jobs(args) -> tuple[list[dict], list[str]]:
    """Return (jobs, warnings)."""
    warnings: list[str] = []
    selection = load_json(args.selection)
    items = selection.get("items") if isinstance(selection, dict) else selection
    if not isinstance(items, list):
        raise SystemExit(f"bad selection file: {args.selection}")

    jobs: list[dict] = []
    seen = set()
    for item in items:
        card_id = item.get("card_id")
        scene_ids = item.get("scene_ids") or []
        if not card_id:
            warnings.append(f"selection item without card_id: {item}")
            continue
        scenes_path = Path(args.scenes_dir) / f"{card_id}.json"
        if not scenes_path.exists():
            warnings.append(f"missing scenes file for {card_id}: {scenes_path}")
            continue
        char_name = card_id
        char_path = Path(args.characters_dir) / f"{card_id}.json"
        if char_path.exists():
            try:
                char_name = load_json(char_path).get("name") or card_id
            except Exception as e:  # noqa: BLE001
                warnings.append(f"bad character file {char_path}: {e}")
        scenes_doc = load_json(scenes_path)
        scene_map = {s.get("scene_id"): s for s in scenes_doc.get("scenes", [])}
        for scene_id in scene_ids:
            scene = scene_map.get(scene_id)
            if scene is None:
                warnings.append(f"scene {scene_id} not found in {scenes_path}")
                continue
            category = scene.get("category", "")
            turns = sorted(scene.get("turns", []), key=lambda t: int(t.get("idx", 0)))
            for turn in turns:
                idx = int(turn.get("idx", 0))
                text = (turn.get("user_text") or "").strip()
                if not text:
                    warnings.append(f"empty user_text: {scene_id} idx={idx}")
                    continue
                tone = turn.get("tone") or "平静"
                if tone not in TONE_SET:
                    warnings.append(f"unknown tone '{tone}' at {scene_id} idx={idx}, using 平静")
                    tone = "平静"
                key = (card_id, scene_id, idx)
                if key in seen:
                    warnings.append(f"duplicate turn in selection: {key}")
                    continue
                seen.add(key)
                prompt = PROMPT_TEMPLATE.format(tone=tone, char=char_name, text=text)
                jobs.append({
                    "card_id": card_id,
                    "scene_id": scene_id,
                    "scene_category": category,
                    "idx": idx,
                    "user_text": text,
                    "tone": tone,
                    "char_name": char_name,
                    "tts_prompt": prompt,
                })
    return jobs, warnings


# ---------------------------------------------------------------- manifest helpers
def read_manifest(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def write_manifest_atomic(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def manifest_key(rec: dict):
    return (rec.get("card_id"), rec.get("scene_id"), rec.get("idx"))


# ---------------------------------------------------------------- mock helpers
def make_silent_mp3(dest: Path, duration: float) -> int:
    import subprocess
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.mp3")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
           "-i", "anullsrc=r=24000:cl=mono", "-t", f"{max(0.4, duration):.2f}",
           "-q:a", "9", "-acodec", "libmp3lame", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        os.replace(tmp, dest)
        return dest.stat().st_size
    except Exception:  # noqa: BLE001
        # fallback: minimal placeholder (mock only)
        data = b"\xff\xfb\x90\x00" + b"\x00" * 1024
        dest.write_bytes(data)
        return len(data)


# ---------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RoleCraft-ZH TTS synthesis for user turns")
    p.add_argument("--env", type=Path, default=ROOT / ".env")
    p.add_argument("--scenes-dir", type=Path, default=DEFAULTS["scenes_dir"])
    p.add_argument("--characters-dir", type=Path, default=DEFAULTS["characters_dir"])
    p.add_argument("--selection", type=Path, default=DEFAULTS["selection"])
    p.add_argument("--voice-ref", type=Path, default=DEFAULTS["voice_ref"])
    p.add_argument("--out-dir", type=Path, default=DEFAULTS["out_dir"])
    p.add_argument("--manifest", type=Path, default=DEFAULTS["manifest"])
    p.add_argument("--failures", type=Path, default=DEFAULTS["failures"])
    p.add_argument("--log-dir", type=Path, default=DEFAULTS["log_dir"])
    p.add_argument("--workers", type=int, default=3, help="parallel requests (default 3)")
    p.add_argument("--rpm", type=float, default=180.0, help="requests-per-minute ceiling")
    p.add_argument("--retry-max", type=int, default=5, help="max attempts per turn")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--format", default="mp3", choices=["mp3", "wav", "opus"])
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--rate", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="only first N pending jobs")
    p.add_argument("--only-card", default="", help="comma separated card_id filter")
    p.add_argument("--only-scene", default="", help="comma separated scene_id filter")
    p.add_argument("--mock", action="store_true", help="offline: write silent audio, no API")
    p.add_argument("--mock-fail-rate", type=float, default=0.0,
                   help="mock only: fraction of runs that fail (tests failures.jsonl)")
    p.add_argument("--no-reference", action="store_true", help="ablation: skip @voice1 reference")
    p.add_argument("--force", action="store_true", help="regenerate existing audio")
    p.add_argument("--dry-run", action="store_true", help="list jobs only, no writes")
    p.add_argument("--max-text-chars", type=int, default=0,
                   help="sanity check: warn if user_text exceeds this (0=off)")
    return p.parse_args(argv)


def rel_to_root(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(Path(p).resolve()).replace("\\", "/")


def main(argv=None) -> int:
    args = parse_args(argv)
    # normalise all path arguments to absolute paths
    for attr in ("env", "scenes_dir", "characters_dir", "selection", "voice_ref",
                 "out_dir", "manifest", "failures", "log_dir"):
        setattr(args, attr, Path(getattr(args, attr)).resolve())
    t_start = time.time()
    log_path = Path(args.log_dir) / f"tts_voice_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log = Log(log_path if not args.dry_run else None)
    log(f"== tts_voice.py start mode={'mock' if args.mock else 'real'} "
        f"workers={args.workers} rpm={args.rpm} ==")

    env = load_env(Path(args.env))
    api_key = env.get("DASHSCOPE_API_KEY", "")
    model = env.get("MODEL_TTS", "qwen-audio-3.1-tts-next")
    ws_base = env.get("DASHSCOPE_WS_BASE", "")
    if not args.mock and (not api_key or not ws_base):
        raise SystemExit("DASHSCOPE_API_KEY / DASHSCOPE_WS_BASE missing in .env")

    # reference voice
    voice_ref_uri = None
    if not args.no_reference:
        if not Path(args.voice_ref).exists():
            raise SystemExit(f"voice reference not found: {args.voice_ref} "
                             f"(run the reference generation step first, or pass --no-reference)")
        with open(args.voice_ref, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        voice_ref_uri = "data:audio/wav;base64," + b64
        log(f"reference voice: {args.voice_ref} (b64 {len(voice_ref_uri) // 1024} KiB)")

    jobs, warnings = build_jobs(args)
    for w in warnings[:20]:
        log(f"  WARN {w}")
    if len(warnings) > 20:
        log(f"  WARN ... {len(warnings) - 20} more")
    log(f"selection: {len(jobs)} turns before filters")

    card_filter = {x.strip() for x in args.only_card.split(",") if x.strip()}
    scene_filter = {x.strip() for x in args.only_scene.split(",") if x.strip()}
    if card_filter:
        jobs = [j for j in jobs if j["card_id"] in card_filter]
    if scene_filter:
        jobs = [j for j in jobs if j["scene_id"] in scene_filter]

    # resume: skip existing audio + manifest record
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True) if not args.dry_run else None
    existing = {manifest_key(r): r for r in read_manifest(Path(args.manifest))}
    pending, skipped = [], 0
    for job in jobs:
        fname = f"{job['scene_id']}_{job['idx']}.{args.format}"
        dest = out_dir / fname
        job["file_rel"] = rel_to_root(dest)
        job["dest"] = dest
        if not args.force and dest.exists() and dest.stat().st_size > 1024 and manifest_key(job) in existing:
            skipped += 1
            continue
        pending.append(job)
    log(f"pending={len(pending)} skipped_already_done={skipped} "
        f"total_selected={len(jobs)}")

    if args.dry_run:
        for j in pending[:10]:
            log(f"  DRY {j['file_rel']} tone={j['tone']} text={j['user_text'][:30]}")
        log(f"DRY RUN: would synthesize {len(pending)} files into {out_dir}")
        return 0
    if args.limit:
        pending = pending[: args.limit]
        log(f"--limit applied: {len(pending)} jobs this run")

    if not pending:
        log("nothing to do")
        summary = {"mode": "mock" if args.mock else "real", "total_selected": len(jobs),
                   "skipped": skipped, "synthesized": 0, "failed": 0, "elapsed_s": 0}
        _write_summary(args, log, summary)
        return 0

    client = None
    if not args.mock:
        client = TTSClient(ws_base, api_key, model, timeout=args.timeout)
    bucket = TokenBucket(args.rpm, burst=max(1, args.workers))

    manifest_lock = threading.Lock()
    failures_lock = threading.Lock()
    counter_lock = threading.Lock()
    mpath = Path(args.manifest)
    fpath = Path(args.failures)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    manifest_fh = open(mpath, "a", encoding="utf-8")
    failures_fh = open(fpath, "a", encoding="utf-8")
    stats = {"ok": 0, "failed": 0, "bytes": 0, "dur": 0.0, "latency": []}

    def write_record(fh, lock, rec):
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with lock:
            fh.write(line)
            fh.flush()

    def synth_one(job: dict):
        """Returns (manifest record, elapsed seconds) or raises TTSError/IOError."""
        t_job = time.time()
        payload = {
            "model": model,
            "input": {
                "text_prompt": job["tts_prompt"],
                "format": args.format,
                "sample_rate": args.sample_rate,
                "channels": args.channels,
                "rate": args.rate,
                "seed": args.seed,
            },
        }
        if voice_ref_uri:
            payload["input"]["references"] = [{"audio_data": voice_ref_uri}]
        tag = f"{job['scene_id']}_{job['idx']}"
        if args.mock:
            if args.mock_fail_rate > 0 and random.random() < args.mock_fail_rate:
                raise TTSError("mock_transient_failure", retryable=False, status=500)
            dur = round(max(0.4, len(job["user_text"]) / 4.5 + 0.4), 2)
            size = make_silent_mp3(job["dest"], dur)
            return {**_rec_base(job, args, model), "duration": dur,
                    "request_id": f"mock-{int(time.time() * 1000)}", "bytes": size}, \
                round(time.time() - t_job, 2)
        url, dur, rid = client.synthesize(payload, bucket, log, tag,
                                          max_attempts=args.retry_max)
        size = client.download(url, job["dest"])
        rec = {**_rec_base(job, args, model), "duration": dur,
               "request_id": rid, "bytes": size}
        return rec, round(time.time() - t_job, 2)

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = {ex.submit(synth_one, j): j for j in pending}
            n_done = 0
            for fut in as_completed(futures):
                job = futures[fut]
                n_done += 1
                try:
                    rec, latency = fut.result()
                    write_record(manifest_fh, manifest_lock, rec)
                    with counter_lock:
                        stats["ok"] += 1
                        stats["bytes"] += rec.get("bytes") or 0
                        stats["dur"] += rec.get("duration") or 0
                        stats["latency"].append(latency)
                    if n_done % 10 == 0 or n_done <= 3:
                        el = time.time() - t_start
                        rate = n_done / el * 60 if el else 0
                        eta = (len(pending) - n_done) / max(1e-9, n_done / el) if el else 0
                        log(f"  progress {n_done}/{len(pending)} ok={stats['ok']} "
                            f"fail={stats['failed']} {rate:.1f} req/min ETA {eta / 60:.1f} min")
                except Exception as e:  # noqa: BLE001
                    err = {
                        "card_id": job["card_id"], "scene_id": job["scene_id"],
                        "idx": job["idx"], "user_text": job["user_text"], "tone": job["tone"],
                        "status": getattr(e, "status", None),
                        "request_id": getattr(e, "request_id", None),
                        "error": f"{type(e).__name__}: {e}",
                    }
                    det = getattr(e, "detail", None)
                    if det is not None:
                        err["detail"] = det
                    if not isinstance(e, TTSError):
                        err["traceback"] = traceback.format_exc()[-1200:]
                    write_record(failures_fh, failures_lock, err)
                    with counter_lock:
                        stats["failed"] += 1
                    log(f"  FAIL {job['scene_id']}_{job['idx']}: {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        log("interrupted by user; partial manifest kept")
    finally:
        manifest_fh.close()
        failures_fh.close()

    # dedupe manifest (keep last record per key), sorted
    try:
        recs = read_manifest(mpath)
        dedup = {}
        for r in recs:
            dedup[manifest_key(r)] = r
        ordered = sorted(dedup.values(), key=lambda r: (r.get("card_id") or "",
                                                        r.get("scene_id") or "",
                                                        r.get("idx") or 0))
        write_manifest_atomic(mpath, ordered)
        log(f"manifest: {len(ordered)} records ({len(recs)} raw lines)")
    except Exception as e:  # noqa: BLE001
        log(f"WARN manifest dedupe failed: {e}")

    elapsed = round(time.time() - t_start, 1)
    lats = sorted(stats["latency"])
    lat_stats = {}
    if lats:
        lat_stats = {"avg_s": round(sum(lats) / len(lats), 2),
                     "p50_s": lats[len(lats) // 2],
                     "p95_s": lats[min(len(lats) - 1, int(len(lats) * 0.95))],
                     "max_s": lats[-1]}
    summary = {
        "mode": "mock" if args.mock else "real",
        "model": model,
        "total_selected": len(jobs),
        "skipped": skipped,
        "synthesized": stats["ok"],
        "failed": stats["failed"],
        "audio_seconds": round(stats["dur"], 1),
        "bytes": stats["bytes"],
        "elapsed_s": elapsed,
        "req_per_min": round(stats["ok"] / elapsed * 60, 1) if elapsed else 0,
        "workers": args.workers,
        "rpm_ceiling": args.rpm,
        "latency": lat_stats,
        "references": bool(voice_ref_uri),
        "manifest": rel_to_root(Path(args.manifest)),
        "failures": rel_to_root(Path(args.failures)),
        "log": str(log_path),
    }
    _write_summary(args, log, summary)
    log(f"== done: ok={summary['synthesized']} fail={summary['failed']} "
        f"skipped={skipped} elapsed={elapsed}s ({summary['req_per_min']} req/min) ==")
    log.close()
    return 0 if summary["failed"] == 0 else 2


def _rec_base(job: dict, args, model: str) -> dict:
    return {
        "card_id": job["card_id"],
        "scene_id": job["scene_id"],
        "scene_category": job["scene_category"],
        "idx": job["idx"],
        "user_text": job["user_text"],
        "tone": job["tone"],
        "file": job["file_rel"],
        "duration": None,
        "request_id": None,
        "model": model,
        "tts_prompt": job["tts_prompt"],
    }


def _write_summary(args, log, summary: dict) -> None:
    try:
        path = Path(args.log_dir) / "tts_voice_last_run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        log(f"summary -> {path}")
    except Exception as e:  # noqa: BLE001
        log(f"WARN could not write summary: {e}")


if __name__ == "__main__":
    sys.exit(main())
