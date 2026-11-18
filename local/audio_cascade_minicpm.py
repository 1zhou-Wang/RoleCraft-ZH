#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MiniCPM5-2B 音频 track B1/B2 级联 runner（greedy :8090 / official :8091）。

复用 audio_cascade.py 的场景/turn 级联逻辑与 schema，仅替换模型配置与输出目录：
  * greedy  → baselines/audio_pipeline_minicpm5-2b-greedy/   (server :8090，请求 temperature=0)
  * official→ baselines/audio_pipeline_minicpm5-2b-official/ (server :8091，请求 temperature=1.0；
              server 端 flag: --temp 1.0 --top-p 0.95 --min-p 0.0 --seed 42)
  * ASR 转写复用 data/audio_track/asr_transcripts.jsonl（不重跑）

用法:
  python local/audio_cascade_minicpm.py --modes greedy,official --concurrency 4
  python local/audio_cascade_minicpm.py --modes official --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import audio_cascade as ac  # noqa: E402  复用级联主逻辑

LOG_DIR = ROOT / "logs" / "minicpm5"
BASELINE_SLUG = "minicpm5-2b"
MODES = {
    "greedy": {"port": 8090, "pipeline": "minicpm5-greedy", "temperature": 0.0},
    "official": {"port": 8091, "pipeline": "minicpm5-official", "temperature": 1.0},
}
SERVER_SAMPLING_NOTE = {
    "greedy": "请求 temperature=0（greedy）；server :8090 无采样 flag",
    "official": "server :8091 flags: --temp 1.0 --top-p 0.95 --min-p 0.0 --seed 42；请求只传 temperature=1.0/max_tokens",
}
MINICPM_OFFICIAL = {"temperature": 1.0, "top_p": 0.95, "min_p": 0.0, "seed": 42}


def build_namespace(mode: str, a: argparse.Namespace) -> argparse.Namespace:
    cfg = MODES[mode]
    if mode == "official":
        generation = {**MINICPM_OFFICIAL, "max_tokens": a.max_tokens, "mode": mode,
                      "server_side": SERVER_SAMPLING_NOTE[mode]}
        sampling = "minicpm-official"
    else:
        generation = {"temperature": 0.0, "max_tokens": a.max_tokens, "mode": mode,
                      "server_side": SERVER_SAMPLING_NOTE[mode]}
        sampling = "greedy"
    return argparse.Namespace(
        cards=a.cards,
        limit_cards=a.limit_cards,
        limit_scenes=a.limit_scenes,
        concurrency=a.concurrency,
        max_tokens=a.max_tokens,
        temperature=cfg["temperature"],
        timeout=a.timeout,
        retries=a.retries,
        wait_server=a.wait_server,
        dry_run=a.dry_run,
        sampling=sampling,
        sampling_payload=None,
        generation=generation,
        out_suffix="",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MiniCPM5-2B 音频级联 runner（greedy/official）")
    ap.add_argument("--modes", default="greedy,official", help="greedy,official 逗号分隔")
    ap.add_argument("--cards", default=None)
    ap.add_argument("--limit-cards", type=int, default=0)
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--wait-server", type=float, default=60.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-file", default=str(LOG_DIR / "audio_cascade_minicpm.log"))
    a = ap.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(a.log_file, encoding="utf-8")],
        force=True,
    )
    log = logging.getLogger("audio_cascade_minicpm")

    modes = [m.strip().lower() for m in a.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            log.error("未知 mode: %s（可选 greedy/official）", m)
            return 2

    ac.PIPELINES = {
        m: {"slug": f"{BASELINE_SLUG}-{m}", "pipeline": MODES[m]["pipeline"],
            "base_url": f"http://127.0.0.1:{MODES[m]['port']}"}
        for m in MODES
    }

    summaries = []
    rc = 0
    for mode in modes:
        log.info("=== MiniCPM5-2B 级联 mode=%s (port %d) ===", mode, MODES[mode]["port"])
        ns = build_namespace(mode, a)
        summary = ac.run_pipeline(ns, mode)
        summaries.append(summary)
        if not a.dry_run and (summary.get("scenes_failed") or summary.get("card_errors")):
            rc = 1
    if not a.dry_run:
        out = LOG_DIR / "cascade_summary.json"
        out.write_text(json.dumps({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model": BASELINE_SLUG,
            "modes": MODES,
            "pipelines": summaries,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("cascade summary -> %s", out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
