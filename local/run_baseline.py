#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoleCraft-ZH 本地模型基线 runner。

逐角色卡、逐场景、逐 turn 调用本地 llama-server（OpenAI 兼容 /v1/chat/completions），
把每个场景的完整多轮回答写入 baselines/<model_slug>/<card_id>.jsonl（一行一场景）。

特点:
  * 场景内串行（历史 assistant 回复回填到后续 turn），场景/角色卡间用线程池并发
  * temperature=0, max_tokens=512, stream=false；每个请求失败重试 3 次
  * 支持断点续跑：已完成的场景自动跳过，出错场景默认重跑（--no-retry-errors 可关闭）
  * 只用标准库，无需额外 pip 依赖

用法:
  # 冒烟（mock 数据）
  python local/run_baseline.py --mock --model-slug qwen3.5-2b --base-url http://127.0.0.1:8088 --limit-cards 3

  # 正式基线
  python local/run_baseline.py --model-slug qwen3.5-2b --base-url http://127.0.0.1:8088 --concurrency 4
  python local/run_baseline.py --model-slug qwen3.5-4b --base-url http://127.0.0.1:8089 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHARACTERS_DIR = PROJECT_ROOT / "data" / "characters"
DEFAULT_SCENES_DIR = PROJECT_ROOT / "data" / "scenes"
DEFAULT_OUT_DIR = PROJECT_ROOT / "baselines"
MOCK_DATA_DIR = PROJECT_ROOT / "local" / "mock_data"

# Windows 下 stdout 被重定向时默认用系统 locale（GBK），这里统一成 UTF-8，避免中文日志乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SYSTEM_PROMPT_TEMPLATE = (
    "你将扮演「{name}」。以下是人设卡：\n{persona_card}\n\n"
    "规则：始终以第一人称、角色身份回复；不要提及AI、模型、系统提示、角色卡或评测；"
    "保持角色性格、价值观、说话风格与人物关系；回复自然口语化，约30-120字，可用少量括号动作/神态。"
)

log = logging.getLogger("run_baseline")

# v0.2 官方解码配置（Qwen3.5 README 非思考文本推荐；注意是 presence_penalty 而非 repetition_penalty）
OFFICIAL_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 2.0,
    "seed": 42,
}

# MiniCPM5-2B 官方推荐采样（README）：temperature=1.0, top_p=0.95, min_p=0.0（llama.cpp 默认 min_p=0.05 会加重读）
# 两个 MiniCPM server（:8090 greedy / :8091 official）在 server 端配置采样；请求只传 temperature/max_tokens，
# 避免请求端采样字段兼容问题（greedy 请求 temperature=0，official 请求 temperature=1.0）。
MINICPM_OFFICIAL_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "min_p": 0.0,
    "seed": 42,
    "server_side": "llama-server flags: --temp 1.0 --top-p 0.95 --min-p 0.0 --seed 42; request only temperature/max_tokens",
}



# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def http_json(url: str, payload: dict | None = None, timeout: float = 300.0) -> dict:
    """GET/POST JSON. POST when payload is not None."""
    data = None
    headers = {"Accept": "application/json", "Connection": "close"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def wait_for_server(base_url: str, timeout_s: float) -> bool:
    """轮询 /health 直到就绪。timeout_s <= 0 时只探测一次。"""
    deadline = time.time() + max(timeout_s, 0)
    attempt = 0
    while True:
        attempt += 1
        try:
            info = http_json(f"{base_url}/health", timeout=10.0)
            if str(info.get("status", "")).lower() == "ok":
                log.info("server ready: %s/health attempt=%d", base_url, attempt)
                return True
            log.warning("server health not ok: %s", info)
        except Exception as exc:  # noqa: BLE001
            log.debug("health attempt %d failed: %s", attempt, exc)
        if time.time() >= deadline:
            return False
        time.sleep(min(2.0, max(0.5, deadline - time.time())))


def chat_completion(
    base_url: str,
    model_name: str,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
    timeout: float,
    retries: int = 3,
    sampling: dict | None = None,
) -> tuple[str, float]:
    """返回 (assistant_text, latency_s)。失败重试 retries 次。"""
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if sampling:
        payload.update(sampling)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            data = http_json(f"{base_url}/v1/chat/completions", payload, timeout=timeout)
            latency = time.perf_counter() - started
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"empty choices: {str(data)[:300]}")
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""
            if not isinstance(text, str):
                text = str(text)
            # 防御：万一模板/服务端把 think 段放进 content，剥掉（Qwen3.5 默认应输出空 think）
            if "</think>" in text:
                text = text.split("</think>")[-1]
            text = text.strip()
            if not text:
                reasoning = msg.get("reasoning_content") or ""
                raise RuntimeError(
                    f"empty content (finish_reason={choices[0].get('finish_reason')!r}, "
                    f"reasoning_len={len(str(reasoning))})"
                )
            return text, round(latency, 3)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError, ValueError, KeyError) as exc:
            last_err = exc
            hint = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    hint = " body=" + exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:  # noqa: BLE001
                    pass
            log.warning("request failed (attempt %d/%d): %s%s", attempt, retries, exc, hint)
            if attempt < retries:
                time.sleep(min(8.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)))
    raise RuntimeError(f"chat completion failed after {retries} attempts: {last_err}")


# --------------------------------------------------------------------------- #
# Data loading / validation
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_card(characters_dir: Path, card_id: str) -> dict:
    path = characters_dir / f"{card_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"character card not found: {path}")
    card = load_json(path)
    if not card.get("name"):
        raise ValueError(f"{path}: missing 'name'")
    persona = card.get("persona_card")
    if not persona or not isinstance(persona, str):
        raise ValueError(f"{path}: missing 'persona_card'")
    return card


def build_system_prompt(card: dict) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(name=str(card["name"]).strip(), persona_card=card["persona_card"].strip())


def validate_scenes(scenes_doc: dict, path: Path) -> list[dict]:
    scenes = scenes_doc.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"{path}: missing/empty 'scenes'")
    for i, scene in enumerate(scenes):
        if not scene.get("scene_id"):
            raise ValueError(f"{path}: scene[{i}] missing scene_id")
        turns = scene.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{path}: scene {scene.get('scene_id')} missing turns")
        for j, turn in enumerate(turns):
            if "user_text" not in turn:
                raise ValueError(f"{path}: scene {scene.get('scene_id')} turn[{j}] missing user_text")
    return scenes


def read_existing_records(path: Path) -> dict[str, dict]:
    """读取已有 jsonl，返回 scene_id -> record。坏行跳过。"""
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
            sid = rec.get("scene_id")
            if sid:
                records[sid] = rec
    return records


def write_records(path: Path, ordered_scene_ids: list[str], records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for sid in ordered_scene_ids:
            rec = records.get(sid)
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Core: one card
# --------------------------------------------------------------------------- #
def run_one_card(
    args: argparse.Namespace,
    card_path: Path,
    stats: dict,
    stats_lock: threading.Lock,
) -> dict:
    card_id = card_path.stem
    out_path = Path(args.out_dir) / args.model_slug / f"{card_id}.jsonl"
    result = {"card_id": card_id, "scenes_total": 0, "scenes_done": 0, "scenes_failed": 0,
              "requests": 0, "turns_done": 0, "turns_failed": 0}
    try:
        card = load_card(Path(args.characters_dir), card_id)
        scenes_path = Path(args.scenes_dir) / f"{card_id}.json"
        if not scenes_path.is_file():
            raise FileNotFoundError(f"scene file not found: {scenes_path}")
        scenes = validate_scenes(load_json(scenes_path), scenes_path)
        if args.limit_scenes and args.limit_scenes > 0:
            scenes = scenes[: args.limit_scenes]
        result["scenes_total"] = len(scenes)
        system_prompt = build_system_prompt(card)
    except Exception as exc:  # noqa: BLE001
        log.error("[%s] 准备失败: %s", card_id, exc)
        result["error"] = str(exc)
        return result

    existing = read_existing_records(out_path)
    todo: list[dict] = []
    for scene in scenes:
        sid = scene["scene_id"]
        rec = existing.get(sid)
        if rec is not None and ("error" not in rec or not args.retry_errors):
            continue
        todo.append(scene)

    log.info("[%s] 场景 %d 个，待跑 %d 个", card_id, len(scenes), len(todo))
    if not todo:
        result["scenes_done"] = sum(1 for s in scenes if s["scene_id"] in existing)
        return result

    for scene in todo:
        sid = scene["scene_id"]
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        turns_out: list[dict] = []
        scene_started = time.perf_counter()
        scene_error: str | None = None
        for turn in scene["turns"]:
            user_text = str(turn["user_text"])
            messages.append({"role": "user", "content": user_text})
            try:
                reply, latency = chat_completion(
                    args.base_url,
                    args.model_name,
                    messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    retries=args.retries,
                    sampling=args.sampling_payload,
                )
                messages.append({"role": "assistant", "content": reply})
                turns_out.append(
                    {"idx": int(turn.get("idx", len(turns_out) + 1)), "user_text": user_text,
                     "assistant_text": reply, "latency_s": latency}
                )
                result["requests"] += 1
                result["turns_done"] += 1
                with stats_lock:
                    stats["turn_latencies"].append(latency)
                    stats["turns"] += 1
            except Exception as exc:  # noqa: BLE001
                scene_error = f"turn {turn.get('idx', '?')} failed: {exc}"
                result["turns_failed"] += 1
                log.error("[%s] 场景 %s %s", card_id, sid, scene_error)
                turns_out.append(
                    {"idx": int(turn.get("idx", len(turns_out) + 1)), "user_text": user_text,
                     "assistant_text": "", "latency_s": None, "error": str(exc)}
                )
                break  # 该场景剩余 turn 无法继续（缺历史回复），整体标记失败
        scene_latency = round(time.perf_counter() - scene_started, 3)
        record = {
            "card_id": card_id,
            "scene_id": sid,
            "model": args.model_slug,
            "generation": dict(args.generation),
            "turns": turns_out,
            "latency_s": scene_latency,
        }
        if scene_error:
            record["error"] = scene_error
            result["scenes_failed"] += 1
        else:
            result["scenes_done"] += 1
        existing[sid] = record
        write_records(out_path, [s["scene_id"] for s in scenes], existing)
        with stats_lock:
            stats["requests"] += len([t for t in turns_out if t.get("assistant_text")])
            stats["scene_latencies"].append(scene_latency)
            if scene_error:
                stats["scene_errors"] += 1

    return result


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoleCraft-ZH 本地基线 runner（llama-server OpenAI 兼容接口）")
    parser.add_argument("--model-slug", required=True, help="模型标识，如 qwen3.5-2b / qwen3.5-4b（决定输出目录）")
    parser.add_argument("--model-name", default=None, help="请求里的 model 字段（默认与 slug 相同）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088", help="llama-server 地址")
    parser.add_argument("--characters-dir", default=str(DEFAULT_CHARACTERS_DIR))
    parser.add_argument("--scenes-dir", default=str(DEFAULT_SCENES_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--mock", action="store_true", help="使用 local/mock_data 冒烟数据（覆盖 characters/scenes 目录）")
    parser.add_argument("--cards", default=None, help="只跑指定卡，逗号分隔，如 rc01_xxx,rc02_yyy")
    parser.add_argument("--limit-cards", type=int, default=0, help="只跑前 N 张卡（0=全部）")
    parser.add_argument("--limit-scenes", type=int, default=0, help="每卡只跑前 N 个场景（0=全部）")
    parser.add_argument("--concurrency", type=int, default=4, help="并发角色卡数（≈ llama-server -np）")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sampling", choices=["greedy", "official", "minicpm-official"], default="greedy",
                        help="解码配置：greedy=v0.1 贪心；official=Qwen3.5 官方非思考推荐"
                             "(temperature=1.0/top_p=1.0/top_k=20/min_p=0.0/presence_penalty=2.0/seed=42)；"
                             "minicpm-official=MiniCPM5 官方推荐(temp=1.0/top_p=0.95/min_p=0.0/seed=42，"
                             "由 server 端 flag 生效，请求只传 temperature/max_tokens)")
    parser.add_argument("--timeout", type=float, default=300.0, help="单请求超时秒")
    parser.add_argument("--retries", type=int, default=3, help="单请求失败重试次数")
    parser.add_argument("--wait-server", type=float, default=60.0, help="启动前等待 /health 的秒数（0=不等待）")
    parser.add_argument("--no-retry-errors", dest="retry_errors", action="store_false", help="续跑时不重跑上次失败的场景")
    parser.add_argument("--dry-run", action="store_true", help="只校验数据/打印计划，不发请求")
    parser.add_argument("--log-file", default=None, help="额外写日志文件（logs/local/...）")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    if args.mock:
        args.characters_dir = str(MOCK_DATA_DIR / "characters")
        args.scenes_dir = str(MOCK_DATA_DIR / "scenes")
        if args.out_dir == str(DEFAULT_OUT_DIR):
            args.out_dir = str(MOCK_DATA_DIR / "baselines")
    if args.model_name is None:
        args.model_name = args.model_slug
    if args.sampling == "official":
        args.temperature = OFFICIAL_SAMPLING["temperature"]
        # 显式带上官方采样字段（llama-server 支持这些 OAI 字段；server 端 flag 也设了同值兜底）
        args.sampling_payload = {k: v for k, v in OFFICIAL_SAMPLING.items() if k != "temperature"}
        args.generation = {**OFFICIAL_SAMPLING, "max_tokens": args.max_tokens}
    elif args.sampling == "minicpm-official":
        args.temperature = MINICPM_OFFICIAL_SAMPLING["temperature"]
        # 采样全部由 :8091 server 端 flag 负责，请求只传 temperature/max_tokens
        args.sampling_payload = None
        args.generation = {**MINICPM_OFFICIAL_SAMPLING, "max_tokens": args.max_tokens,
                           "request_params": ["temperature", "max_tokens"]}
    else:
        args.sampling_payload = None
        args.generation = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    return args


def setup_logging(args: argparse.Namespace) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args)

    characters_dir = Path(args.characters_dir).resolve()
    scenes_dir = Path(args.scenes_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not characters_dir.is_dir():
        log.error("characters 目录不存在: %s", characters_dir)
        return 2
    if not scenes_dir.is_dir():
        log.error("scenes 目录不存在: %s", scenes_dir)
        return 2

    card_paths = sorted(characters_dir.glob("*.json"))
    if args.cards:
        wanted = {c.strip() for c in args.cards.split(",") if c.strip()}
        card_paths = [p for p in card_paths if p.stem in wanted]
        missing = wanted - {p.stem for p in card_paths}
        if missing:
            log.error("指定的卡不存在: %s", ", ".join(sorted(missing)))
            return 2
    if args.limit_cards and args.limit_cards > 0:
        card_paths = card_paths[: args.limit_cards]
    if not card_paths:
        log.error("没有找到任何角色卡（%s/*.json）。如果数据未就绪，可先加 --mock 冒烟。", characters_dir)
        return 2

    # 只保留 scenes 已就绪的卡（数据可能分批生成）
    ready: list[Path] = []
    for p in card_paths:
        if (scenes_dir / f"{p.stem}.json").is_file():
            ready.append(p)
        else:
            log.warning("跳过 %s：scenes/%s.json 不存在", p.stem, p.stem)
    if not ready:
        log.error("scenes 目录 %s 下没有任何就绪的场景文件", scenes_dir)
        return 2
    card_paths = ready

    total_scenes = 0
    for p in card_paths:
        try:
            doc = load_json(scenes_dir / f"{p.stem}.json")
            n = len(doc.get("scenes") or [])
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] 读取场景文件失败: %s", p.stem, exc)
            continue
        total_scenes += n
    log.info(
        "=%s | 卡 %d 张 | 场景 %d 个 | server %s | sampling=%s | concurrency=%d | 输出 %s",
        args.model_slug, len(card_paths), total_scenes, args.base_url, args.sampling, args.concurrency,
        out_dir / args.model_slug,
    )

    if args.dry_run:
        for p in card_paths:
            log.info("[dry-run] %s -> %s", p.stem, out_dir / args.model_slug / f"{p.stem}.jsonl")
        return 0

    if args.wait_server and args.wait_server > 0:
        if not wait_for_server(args.base_url, args.wait_server):
            log.error("llama-server 未就绪: %s/health（检查 serve.ps1 是否已启动）", args.base_url)
            return 2
    else:
        try:
            http_json(f"{args.base_url}/health", timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            log.error("连不上 llama-server %s: %s", args.base_url, exc)
            return 2

    stats = {"requests": 0, "turns": 0, "scene_errors": 0, "scene_latencies": [], "turn_latencies": [], "cards_errors": []}
    stats_lock = threading.Lock()
    started_at = datetime.now()
    started = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(run_one_card, args, p, stats, stats_lock): p for p in card_paths}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.exception("[%s] 未捕获异常", p.stem)
                res = {"card_id": p.stem, "error": str(exc), "scenes_total": 0, "scenes_done": 0, "scenes_failed": 0, "requests": 0}
            results.append(res)
            if res.get("error"):
                with stats_lock:
                    stats["cards_errors"].append({"card_id": res["card_id"], "error": res["error"]})
            log.info(
                "[%s] done: scenes %d/%d, failed %d",
                res["card_id"], res.get("scenes_done", 0), res.get("scenes_total", 0), res.get("scenes_failed", 0),
            )

    elapsed = time.perf_counter() - started
    lat = sorted(stats["scene_latencies"])
    tlat = sorted(stats["turn_latencies"])

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

        def _pct(p: float) -> float:
            idx = min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1))))
            return round(values[idx], 3)

        return {"mean": round(sum(values) / len(values), 3), "p50": _pct(50), "p95": _pct(95), "max": round(values[-1], 3)}

    finished_at = datetime.now()
    manifest = {
        "model": args.model_slug,
        "model_name": args.model_name,
        "base_url": args.base_url,
        "characters_dir": str(characters_dir),
        "scenes_dir": str(scenes_dir),
        "cards": len(card_paths),
        "scenes_total": total_scenes,
        "scenes_ok": sum(r.get("scenes_done", 0) for r in results),
        "scenes_failed": sum(r.get("scenes_failed", 0) for r in results),
        "cards_failed": len(stats["cards_errors"]),
        "requests": stats["requests"],
        "turns": stats["turns"],
        "turns_failed": sum(r.get("turns_failed", 0) for r in results),
        "elapsed_s": round(elapsed, 2),
        "scene_latency_s": _stats(lat),
        "turn_latency_s": _stats(tlat),
        "avg_turn_latency_s": round(sum(tlat) / len(tlat), 3) if tlat else 0.0,
        "manifest": "",
        "sampling": args.sampling,
        "generation": dict(args.generation),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "cards_errors": stats["cards_errors"],
    }
    manifest["manifest"] = str((out_dir / args.model_slug / "_manifest.json").relative_to(PROJECT_ROOT)) \
        if (out_dir / args.model_slug / "_manifest.json").is_relative_to(PROJECT_ROOT) \
        else str(out_dir / args.model_slug / "_manifest.json")
    manifest_path = out_dir / args.model_slug / "_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info(
        "全部完成: %d 场景 ok / %d 失败 | turn %d ok / %d 失败 | 请求 %d | 用时 %.1fs | 场景延迟 mean=%ss p95=%ss | 每轮延迟 mean=%ss p50=%ss p95=%ss | manifest: %s",
        manifest["scenes_ok"], manifest["scenes_failed"], manifest["turns"], manifest["turns_failed"],
        manifest["requests"], elapsed,
        manifest["scene_latency_s"]["mean"], manifest["scene_latency_s"]["p95"],
        manifest["turn_latency_s"]["mean"], manifest["turn_latency_s"]["p50"], manifest["turn_latency_s"]["p95"],
        manifest_path,
    )
    return 0 if manifest["scenes_failed"] == 0 and manifest["cards_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
