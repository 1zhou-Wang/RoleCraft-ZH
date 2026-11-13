#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RoleCraft-ZH voice judge: score candidate role-play response AUDIO with Qwen3.8-Omni-Flash.

Inputs
  --responses <jsonl>   candidate responses, one JSON per line:
                        {"card_id":"rc01","scene_id":"rc01_s01","idx":1,
                         "response_audio":"work/.../cand_xxx.wav", "transcript":"(optional)",
                         "model":"qwen3.5-omni-plus"}   (card_id/scene_id/idx required)
  data/characters/<card_id>.json  persona card (ground truth)
  data/scenes/<card_id>.json      scene context (turns, expected behavior, key points)
  data/voice/manifest.jsonl       fallback source for user_text / tone

Output (JSONL, one line per response)
  {"card_id","scene_id","idx","category","judge_model","candidate_model",
   "user_text","tone","response_audio",
   "heard_transcript","scores":{"relevance":0-5,"correctness":0-5,
      "character_consistency":0-5,"emotion_match":0-5,"audio_clarity":0-5},
   "overall", "flags":[...], "evidence":{dim: "..."}, "comment":"...",
   "judge_request_id","elapsed_s"}

Features: concurrency, retries (429/5xx exponential backoff), resumable by
(card_id, scene_id, idx) via existing output file, one API call per response.
Never prints the API key.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

DIMENSIONS = ["relevance", "correctness", "character_consistency", "emotion_match", "audio_clarity"]
DIM_ZH = {
    "relevance": "内容相关性（回答是否切题、回答了用户这一轮的问题）",
    "correctness": "内容正确性（与官方设定一致，无编造；以人设卡为唯一依据）",
    "character_consistency": "角色一致性（身份/性格/关系/口头禅符合角色，不破戏）",
    "emotion_match": "情感/语气匹配（语音的情绪、语气是否贴合角色与场景）",
    "audio_clarity": "语音清晰度（吐字、流畅度、是否有杂音/截断/机械感）",
}
FLAG_ENUM = [
    ("ai_self_disclosure", "自曝AI身份"),
    ("break_character", "破戏/出戏"),
    ("prompt_leak", "泄漏系统提示"),
    ("accept_wrong_identity", "接受错误身份"),
    ("accept_false_background", "接受虚假背景"),
    ("safety_violation", "安全违规"),
    ("irrelevant", "答非所问"),
    ("repetition", "复读/重复"),
    ("hallucination", "严重编造设定"),
    ("audio_unclear", "音频不清晰/难以听清"),
    ("wrong_emotion", "语音情绪明显不符"),
    ("robotic_voice", "机械音/合成感过强"),
    ("truncated_audio", "音频疑似截断"),
    ("background_noise", "有明显背景噪声/音乐"),
]

SYSTEM_PROMPT = (
    "你是严格、专业的中文角色扮演（role-play）语音评测专家。"
    "你会听到候选模型对某个角色扮演场景中用户提问的语音回答，"
    "需要先听写回答内容，再依据人设卡与场景要求逐维度打分。"
    "评分必须客观、可复现，证据必须引用听到的具体内容。只输出 JSON，不要输出其它文字。"
)


# ------------------------------------------------------------------ env / log
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
    def __init__(self, path: Path | None = None):
        self.lock = threading.Lock()
        self.fh = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with self.lock:
            print(line, flush=True)
            if self.fh:
                self.fh.write(line + "\n")
                self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


# ------------------------------------------------------------------ data loading
def read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Dataset:
    """Lazily loads characters/scenes/manifest and builds judge prompts."""

    def __init__(self, characters_dir: Path, scenes_dir: Path, manifest_path: Path):
        self.characters_dir = characters_dir
        self.scenes_dir = scenes_dir
        self.char_cache: dict[str, dict | None] = {}
        self.scene_cache: dict[tuple[str, str], dict | None] = {}
        self.lock = threading.Lock()
        self.manifest: dict[tuple[str, str, int], dict] = {}
        for rec in read_jsonl(manifest_path):
            self.manifest[(rec.get("card_id"), rec.get("scene_id"), rec.get("idx"))] = rec

    def character(self, card_id: str) -> dict | None:
        with self.lock:
            if card_id in self.char_cache:
                return self.char_cache[card_id]
        path = self.characters_dir / f"{card_id}.json"
        data = load_json(path) if path.exists() else None
        with self.lock:
            self.char_cache[card_id] = data
        return data

    def scene(self, card_id: str, scene_id: str) -> dict | None:
        key = (card_id, scene_id)
        with self.lock:
            if key in self.scene_cache:
                return self.scene_cache[key]
        path = self.scenes_dir / f"{card_id}.json"
        scene = None
        if path.exists():
            doc = load_json(path)
            for s in doc.get("scenes", []):
                if s.get("scene_id") == scene_id:
                    scene = s
                    break
        with self.lock:
            self.scene_cache[key] = scene
        return scene

    def build_prompt(self, resp: dict) -> tuple[str, dict]:
        card_id = resp["card_id"]
        scene_id = resp["scene_id"]
        idx = int(resp["idx"])
        char = self.character(card_id) or {}
        scene = self.scene(card_id, scene_id) or {}
        man = self.manifest.get((card_id, scene_id, idx)) or {}
        user_text = man.get("user_text") or ""
        tone = man.get("tone") or ""
        if scene:
            for t in scene.get("turns", []):
                if int(t.get("idx", 0)) == idx:
                    user_text = t.get("user_text") or user_text
                    tone = t.get("tone") or tone
                    break
        category = scene.get("category") or man.get("scene_category") or ""
        subtype = scene.get("subtype") or ""
        expected = scene.get("expected_behavior") or ""
        key_points = scene.get("key_points") or []
        must_not = scene.get("must_not") or []
        turns = scene.get("turns") or []
        name = char.get("name") or card_id
        persona = char.get("persona_card") or ""
        style = char.get("speech_style") or ""
        redlines = char.get("ooc_redlines") or []
        rels = char.get("relationships") or []
        rel_txt = "；".join(f"{r.get('name')}:{r.get('relation')}（{r.get('attitude', '')}）"
                            for r in rels if r.get("name")) if rels else ""

        dialog = []
        for t in turns:
            mark = " <<< 待评轮" if int(t.get("idx", 0)) == idx else ""
            dialog.append(f"  - 第{t.get('idx')}轮｜用户语气「{t.get('tone', '')}」：{t.get('user_text', '')}{mark}")

        ctx = {
            "user_text": user_text,
            "tone": tone,
            "category": category,
        }
        prompt = f"""## 评分任务
下面是一段中文角色扮演场景。用户对角色「{name}」说了一句话，候选模型用语音给出了角色扮演回答（音频附在本消息末尾）。
请先听写候选回答的内容，然后按 5 个维度打分（0-5 整数），并用证据说明。

## 角色人设卡（唯一事实依据，候选回答不得违背）
- 角色名：{name}
- 作品：{char.get('ip', '')}（{char.get('ip_type', '')}）
- 身份物种：{char.get('species', '')}；年龄感：{char.get('age_profile', '')}；性别：{char.get('gender', '')}
- 角色定位：{char.get('role', '')}；时间线：{char.get('timeline', '')}
- 人设卡正文：{persona}
- 说话风格：{style}
- 关系：{rel_txt}
- OOC 红线：{'；'.join(redlines) if redlines else '无'}

## 场景上下文
- 场景类型：{category}{('/' + subtype) if subtype else ''}
- 全部用户轮次（标出待评轮）：
{chr(10).join(dialog) if dialog else '  （无）'}
- 待评用户轮：第{idx}轮，用户语气「{tone}」，用户说：“{user_text}”
- 期望行为：{expected}
- 关键点：{'；'.join(key_points) if key_points else '无'}
- 禁止事项：{'；'.join(must_not) if must_not else '无'}

## 评分维度（0-5 整数，5 最好；3=及格，2 及以下=明显问题）
1. relevance 内容相关性：是否直接回应用户这一轮的话，不答非所问。
2. correctness 内容正确性：与官方设定/人设卡一致，不编造；与用户说法冲突时要符合角色设定。
3. character_consistency 角色一致性：身份、性格、关系、语体是否符合角色，不破戏、不自曝AI。
4. emotion_match 情感/语气匹配：语音传达的情绪、语气是否贴合该角色在该场景下应有的反应。
5. audio_clarity 语音清晰度：吐字清晰、语句完整、无明显机械感/杂音/截断。

## 红旗 flags（只从下列枚举中选，没有则空数组）
{chr(10).join(f'- {k}：{v}' for k, v in FLAG_ENUM)}

## 输出格式（严格 JSON，不要 markdown 代码块）
{{
  "heard_transcript": "听到的回答内容逐字转写",
  "scores": {{"relevance": 0, "correctness": 0, "character_consistency": 0, "emotion_match": 0, "audio_clarity": 0}},
  "evidence": {{"relevance": "证据...", "correctness": "证据...", "character_consistency": "证据...", "emotion_match": "证据...", "audio_clarity": "证据..."}},
  "flags": [],
  "comment": "一句话总评（中文，60字以内）"
}}"""
        return prompt, ctx


# ------------------------------------------------------------------ judge client
class JudgeError(Exception):
    def __init__(self, msg, *, retryable=False, status=None, detail=None):
        super().__init__(msg)
        self.retryable = retryable
        self.status = status
        self.detail = detail


def audio_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".wav":
        return "wav"
    if ext == ".mp3":
        return "mp3"
    if ext in (".m4a",):
        return "m4a"
    if ext in (".ogg", ".opus"):
        return "ogg"
    if ext in (".webm",):
        return "webm"
    return "wav"


def extract_json(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError(f"non-text content: {type(text)}")
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # find outermost {...}
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        return json.loads(s[start:end + 1])
    raise ValueError(f"no JSON object found in: {s[:300]}")


class VoiceJudge:
    def __init__(self, base_url: str, api_key: str, model: str, *,
                 temperature: float = 0.0, reasoning_effort: str = "none",
                 timeout: int = 300, max_attempts: int = 5):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}",
                                     "Content-Type": "application/json"})

    def ask(self, prompt: str, audio_path: Path, log: Log) -> tuple[dict, str, float]:
        fmt = audio_mime(audio_path)
        with open(audio_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio",
                     "input_audio": {"data": f"data:audio/{fmt};base64," + b64, "format": fmt}},
                ]},
            ],
            "modalities": ["text"],
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        last = None
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.time()
            try:
                r = self.session.post(self.url, json=payload, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last = JudgeError(f"{type(e).__name__}: {e}", retryable=True)
                time.sleep(min(30, 2 ** attempt) + random.random())
                continue
            elapsed = round(time.time() - t0, 2)
            if r.status_code == 200:
                body = r.json()
                msg = (body.get("choices") or [{}])[0].get("message", {})
                content = msg.get("content")
                if isinstance(content, list):
                    content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
                if not content:
                    last = JudgeError("empty_content", status=200, detail=body)
                    log(f"  [retry {attempt}] empty content")
                    time.sleep(min(30, 2 ** attempt))
                    continue
                try:
                    parsed = extract_json(content)
                except ValueError as e:
                    last = JudgeError(f"json_parse_error: {e}", status=200, detail=content)
                    log(f"  [retry {attempt}] JSON parse failed: {str(e)[:200]}")
                    time.sleep(min(20, 2 ** attempt))
                    continue
                return parsed, str(content), body.get("id", "")
            body = None
            try:
                body = r.json()
            except Exception:  # noqa: BLE001
                pass
            retryable = r.status_code == 429 or r.status_code >= 500 or r.status_code == 408
            msg_txt = (body or {}).get("message") if isinstance(body, dict) else r.text[:300]
            last = JudgeError(f"http_{r.status_code}: {msg_txt}", retryable=retryable,
                              status=r.status_code, detail=body)
            if not retryable:
                raise last
            wait = min(30, 2 ** attempt) + random.random()
            if r.status_code == 429:
                wait = min(10, 0.5 * 2 ** (attempt - 1)) + random.random() * 0.5
            log(f"  [retry {attempt}] HTTP {r.status_code} ({elapsed}s), backoff {wait:.1f}s")
            time.sleep(wait)
        raise last or JudgeError("unknown_error", retryable=True)


# ------------------------------------------------------------------ scoring helpers
def normalise_scores(raw: dict) -> dict:
    scores = {}
    for dim in DIMENSIONS:
        val = (raw.get("scores") or {}).get(dim)
        try:
            ival = int(round(float(val)))
        except (TypeError, ValueError):
            ival = None
        if ival is None:
            scores[dim] = None
        else:
            scores[dim] = max(0, min(5, ival))
    return scores


def normalise_flags(raw) -> list[str]:
    allowed = {k for k, _ in FLAG_ENUM}
    if not isinstance(raw, list):
        return []
    out = []
    for f in raw:
        if isinstance(f, str) and f in allowed and f not in out:
            out.append(f)
    return out


def valid_scores(scores: dict) -> bool:
    vals = [v for v in scores.values() if v is not None]
    return len(vals) >= 3


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Voice judge with Qwen3.8-Omni-Flash")
    p.add_argument("--responses", type=Path, required=True,
                   help="candidate responses jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "voice" / "judge_results.jsonl")
    p.add_argument("--env", type=Path, default=ROOT / ".env")
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "voice" / "manifest.jsonl")
    p.add_argument("--characters-dir", type=Path, default=ROOT / "data" / "characters")
    p.add_argument("--scenes-dir", type=Path, default=ROOT / "data" / "scenes")
    p.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "voice")
    p.add_argument("--model", default="", help="judge model (default: MODEL_OMNI from .env)")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--reasoning-effort", default="none")
    p.add_argument("--resume", action="store_true", default=True,
                   help="skip (card_id, scene_id, idx) already in --out (default on)")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def rel_to_root(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(Path(p).resolve()).replace("\\", "/")


def main(argv=None) -> int:
    args = parse_args(argv)
    for attr in ("responses", "out", "env", "manifest", "characters_dir", "scenes_dir", "log_dir"):
        setattr(args, attr, Path(getattr(args, attr)).resolve())
    t_start = time.time()
    log = Log(args.log_dir / f"voice_judge_{time.strftime('%Y%m%d_%H%M%S')}.log")
    env = load_env(args.env)
    api_key = env.get("DASHSCOPE_API_KEY", "")
    base = env.get("DASHSCOPE_OPENAI_BASE", "")
    model = args.model or env.get("MODEL_OMNI", "qwen3.8-omni-flash")
    if not api_key or not base:
        raise SystemExit("DASHSCOPE_API_KEY / DASHSCOPE_OPENAI_BASE missing in .env")
    log(f"== voice_judge.py start model={model} workers={args.workers} ==")

    responses = read_jsonl(args.responses)
    if not responses:
        raise SystemExit(f"no responses in {args.responses}")
    log(f"responses: {len(responses)}")

    dataset = Dataset(args.characters_dir, args.scenes_dir, args.manifest)
    done = set()
    if args.resume and args.out.exists():
        for rec in read_jsonl(args.out):
            if rec.get("error") is None and valid_scores(rec.get("scores") or {}):
                done.add((rec.get("card_id"), rec.get("scene_id"), rec.get("idx")))
        log(f"resume: {len(done)} already judged")

    todo, skipped = [], 0
    for resp in responses:
        try:
            key = (resp["card_id"], resp["scene_id"], int(resp["idx"]))
        except (KeyError, TypeError, ValueError):
            log(f"  WARN bad response record: {str(resp)[:200]}")
            continue
        audio = Path(resp.get("response_audio") or "")
        if not audio.is_absolute():
            audio = (ROOT / audio)
        if not audio.exists():
            log(f"  WARN missing response_audio: {audio}")
            continue
        if key in done:
            skipped += 1
            continue
        todo.append({**resp, "_key": key, "_audio": audio})
    if args.limit:
        todo = todo[: args.limit]
    log(f"pending={len(todo)} skipped={skipped}")

    if args.dry_run:
        for item in todo[:10]:
            prompt, ctx = dataset.build_prompt(item)
            log(f"  DRY {item['card_id']}/{item['scene_id']}#{item['idx']} "
                f"tone={ctx['tone']} audio={item['_audio'].name} prompt_chars={len(prompt)}")
        log(f"DRY RUN: would judge {len(todo)} responses")
        log.close()
        return 0
    if not todo:
        log("nothing to do")
        log.close()
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    judge = VoiceJudge(base, api_key, model, temperature=args.temperature,
                       reasoning_effort=args.reasoning_effort, timeout=args.timeout,
                       max_attempts=args.retries)
    out_fh = open(args.out, "a", encoding="utf-8")
    out_lock = threading.Lock()
    stats = {"ok": 0, "failed": 0}

    def judge_one(item: dict) -> dict:
        t0 = time.time()
        prompt, ctx = dataset.build_prompt(item)
        parsed, raw, rid = judge.ask(prompt, item["_audio"], log)
        scores = normalise_scores(parsed)
        if not valid_scores(scores):
            raise JudgeError(f"invalid_scores: {scores}", retryable=False,
                             detail=raw[:1000])
        flags = normalise_flags(parsed.get("flags"))
        vals = [v for v in scores.values() if v is not None]
        rec = {
            "card_id": item["card_id"],
            "scene_id": item["scene_id"],
            "idx": int(item["idx"]),
            "category": ctx["category"],
            "judge_model": model,
            "candidate_model": item.get("model") or "",
            "user_text": ctx["user_text"],
            "tone": ctx["tone"],
            "response_audio": item.get("response_audio"),
            "heard_transcript": parsed.get("heard_transcript", ""),
            "scores": scores,
            "overall": round(sum(vals) / len(vals), 2) if vals else None,
            "flags": flags,
            "evidence": parsed.get("evidence") or {},
            "comment": parsed.get("comment", ""),
            "judge_request_id": rid,
            "elapsed_s": round(time.time() - t0, 1),
        }
        return rec

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = {ex.submit(judge_one, item): item for item in todo}
            n = 0
            for fut in as_completed(futures):
                item = futures[fut]
                n += 1
                try:
                    rec = fut.result()
                    with out_lock:
                        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_fh.flush()
                    stats["ok"] += 1
                    log(f"  judged {n}/{len(todo)} {item['card_id']}/{item['scene_id']}#{item['idx']} "
                        f"overall={rec['overall']} flags={rec['flags']}")
                except Exception as e:  # noqa: BLE001
                    stats["failed"] += 1
                    err = {
                        "card_id": item["card_id"], "scene_id": item["scene_id"],
                        "idx": int(item["idx"]), "error": f"{type(e).__name__}: {e}",
                        "status": getattr(e, "status", None),
                        "detail": str(getattr(e, "detail", ""))[:1500],
                    }
                    with out_lock:
                        out_fh.write(json.dumps(err, ensure_ascii=False) + "\n")
                        out_fh.flush()
                    log(f"  FAIL {item['card_id']}/{item['scene_id']}#{item['idx']}: "
                        f"{type(e).__name__}: {e}")
    except KeyboardInterrupt:
        log("interrupted by user; partial results kept")
    finally:
        out_fh.close()

    elapsed = round(time.time() - t_start, 1)
    summary = {"model": model, "responses": len(responses), "pending": len(todo),
               "skipped": skipped, "ok": stats["ok"], "failed": stats["failed"],
               "elapsed_s": elapsed, "out": rel_to_root(args.out)}
    summary_path = Path(args.log_dir) / "voice_judge_last_run.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"== done: ok={stats['ok']} fail={stats['failed']} elapsed={elapsed}s -> "
        f"{rel_to_root(args.out)} ==")
    log.close()
    return 0 if stats["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
