# -*- coding: utf-8 -*-
"""Detect severe repetition / looping badcases across all RoleCraft-ZH tracks, compare the same test item
across models, and build a report for training staff.

Tiers:
  severe    : 明显死循环/退化（字符长串、连续重复单元、极低唯一字符率/熵、连续重复句）
  moderate  : 疑似复读/退化（较弱指标）
  suspected : 程序化未命中，但 DeepSeek 判官打了 repetition flag 且文本较长（仅统计，不进明细）

Outputs:
  results/badcases/repetition_flags.jsonl   # backend records (severe/moderate/suspected)
  results/badcases/summary.json             # per-model counts + cross-model item matrix
  results/badcases/report.md                # report for training staff
Usage: python scripts/detect_repetition_badcases.py
"""
import json, re, zlib, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE = ROOT / "baselines"
RES = ROOT / "results"
OUT = RES / "badcases"

MODELS = [
    ("text", "qwen3.5-2b", BASE / "qwen3.5-2b", RES / "qwen3.5-2b" / "judge"),
    ("text", "qwen3.5-4b", BASE / "qwen3.5-4b", RES / "qwen3.5-4b" / "judge"),
    ("text", "edgeinstant-omni-2b-text", BASE / "edgeinstant-omni-2b-text", RES / "edgeinstant-omni-2b-text" / "judge"),
    ("text", "qwen3.5-2b-official", BASE / "qwen3.5-2b-official", RES / "qwen3.5-2b-official" / "judge"),
    ("text", "qwen3.5-4b-official", BASE / "qwen3.5-4b-official", RES / "qwen3.5-4b-official" / "judge"),
    ("text", "minicpm5-2b-greedy", BASE / "minicpm5-2b-greedy", RES / "minicpm5-2b-greedy" / "judge"),
    ("text", "minicpm5-2b-official", BASE / "minicpm5-2b-official", RES / "minicpm5-2b-official" / "judge"),
    ("audio", "audio_pipeline_qwen3.5-2b", BASE / "audio_pipeline_qwen3.5-2b", RES / "audio_pipeline_qwen3.5-2b" / "judge"),
    ("audio", "audio_pipeline_qwen3.5-4b", BASE / "audio_pipeline_qwen3.5-4b", RES / "audio_pipeline_qwen3.5-4b" / "judge"),
    ("audio", "audio_pipeline_qwen3.5-2b-official", BASE / "audio_pipeline_qwen3.5-2b-official", RES / "audio_pipeline_qwen3.5-2b-official" / "judge"),
    ("audio", "audio_pipeline_qwen3.5-4b-official", BASE / "audio_pipeline_qwen3.5-4b-official", RES / "audio_pipeline_qwen3.5-4b-official" / "judge"),
    ("audio", "audio_pipeline_minicpm5-2b-greedy", BASE / "audio_pipeline_minicpm5-2b-greedy", RES / "audio_pipeline_minicpm5-2b-greedy" / "judge"),
    ("audio", "audio_pipeline_minicpm5-2b-official", BASE / "audio_pipeline_minicpm5-2b-official", RES / "audio_pipeline_minicpm5-2b-official" / "judge"),
    ("audio", "audio_edgeinstant", BASE / "audio_edgeinstant", RES / "audio_edgeinstant" / "judge"),
]
LABEL = {
    "qwen3.5-2b": "Qwen3.5-2B（文字·greedy）",
    "qwen3.5-4b": "Qwen3.5-4B（文字·greedy）",
    "edgeinstant-omni-2b-text": "EdgeInstant-Omni-2B（文字·greedy）",
    "qwen3.5-2b-official": "Qwen3.5-2B（文字·官方采样）",
    "qwen3.5-4b-official": "Qwen3.5-4B（文字·官方采样）",
    "minicpm5-2b-greedy": "MiniCPM5-2B（文字·greedy）",
    "minicpm5-2b-official": "MiniCPM5-2B（文字·官方采样）",
    "audio_pipeline_qwen3.5-2b": "ASR+2B+TTS 级联（greedy）",
    "audio_pipeline_qwen3.5-4b": "ASR+4B+TTS 级联（greedy）",
    "audio_pipeline_qwen3.5-2b-official": "ASR+2B+TTS 级联（官方采样）",
    "audio_pipeline_qwen3.5-4b-official": "ASR+4B+TTS 级联（官方采样）",
    "audio_pipeline_minicpm5-2b-greedy": "ASR+MiniCPM5-2B+TTS 级联（greedy）",
    "audio_pipeline_minicpm5-2b-official": "ASR+MiniCPM5-2B+TTS 级联（官方采样）",
    "audio_edgeinstant": "EdgeInstant-Omni-2B（音频·greedy）",
}
LAUGH = set("哈呵嘿嘻嗯啊哦噢额唔呀吧呢吗啦哇咦嗷么嘛咯嘞")
SENT_SPLIT = re.compile(r"[。！？!?；;\n]+")


def load_scenes():
    m = {}
    for f in sorted((DATA / "scenes").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d["scenes"]:
            m[s["scene_id"]] = s
    return m


def load_judge_flags(judge_dir):
    flags = {}
    if not judge_dir.exists():
        return flags
    for f in sorted(judge_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for t in rec.get("turns", []):
                flags[(rec["card_id"], rec["scene_id"], t.get("idx"))] = t.get("flags") or []
    return flags


def repeat_metrics(text):
    t = (text or "").strip()
    n = len(t)
    m = {"len": n, "char_run": 0, "char_run_char": "", "unit": "", "unit_count": 0,
         "unique_ratio": round(len(set(t)) / n, 3) if n else 1.0,
         "zlib_ratio": round(len(zlib.compress(t.encode("utf-8"))) / max(1, n), 3),
         "sent_repeat": 0, "max4gram": 0, "max4gram_text": ""}
    if n == 0:
        return m
    run = 1
    for i in range(1, n):
        if t[i] == t[i - 1]:
            run += 1
            if run > m["char_run"]:
                m["char_run"], m["char_run_char"] = run, t[i]
        else:
            run = 1
    best = (1, "")
    for L in range(1, 25):
        i = 0
        while i + 2 * L <= n:
            if t[i:i + L] == t[i + L:i + 2 * L]:
                k = 2
                while i + (k + 1) * L <= n and t[i + k * L:i + (k + 1) * L] == t[i:i + L]:
                    k += 1
                if k > best[0]:
                    best = (k, t[i:i + L])
                i += k * L
            else:
                i += 1
    m["unit"], m["unit_count"] = best[1], best[0]
    cnt = collections.Counter(t[i:i + 4] for i in range(max(0, n - 3)))
    if cnt:
        unit, c = cnt.most_common(1)[0]
        m["max4gram"], m["max4gram_text"] = c, unit
    sents = [s.strip() for s in SENT_SPLIT.split(t) if len(s.strip()) >= 4]
    cur = 1
    for i in range(1, len(sents)):
        cur = cur + 1 if sents[i] == sents[i - 1] else 1
        m["sent_repeat"] = max(m["sent_repeat"], cur)
    return m


def classify(tm):
    """-> (level, reasons). level 2 = severe, 1 = moderate, 0 = none."""
    reasons, sev = [], 0
    u, c = tm["unit"], tm["unit_count"]
    if c >= 8 and len(u) >= 2:
        sev = 2; reasons.append(f"连续重复单元 {u[:12]!r}×{c}")
    elif c >= 5 and len(u) >= 2:
        sev = max(sev, 1); reasons.append(f"连续重复单元 {u[:12]!r}×{c}")
    if tm["char_run"] >= 14:
        sev = 2; reasons.append(f"同字连续 {tm['char_run_char']}×{tm['char_run']}")
    elif tm["char_run"] >= 10 and tm["char_run_char"] not in LAUGH:
        sev = max(sev, 1); reasons.append(f"同字连续 {tm['char_run_char']}×{tm['char_run']}")
    if tm["len"] >= 120 and tm["unique_ratio"] < 0.22:
        sev = 2; reasons.append(f"唯一字符率 {tm['unique_ratio']}")
    elif tm["len"] >= 120 and tm["unique_ratio"] < 0.32:
        sev = max(sev, 1); reasons.append(f"唯一字符率 {tm['unique_ratio']}")
    if tm["len"] >= 120 and tm["zlib_ratio"] < 0.20:
        sev = 2; reasons.append(f"压缩比 {tm['zlib_ratio']}")
    elif tm["len"] >= 140 and tm["zlib_ratio"] < 0.30:
        sev = max(sev, 1); reasons.append(f"压缩比 {tm['zlib_ratio']}")
    if tm["sent_repeat"] >= 4:
        sev = 2; reasons.append(f"连续重复句×{tm['sent_repeat']}")
    elif tm["sent_repeat"] >= 3:
        sev = max(sev, 1); reasons.append(f"连续重复句×{tm['sent_repeat']}")
    if tm["len"] >= 450 and tm["unique_ratio"] < 0.55:
        sev = max(sev, 1); reasons.append(f"长文本退化 len={tm['len']}")
    return sev, reasons


def excerpt(text, tm, width=48):
    t = (text or "").strip().replace("\n", " ")
    if tm["char_run"] >= 8:
        ch = tm["char_run_char"]
        idx = t.find(ch * 8)
        if idx >= 0:
            return t[max(0, idx - width):idx + width * 2].strip() + f"  [{ch} ×{tm['char_run']}]"
    if tm["unit_count"] >= 3 and tm["unit"]:
        idx = t.find(tm["unit"] * 3)
        if idx >= 0:
            return t[max(0, idx - width):idx + width * 2].strip() + f"  [{tm['unit']} ×{tm['unit_count']}]"
    return t[:width * 3].strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = load_scenes()
    outputs, judge_flags = {}, {}
    for track, model, bdir, jdir in MODELS:
        judge_flags[model] = load_judge_flags(jdir)
        ops = {}
        if bdir.exists():
            for f in sorted(bdir.glob("*.jsonl")):
                if not f.name.startswith("rc"):
                    continue
                for line in f.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    for t in rec.get("turns", []):
                        ops[(rec["card_id"], rec["scene_id"], t.get("idx"))] = {
                            "model": model, "track": track, "card_id": rec["card_id"],
                            "scene_id": rec["scene_id"], "idx": t.get("idx"),
                            "user_text": t.get("user_text", ""),
                            "assistant_text": t.get("assistant_text", "")}
        outputs[(track, model)] = ops

    # detection
    records = []
    summary = {}
    for (track, model), ops in outputs.items():
        counts = collections.Counter(); judge_rep = 0
        for key, rec in ops.items():
            tm = repeat_metrics(rec["assistant_text"])
            level, reasons = classify(tm)
            jf = judge_flags.get(model, {}).get(key, [])
            has_judge = "repetition" in jf
            if has_judge:
                judge_rep += 1
            tier = "severe" if level == 2 else "moderate" if level == 1 else None
            if tier is None and has_judge and tm["len"] >= 180:
                tier = "suspected"
            if tier:
                counts[tier] += 1
                item = dict(rec)
                item.update({"tier": tier, "char_len": tm["len"], "metrics": tm,
                             "reasons": reasons, "judge_repetition": has_judge,
                             "judge_flags": jf,
                             "excerpt": excerpt(rec["assistant_text"], tm),
                             "category": scenes.get(rec["scene_id"], {}).get("category", "")})
                records.append(item)
        n = len(ops)
        summary[model] = {"track": track, "label": LABEL.get(model, model), "n_turns": n,
                          "severe": counts["severe"], "moderate": counts["moderate"],
                          "suspected": counts["suspected"],
                          "n_flagged": counts["severe"] + counts["moderate"],
                          "flag_rate": round((counts["severe"] + counts["moderate"]) / max(1, n), 4),
                          "judge_repetition": judge_rep,
                          "judge_repetition_rate": round(judge_rep / max(1, n), 4)}

    # cross-model matrix for severe/moderate items
    item_keys = sorted({(r["card_id"], r["scene_id"], r["idx"]) for r in records if r["tier"] in ("severe", "moderate")})
    matrix = {}
    for key in item_keys:
        row = {}
        for track, model, *_ in MODELS:
            rec = outputs.get((track, model), {}).get(key)
            if rec is None:
                row[model] = None; continue
            tm = repeat_metrics(rec["assistant_text"])
            level, reasons = classify(tm)
            jf = judge_flags.get(model, {}).get(key, [])
            row[model] = {"char_len": tm["len"],
                          "tier": "severe" if level == 2 else "moderate" if level == 1 else None,
                          "reasons": reasons, "judge_repetition": "repetition" in jf,
                          "excerpt": excerpt(rec["assistant_text"], tm) if level else rec["assistant_text"][:60].strip()}
        matrix["|".join(map(str, key))] = {"key": key, "models": row,
                                           "scene": {"category": scenes.get(key[1], {}).get("category", ""),
                                                     "expected_behavior": scenes.get(key[1], {}).get("expected_behavior", "")}}
    # per-item overlap: which models have severe on the same item
    overlap = collections.Counter()
    for key_str, item in matrix.items():
        for track, model, *_ in MODELS:
            r = item["models"].get(model)
            if r and r["tier"] == "severe":
                overlap[model] += 1

    with open(OUT / "repetition_flags.jsonl", "w", encoding="utf-8") as f:
        for item in sorted(records, key=lambda x: (x["model"], x["card_id"], x["scene_id"], x["idx"])):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    (OUT / "summary.json").write_text(json.dumps(
        {"summary": summary, "n_items_severe_or_moderate": len(item_keys), "matrix": matrix,
         "severe_item_count_by_model": dict(overlap)}, ensure_ascii=False, indent=1), encoding="utf-8")

    # report
    L = ["# 复读 / 死循环 badcase 清单（v0.1）", "",
         "检测方式：程序化复读指标（连续重复单元、同字连续、低唯一字符率、低 zlib 压缩比、连续重复句、长文本退化）"
         " + DeepSeek 判官 `repetition` flag 作为辅助。",
         "",
         "| Tier | 含义 |", "|---|---|",
         "| severe | 明显死循环/退化：连续重复单元≥8、同字连续≥14、唯一字符率<0.22、压缩比<0.20、连续重复句≥4 等 |",
         "| moderate | 疑似复读：较弱指标（重复单元 5–7、同字连续 10–13、唯一字符率<0.32、压缩比<0.30 等） |",
         "| suspected | 程序化未命中，但判官标 repetition 且长度≥180 字（只统计，不逐条列出） |",
         "",
         "## 1. 各模型统计", "",
         "| 模型 | Track | 总轮数 | severe | moderate | suspected | 判官 repetition | 判官 rate |",
         "|---|---|---|---|---|---|---|---|"]
    for m, s in summary.items():
        L.append(f"| {s['label']} | {s['track']} | {s['n_turns']} | {s['severe']} | {s['moderate']} | {s['suspected']} | {s['judge_repetition']} | {s['judge_repetition_rate']:.1%} |")
    L += ["", "## 2. 严重/疑似复读测试项明细（跨模型对比）", "",
          f"共 {len(item_keys)} 个测试项被至少一个模型判定为 severe/moderate；下表列出同一 (card, scene, turn) 上全部模型的回答表现（`—` 表示该 track 未覆盖该场景）。", ""]
    for key_str, item in sorted(matrix.items()):
        card, sid, idx = item["key"]
        L.append(f"### {sid}（{item['scene']['category']}）· turn {idx}")
        if item["scene"].get("expected_behavior"):
            L.append(f"> 期望行为：{item['scene']['expected_behavior']}")
        L.append("")
        L.append("| 模型 | 字数 | Tier | 程序化原因 | 判官 repetition | 回答摘录 |")
        L.append("|---|---|---|---|---|---|")
        for track, model, *_ in MODELS:
            r = item["models"].get(model)
            if r is None:
                L.append(f"| {LABEL.get(model, model)} | — | — | — | — | — |")
            else:
                L.append(f"| {LABEL.get(model, model)} | {r['char_len']} | {r['tier'] or ''} | {', '.join(r['reasons']) or '—'} | {'Y' if r['judge_repetition'] else ''} | {r['excerpt'][:140].replace('|','/')} |")
        L.append("")
    L += ["## 3. 给训练人员的备注", "",
          "- 逐条后台记录：`results/badcases/repetition_flags.jsonl`（含完整 assistant_text、指标、判官 flags），可按 tier/model 过滤。",
          "- 同一测试项跨模型对比可直接定位是「模型能力问题」还是「输入/场景问题」：Qwen3.5-2B 的 severe 多为长文本自回归退化（512 字上限附近）；ASR+2B+TTS 级联复现同类复读，说明来自 LLM 本身而非 ASR/TTS。",
          "- Qwen3.5-4B 无 severe，仅有少量 moderate；EdgeInstant 系列 severe/moderate 与其 preview 状态和偏长回复有关。",
          "- 建议训练侧优先看 severe 列表（尤其重复单元/同字连续/长文本退化三类），并检查 max_new_tokens 截断附近的样本。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"records: {len(records)}, severe/moderate items: {len(item_keys)}")
    print(f"severe item overlap: {dict(overlap)}")
    print("report:", OUT / "report.md")


if __name__ == "__main__":
    main()
