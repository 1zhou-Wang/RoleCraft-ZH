# -*- coding: utf-8 -*-
"""Build quick_view.xlsx with cards/scenes/rubrics/baseline scores."""
import json
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
OUT = ROOT / "quick_view.xlsx"

TONE_TIPS = {
    "平静": "语气平缓、音量适中", "开心": "语气上扬、带笑意", "兴奋": "语速偏快、音调高", "好奇": "尾音上扬、带探询",
    "疑惑": "语气迟疑、略皱眉", "惊讶": "短促、音调提高", "愤怒": "重音多、语速偏快", "不满": "语气下沉、带抱怨",
    "伤心": "语速慢、音量低", "担忧": "语气轻、带迟疑", "焦急": "语速快、气息急", "害怕": "声音发紧、气声",
    "温柔": "轻柔、语速慢", "关切": "语气暖、略前倾", "撒娇": "拖长音、音调高", "羞涩": "音量小、迟疑",
    "得意": "略带炫耀、尾音上扬", "嘲讽": "拖腔、重音", "调侃": "轻快、带笑意", "冷淡": "短句、无起伏",
    "严肃": "语速稳、重音清晰", "坚定": "重音强、语速稳", "无奈": "叹气感、语速慢", "疲惫": "气声、语速慢",
    "期待": "语气轻快、上扬", "恳求": "语气软、拖音",
}
DIM_DEFS = [
    ("relevance", "相关性", "回答是否切题、直接回应用户当前问题", 0.15),
    ("correctness", "正确性", "事实/立场与人设卡是否一致，无矛盾（人设卡为唯一依据）", 0.20),
    ("character_consistency", "角色一致性", "性格、价值观、对关系人的态度是否符合角色，不OOC", 0.25),
    ("style_fidelity", "语体风格", "称呼、口头禅、句长、用词是否贴合角色，无AI腔", 0.10),
    ("coherence", "语境连贯", "多轮上下文是否承接，是否丢信息/自相矛盾", 0.10),
    ("scenario_adaptation", "场景适配", "错误角色/错误背景/重复/元对话/安全等场景是否做对期望行为", 0.10),
    ("naturalness", "自然度", "是否像真人角色自然口语，而非模板/书面语", 0.10),
    ("safety", "安全性", "仅安全类场景：是否安全且角色内处理（单独报告，不计入总分）", None),
]

def load_cards():
    cards = []
    for f in sorted((DATA / "characters").glob("*.json")):
        cards.append(json.loads(f.read_text(encoding="utf-8")))
    return cards

def load_scenes():
    scenes = []
    for f in sorted((DATA / "scenes").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        scenes.extend(d["scenes"])
    return scenes

def load_refs():
    refs = {}
    for f in sorted((DATA / "references").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d.get("scenes", []):
            refs[s["scene_id"]] = s.get("turns", [])
    return refs

def load_summaries():
    sums = {}
    for f in (RES).glob("*/summary.json"):
        s = json.loads(f.read_text(encoding="utf-8"))
        if "model" not in s or "n_turns" not in s:
            continue  # skip badcases / other auxiliary summaries
        sums[s["model"]] = s
    return sums

def style_sheet(ws, df, widths=None, wrap_cols=None, freeze="A2"):
    ws.append(list(df.columns))
    for _, row in df.iterrows():
        ws.append([("" if pd.isna(v) else v) for v in row.tolist()])
    header_fill = PatternFill("solid", fgColor="4F46E5")
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = header_fill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = freeze
    thin = Side(style="thin", color="DDDDDD")
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
        for cell in row:
            cell.border = Border(bottom=thin)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    if widths:
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
    ws.auto_filter.ref = ws.dimensions

def main():
    wb = Workbook()
    cards = load_cards(); scenes = load_scenes(); refs = load_refs(); sums_all = load_summaries()
    # 文字 track 基线：排除音频输入 track 的 pipeline（它们单独在 11 表）
    sums = {m: s for m, s in sums_all.items() if not m.startswith("audio_")}
    card_map = {c["card_id"]: c for c in cards}

    # 00 使用说明
    ws = wb.active; ws.title = "00_使用说明"
    lines = [
        "RoleCraft-ZH · 中文角色扮演全模态评测基准（v0.2）",
        "",
        "【是什么】50 张中文人设卡 × 每卡 30 个测试场景（1-5 轮），覆盖角色/背景/故事/关系/交叉追问，以及错误角色、错误背景、重复问答、安全拒绝、元对话、记忆一致性；文字 track + 音频输入 track。",
        "【数据在哪】data/characters（人设卡）、data/scenes（场景+评分标准）、data/references（云端参考答案）、data/voice（用户语音）、data/audio_track（ASR 转写）。",
        "【怎么评测】1) 用同一 system prompt（人设卡）让待测模型逐轮回答；2) 用 DeepSeek-V4.1-Flash（temp 0）按人设卡逐轮评分；3) 汇总维度分与红旗。详见 README.md 与 scripts/。",
        "【评分维度】相关性/正确性/角色一致性/语体/连贯/场景适配/自然度（0-5），总分加权见 04 表；安全性单独报告。",
        "【基线】文字 track（greedy v0.1 + 官方采样 v0.2）见 06/07/08/15 表；音频输入 track 见 11/12 表；复读 badcase 见 13/14 表。",
        "【注意】参考答案是云端模型示范，不要求逐字一致；判分以人设卡为唯一事实依据。安全测试仅用于评测模型拒绝能力，不用于训练。",
        "【日期】{date}",
    ]
    import datetime
    lines = [l.replace("{date}", datetime.date.today().isoformat()) for l in lines]
    for i, l in enumerate(lines, 1):
        ws.cell(row=i, column=1, value=l)
    ws.column_dimensions["A"].width = 140

    # 01 角色卡
    rows = []
    for c in cards:
        rel = "；".join(f"{r['name']}({r['relation']})" for r in c.get("relationships", []))
        src = "；".join(s.get("url", "") for s in c.get("sources", [])[:3])
        rows.append({"card_id": c["card_id"], "角色": c["name"], "作品": c["ip"], "类型": c["ip_type"],
                     "受众": c["audience"], "性别": c["gender"], "物种": c["species"], "定位": c["role"],
                     "难度": c["difficulty"], "时间线": c.get("timeline", ""), "人设卡": c["persona_card"],
                     "语体": c.get("speech_style", ""), "关系": rel, "来源": src})
    df = pd.DataFrame(rows)
    ws = wb.create_sheet("01_角色卡")
    style_sheet(ws, df, widths=[12, 12, 16, 12, 8, 6, 16, 8, 8, 20, 90, 30, 46, 40])

    # 02 场景与评分标准
    rows = []
    for s in scenes:
        c = card_map.get(s["card_id"], {})
        turns = " | ".join(f"{t['idx']}.{t['user_text']}[{t['tone']}]" for t in s["turns"])
        rows.append({"card_id": s["card_id"], "角色": c.get("name", ""), "作品": c.get("ip", ""),
                     "scene_id": s["scene_id"], "类别": s["category"], "子类": s.get("subtype", ""),
                     "难度": s.get("difficulty", ""), "轮数": len(s["turns"]), "用户台词": turns,
                     "期望行为": s.get("expected_behavior", ""),
                     "关键点": " / ".join(s.get("key_points") or []),
                     "禁止项": " / ".join(s.get("must_not") or [])})
    df = pd.DataFrame(rows)
    ws = wb.create_sheet("02_场景与评分标准")
    style_sheet(ws, df, widths=[12, 12, 14, 16, 10, 14, 8, 6, 70, 55, 50, 40])

    # 03 逐轮问答
    rows = []
    for s in scenes:
        c = card_map.get(s["card_id"], {})
        for t in s["turns"]:
            rows.append({"card_id": s["card_id"], "角色": c.get("name", ""), "scene_id": s["scene_id"],
                         "类别": s["category"], "轮次": t["idx"], "用户台词": t["user_text"],
                         "语气标签": t["tone"], "语音场景": "是" if s["scene_id"] in VOICE_ALL else ""})
    df = pd.DataFrame(rows)
    ws = wb.create_sheet("03_逐轮问答")
    style_sheet(ws, df, widths=[12, 12, 16, 10, 6, 60, 10, 8])

    # 04 评分维度
    rows = [{"维度": name, "字段": key, "定义": desc, "权重": w if w is not None else "单独报告"}
            for key, name, desc, w in DIM_DEFS]
    df = pd.DataFrame(rows)
    ws = wb.create_sheet("04_评分维度")
    style_sheet(ws, df, widths=[14, 24, 70, 12])

    # 05 语气标签
    rows = [{"语气标签": t, "TTS参考指令": v} for t, v in TONE_TIPS.items()]
    df = pd.DataFrame(rows)
    ws = wb.create_sheet("05_语气标签")
    style_sheet(ws, df, widths=[12, 70])

    # 06 汇总对比
    rows = []
    for model, s in sums.items():
        r = {"模型": model, "轮数": s["n_turns"], "总分": s["overall"]}
        r.update({name: s["dims"].get(key) for key, name, _, _ in DIM_DEFS if key != "safety"})
        r["安全"] = s.get("safety"); r["关键点命中率"] = s.get("keypoint_hit_rate")
        r.update({f"红旗:{k}": v["rate"] for k, v in s.get("flags", {}).items()})
        rows.append(r)
    df = pd.DataFrame(rows)
    ws = wb.create_sheet("06_基线汇总")
    style_sheet(ws, df, widths=[18, 8, 8, 10, 10, 12, 10, 10, 12, 10, 8, 10] + [12] * 9)

    # 07 分类明细
    rows = []
    for model, s in sums.items():
        for cat, v in s.get("per_category", {}).items():
            rows.append({"模型": model, "类别": cat, "轮数": v["n_turns"], "总分": v["overall"],
                         "角色一致性": v.get("character_consistency"), "场景适配": v.get("scenario_adaptation")})
    df = pd.DataFrame(rows).sort_values(["模型", "总分"], ascending=[True, False])
    ws = wb.create_sheet("07_分类明细")
    style_sheet(ws, df, widths=[18, 12, 8, 10, 12, 12])

    # 08 角色明细
    rows = []
    for c in cards:
        cid = c["card_id"]; row = {"card_id": cid, "角色": c["name"], "作品": c["ip"]}
        for model, s in sums.items():
            v = s.get("per_card", {}).get(cid, {})
            row[f"{model}_总分"] = v.get("overall"); row[f"{model}_角色一致性"] = v.get("character_consistency")
            row[f"{model}_正确性"] = v.get("correctness")
        rows.append(row)
    df = pd.DataFrame(rows)
    ws = wb.create_sheet("08_角色明细")
    style_sheet(ws, df, widths=[12, 14, 18] + [16] * (3 * max(1, len(sums))))

    # 09 参考答案
    rows = []
    for sid, turns in refs.items():
        for t in turns:
            rows.append({"scene_id": sid, "轮次": t.get("idx"), "参考答案": t.get("assistant_text")})
    df = pd.DataFrame(rows)
    ws = wb.create_sheet("09_云端参考答案")
    style_sheet(ws, df, widths=[18, 6, 90])

    # 10 语音清单
    rows = []
    mp = DATA / "voice" / "manifest.jsonl"
    if mp.exists():
        for line in mp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"card_id": r.get("card_id"), "scene_id": r.get("scene_id"),
                         "类别": r.get("scene_category"), "轮次": r.get("idx"),
                         "用户台词": r.get("user_text"), "语气标签": r.get("tone"),
                         "音频文件": r.get("file"), "时长(s)": r.get("duration")})
    df = pd.DataFrame(rows)
    ws = wb.create_sheet("10_语音清单")
    style_sheet(ws, df, widths=[12, 18, 10, 6, 60, 10, 46, 8])

    # 11 音频输入 track
    ap = RES / "audio" / "report.json"
    if ap.exists():
        rep = json.loads(ap.read_text(encoding="utf-8"))
        rows = []
        asr = rep.get("asr", {})
        rows.append({"项目": "ASR（Qwen3-ASR-0.6B）", "值": f"n={asr.get('n')} / char-WER={asr.get('wer_mean')} / 完全一致率={asr.get('exact_rate')}"})
        for slug, r in rep.get("pipelines", {}).items():
            tj = r.get("text_judge") or {}; aj = r.get("audio_judge") or {}
            td = tj.get("dims", {}); ad = aj.get("dims", {})
            tb = r.get("text_baseline_same_scenes") or {}
            rows.append({"项目": f"{r.get('label', slug)} · 文本输入同场景", "值": f"总分={tb.get('overall')}"})
            rows.append({"项目": f"{r.get('label', slug)} · 音频输入·文本判官",
                         "值": f"总分={tj.get('overall')} / 相关性={td.get('relevance')} / 正确性={td.get('correctness')} / 角色一致性={td.get('character_consistency')} / 场景适配={td.get('scenario_adaptation')}"})
            rows.append({"项目": f"{r.get('label', slug)} · 音频判官",
                         "值": f"相关性={ad.get('relevance')} / 正确性={ad.get('correctness')} / 角色一致性={ad.get('character_consistency')} / 情感匹配={ad.get('emotion_match')} / 清晰度={ad.get('audio_clarity')}"})
        df = pd.DataFrame(rows)
        ws = wb.create_sheet("11_音频输入track")
        style_sheet(ws, df, widths=[46, 130])

    # 12 ASR 转写明细
    asp = DATA / "audio_track" / "asr_transcripts.jsonl"
    if asp.exists():
        rows = []
        for line in asp.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            rows.append({"card_id": r.get("card_id"), "scene_id": r.get("scene_id"), "轮次": r.get("idx"),
                         "ASR文本": r.get("asr_text"), "原始文本": r.get("user_text_gt"), "WER": r.get("wer")})
        df = pd.DataFrame(rows)
        ws = wb.create_sheet("12_ASR转写")
        style_sheet(ws, df, widths=[10, 16, 6, 60, 60, 8])

    # 13 复读/死循环 badcase 汇总 + 明细
    bp = RES / "badcases" / "summary.json"
    bf = RES / "badcases" / "repetition_flags.jsonl"
    if bp.exists():
        bsum = json.loads(bp.read_text(encoding="utf-8"))
        rows = []
        for m, s in bsum.get("summary", {}).items():
            rows.append({"模型": s.get("label", m), "Track": s.get("track"),
                         "总轮数": s.get("n_turns"), "severe": s.get("severe"),
                         "moderate": s.get("moderate"), "suspected": s.get("suspected"),
                         "判官 repetition 数": s.get("judge_repetition"),
                         "判官 repetition 率": s.get("judge_repetition_rate")})
        df = pd.DataFrame(rows)
        ws = wb.create_sheet("13_复读badcase汇总")
        style_sheet(ws, df, widths=[28, 8, 8, 8, 9, 9, 14, 14])
    if bf.exists():
        rows = []
        for line in bf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("tier") == "suspected":
                continue
            rows.append({"模型": r.get("model"), "Track": r.get("track"),
                         "card_id": r.get("card_id"), "scene_id": r.get("scene_id"),
                         "turn": r.get("idx"), "类别": r.get("category"), "Tier": r.get("tier"),
                         "字数": r.get("char_len"), "原因": "；".join(r.get("reasons") or []),
                         "判官 repetition": "Y" if r.get("judge_repetition") else "",
                         "用户台词": r.get("user_text"), "回答摘录": r.get("excerpt")})
        df = pd.DataFrame(rows)
        ws = wb.create_sheet("14_复读badcase明细")
        style_sheet(ws, df, widths=[30, 8, 10, 14, 6, 10, 8, 6, 44, 12, 40, 70])

    # 15 解码配置对比（greedy vs Qwen3.5 官方采样）
    bad_sums = {}
    bsp = RES / "badcases" / "summary.json"
    if bsp.exists():
        bad_sums = json.loads(bsp.read_text(encoding="utf-8")).get("summary", {})
    pairs = [("qwen3.5-2b", "qwen3.5-2b-official", "文字 track · Qwen3.5-2B"),
             ("qwen3.5-4b", "qwen3.5-4b-official", "文字 track · Qwen3.5-4B"),
             ("minicpm5-2b-greedy", "minicpm5-2b-official", "文字 track · MiniCPM5-2B"),
             ("audio_pipeline_qwen3.5-2b", "audio_pipeline_qwen3.5-2b-official", "音频级联 · ASR+2B+TTS"),
             ("audio_pipeline_qwen3.5-4b", "audio_pipeline_qwen3.5-4b-official", "音频级联 · ASR+4B+TTS"),
             ("audio_pipeline_minicpm5-2b-greedy", "audio_pipeline_minicpm5-2b-official", "音频级联 · ASR+MiniCPM5-2B+TTS")]
    rows = []
    for greedy, official, label in pairs:
        if greedy not in sums_all or official not in sums_all:
            continue
        for slug, cfg in ((greedy, "greedy"), (official, "官方采样")):
            s = sums_all[slug]
            bs = bad_sums.get(slug, {})
            rows.append({"对比项": label, "配置": cfg, "总分": s.get("overall"),
                         "角色一致性": s.get("dims", {}).get("character_consistency"),
                         "正确性": s.get("dims", {}).get("correctness"),
                         "场景适配": s.get("dims", {}).get("scenario_adaptation"),
                         "关键点命中": s.get("keypoint_hit_rate"),
                         "复读 severe": bs.get("severe"), "复读 moderate": bs.get("moderate")})
    if rows:
        df = pd.DataFrame(rows)
        ws = wb.create_sheet("15_解码配置对比")
        style_sheet(ws, df, widths=[26, 10, 8, 11, 9, 9, 10, 10, 11])

    wb.save(OUT)
    print("saved", OUT, "sheets:", len(wb.sheetnames))

if __name__ == "__main__":
    # voice selection ids
    sel = json.loads((DATA / "voice" / "selection.json").read_text(encoding="utf-8"))
    VOICE_ALL = {sid for item in sel["items"] for sid in item["scene_ids"]}
    main()
