# -*- coding: utf-8 -*-
"""Qwen2.5-Omni-3B audio-in/text-out baseline (B5) for RoleCraft-ZH (greedy).
User turns are the 784 TTS user audios (data/voice/audio/<scene>_<idx>.mp3).
Usage: python local/run_qwen_omni_audio.py --out baselines/audio_pipeline_qwen25-omni-3b-greedy
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local"))
import qwen_omni_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "baselines" / "audio_pipeline_qwen25-omni-3b-greedy"))
    ap.add_argument("--cards", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    sel = json.loads((ROOT / "data" / "voice" / "selection.json").read_text(encoding="utf-8"))["items"]
    if args.cards:
        want = set(args.cards.split(","))
        sel = [x for x in sel if x["card_id"] in want]
    scenes = C.scene_map()
    t0 = time.time()
    model, processor = C.load_model()
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)
    total_turns = 0
    for ci, item in enumerate(sel, 1):
        cid = item["card_id"]
        card = C.load_card(cid)
        outp = out / f"{cid}.jsonl"
        existing = []
        if outp.exists():
            existing = [json.loads(l) for l in outp.read_text(encoding="utf-8").splitlines() if l.strip()]
        done = {r["scene_id"] for r in existing}
        recs = []
        for sid in item["scene_ids"]:
            if sid in done or sid not in scenes:
                continue
            scene = scenes[sid]
            history = []
            turns_out = []
            for turn in scene["turns"]:
                rel_audio = f"data/voice/audio/{sid}_{turn['idx']}.mp3"
                audio_path = ROOT / rel_audio
                if not audio_path.exists():
                    raise FileNotFoundError(audio_path)
                reply, dt, nprompt = C._generate(model, processor,
                                                 C.audio_conversation(card, history, audio_path),
                                                 audio_mode=True, max_new_tokens=args.max_new_tokens)
                turns_out.append({"idx": turn["idx"], "tone": turn.get("tone"),
                                  "user_audio": rel_audio, "user_text": turn["user_text"],
                                  "user_text_gt": turn["user_text"], "assistant_text": reply,
                                  "latency_s": dt, "prompt_tokens": nprompt})
                history.append({"user_audio": audio_path, "assistant_text": reply})
                total_turns += 1
            recs.append({"card_id": cid, "scene_id": sid, "category": scene.get("category"),
                         "pipeline": "B5", "model": "qwen25-omni-3b", "mode": "audio-input",
                         "input_modality": "audio",
                         "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                         "turns": turns_out})
        with outp.open("w", encoding="utf-8") as f:
            for r in existing + recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{ci}/{len(sel)}] {cid}: scenes {len(existing)+len(recs)}/{len(item['scene_ids'])} | turns so far {total_turns} | {time.time()-t0:.0f}s", flush=True)
    print("DONE", total_turns, "turns", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
