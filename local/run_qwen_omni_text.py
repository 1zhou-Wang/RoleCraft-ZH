# -*- coding: utf-8 -*-
"""Qwen2.5-Omni-3B text-in/text-out baseline for RoleCraft-ZH (greedy).
Usage: python local/run_qwen_omni_text.py --out baselines/qwen25-omni-3b-greedy [--cards rc01,rc02]
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local"))
import qwen_omni_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "baselines" / "qwen25-omni-3b-greedy"))
    ap.add_argument("--cards", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cards = sorted(p.stem for p in (ROOT / "data" / "characters").glob("*.json"))
    if args.cards:
        want = set(args.cards.split(","))
        cards = [c for c in cards if c in want]
    t0 = time.time()
    model, processor = C.load_model()
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)
    total_turns = 0
    for ci, cid in enumerate(cards, 1):
        card = C.load_card(cid)
        scenes = C.load_scenes(cid)["scenes"]
        outp = out / f"{cid}.jsonl"
        done = set()
        if outp.exists():
            for line in outp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line)["scene_id"])
        recs = []
        for scene in scenes:
            sid = scene["scene_id"]
            if sid in done:
                continue
            history = []
            turns_out = []
            for turn in scene["turns"]:
                reply, dt, nprompt = C._generate(model, processor,
                                                 C.text_conversation(card, history, turn["user_text"]),
                                                 audio_mode=False, max_new_tokens=args.max_new_tokens)
                turns_out.append({"idx": turn["idx"], "tone": turn.get("tone"), "user_text": turn["user_text"],
                                  "assistant_text": reply, "latency_s": dt, "prompt_tokens": nprompt})
                history.append({"user_text": turn["user_text"], "assistant_text": reply})
                total_turns += 1
            recs.append({"card_id": cid, "scene_id": sid, "category": scene.get("category"),
                         "pipeline": "text", "model": "qwen25-omni-3b",
                         "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                         "turns": turns_out})
        # rewrite card file = done recs + new recs
        existing = []
        if outp.exists():
            existing = [json.loads(l) for l in outp.read_text(encoding="utf-8").splitlines() if l.strip()]
        with outp.open("w", encoding="utf-8") as f:
            for r in existing + recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{ci}/{len(cards)}] {cid}: scenes {len(existing)+len(recs)}/{len(scenes)} | turns so far {total_turns} | {time.time()-t0:.0f}s", flush=True)
    print("DONE", total_turns, "turns", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
