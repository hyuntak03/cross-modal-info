"""
Stage 0 — R2R_4way_1500 dataset을 canonical candidate 순서로 통일.

기존: candidates 순서가 sample마다 random → letter mapping이 달라서 averaging 어려움.
변경 후:
  candidates = ["Up", "Right", "Down", "Left"] (모든 sample 동일)
  direction → letter 결정론적: up=A, right=B, down=C, left=D

이렇게 하면:
  - 같은 direction sample은 모두 같은 letter answer
  - Last token hidden을 (direction, task)별로 100+ sample에서 평균 가능
  - Identity만 다른 controlled hidden state 추출 가능

Output: cross-modal-info/analysis/swap_intervention/canonical_R2R/{task}.json
"""

import json
import os
from collections import Counter

SRC_ROOT = "/data/takhyun03/project/2026/vlm_direction/synthetic_testbed/Testbed/huggingface/R2R_4way_1500"
DST_ROOT = "/data/takhyun03/project/2026/vlm_direction/cross-modal-info/analysis/swap_intervention/canonical_R2R"

TASKS = ["shape_color", "obj_color", "shape_place", "obj_place"]

CANONICAL_CANDIDATES = ["Up", "Right", "Down", "Left"]
DIR_TO_LETTER = {"up": "A", "right": "B", "down": "C", "left": "D"}
DIR_TO_TEXT = {"up": "Up", "right": "Right", "down": "Down", "left": "Left"}


def canonicalize(sample):
    direction = sample["direction"].lower().strip()
    if direction not in DIR_TO_LETTER:
        return None
    out = dict(sample)
    out["candidates"] = list(CANONICAL_CANDIDATES)
    out["answer"] = DIR_TO_LETTER[direction]
    out["answer_text"] = DIR_TO_TEXT[direction]
    out["answer_direction"] = direction
    return out


def main():
    os.makedirs(DST_ROOT, exist_ok=True)
    summary = {}
    for task in TASKS:
        src = os.path.join(SRC_ROOT, f"{task}.json")
        with open(src) as f:
            data = json.load(f)
        canon = []
        skipped = 0
        for s in data:
            c = canonicalize(s)
            if c is None:
                skipped += 1
                continue
            canon.append(c)
        dst = os.path.join(DST_ROOT, f"{task}.json")
        with open(dst, "w") as f:
            json.dump(canon, f, indent=2)
        dir_counts = Counter(s["direction"].lower() for s in canon)
        summary[task] = {
            "total": len(canon), "skipped": skipped,
            "per_direction": dict(dir_counts),
        }
        print(f"[{task}] {len(canon)} samples (skipped {skipped})")
        for d, n in sorted(dir_counts.items()):
            print(f"  {d}: {n}  (letter {DIR_TO_LETTER[d]})")
    with open(os.path.join(DST_ROOT, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVED] {DST_ROOT}/")


if __name__ == "__main__":
    main()
