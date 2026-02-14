import os
import csv

root = "/local_datasets/2D_direction_video_symmetry_4class_1combo/val"
output_csv = "./2D_direction_video_symmetry_4class_1combo_val.csv"

rows = []
qid = 1

for direction in sorted(os.listdir(root)):
    dir_path = os.path.join(root, direction)
    if not os.path.isdir(dir_path):
        continue
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".mp4"):
            continue
        rows.append({
            "question_id": f"{qid:08d}",
            "video": f"{direction}/{fname}",
            "question": "What direction does the circle move in the frame?",
            "answer": direction,
        })
        qid += 1

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question_id", "video", "question", "answer"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows to {output_csv}")