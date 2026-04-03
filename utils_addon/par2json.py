import pandas as pd

# parquet 로드
df = pd.read_parquet("/data/hyuntak/project/2026/vlm_direction/cross-modal-information-flow-in-MLLM/test-00000-of-00001.parquet")

# 사람이 보기 좋게 JSON으로 저장
df.to_json(
    "./ActivityNetQa.json",
    orient="records",
    indent=2,
    force_ascii=False
)