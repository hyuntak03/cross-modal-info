import pandas as pd
import glob

for f in sorted(glob.glob("datasets/GQA_*.csv")):
    df = pd.read_csv(f)
    print(f"\n=== {f.split('/')[-1]} ===")
    print(df["answer"].head(10))