import pandas as pd

df = pd.read_csv("data_processed/consumer_ugly_core_subset.csv")

n = min(150, len(df))
sample = df.sample(n=n, random_state=42).copy()

sample["label"] = ""
sample["note"] = ""

sample.to_csv("data_processed/core_sample_for_labeling.csv",
              index=False,
              encoding="utf-8-sig")

print("Saved core sample:", n)