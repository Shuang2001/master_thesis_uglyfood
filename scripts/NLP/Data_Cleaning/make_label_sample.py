import os
import pandas as pd

IN_PATH  = r"data_processed/df_clean_en.csv"
OUT_DIR  = r"data_processed"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_PATH)

# 只保留标注需要的字段，减少你阅读负担
keep = ["eid","doi","year","journal","title","abstract","author_keywords","index_keywords"]
keep = [c for c in keep if c in df.columns]
df = df[keep].copy()

# 随机抽样150篇（固定random_state保证可复现）
n = 150
sample = df.sample(n=min(n, len(df)), random_state=42).copy()

# 预留标注列：你在Excel里填 1(相关) / 0(不相关)
sample["label"] = ""
sample["note"] = ""  # 可选：你想写一句排除原因/关键词就写这里

out_path = os.path.join(OUT_DIR, "sample_for_labeling.csv")
sample.to_csv(out_path, index=False, encoding="utf-8-sig")

print("Saved:", out_path)
print("Rows:", len(sample))