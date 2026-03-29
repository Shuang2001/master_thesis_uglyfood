import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data_processed/final_dataset.csv")

# === 1. 年度发文量 ===
trend = df["year"].value_counts().sort_index()

plt.figure(figsize=(10,6))
plt.plot(trend.index, trend.values)
plt.xlabel("Year")
plt.ylabel("Number of Publications")
plt.title("Publication Trend of Ugly/Imperfect Produce Consumer Research")
plt.grid(True)
plt.tight_layout()
plt.savefig("data_processed/publication_trend.png", dpi=300)
plt.show()

# === 2. CAGR 计算 ===
start_year = trend.index.min()
end_year = trend.index.max()
start_value = trend.loc[start_year]
end_value = trend.loc[end_year]

years = end_year - start_year

if years > 0:
    cagr = (end_value / start_value) ** (1/years) - 1
    print("CAGR:", round(cagr*100, 2), "%")
else:
    print("CAGR cannot be computed.")