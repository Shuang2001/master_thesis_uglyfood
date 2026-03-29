import pandas as pd

df = pd.read_csv("data_processed/df_clean_en.csv")

text = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.lower()

ugly_terms = [
    "ugly", "imperfect", "suboptimal", "wonky",
    "cosmetic standard", "visually imperfect",
    "odd-shaped", "misshapen"
]

consumer_terms = [
    "consumer", "consumer behavior", "consumer behaviour"
]

purchase_terms = [
    "purchase", "buy", "willingness to pay",
    "willingness to buy", "wtp",
    "purchase intention", "choice experiment",
    "discrete choice", "preference"
]

cond_ugly = False
for t in ugly_terms:
    cond_ugly |= text.str.contains(t)

cond_consumer = False
for t in consumer_terms:
    cond_consumer |= text.str.contains(t)

cond_purchase = False
for t in purchase_terms:
    cond_purchase |= text.str.contains(t)

subset = df[cond_ugly & cond_consumer & cond_purchase].copy()

subset.to_csv("data_processed/consumer_ugly_core_subset.csv", index=False)

print("Core subset size:", len(subset))