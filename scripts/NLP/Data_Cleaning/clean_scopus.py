import os, glob
import pandas as pd

DATA_DIR = r"data_raw"
OUT_DIR  = r"data_processed"
os.makedirs(OUT_DIR, exist_ok=True)

files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print("Found CSV files:", [os.path.basename(f) for f in files])

if len(files) == 0:
    raise SystemExit("No CSV found in data_raw/. Please put your Scopus CSV exports there.")

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["__source_file"] = os.path.basename(f)
    dfs.append(df)

raw = pd.concat(dfs, ignore_index=True)
print("Raw shape (rows, cols):", raw.shape)

rename_map = {
    "EID": "eid",
    "DOI": "doi",
    "文献标题": "title",
    "年份": "year",
    "来源出版物名称": "journal",
    "作者": "authors",
    "归属机构": "affiliations",
    "带归属机构的作者": "authors_with_affiliations",
    "摘要": "abstract",
    "作者关键字": "author_keywords",
    "索引关键字": "index_keywords",
    "文献类型": "doc_type",
    "出版阶段": "publication_stage",
    "开放获取": "open_access",
    "链接": "link",
}
raw = raw.rename(columns=rename_map)

need_cols = ["title", "abstract"]
for c in need_cols:
    if c not in raw.columns:
        raise SystemExit(f"Missing required column: {c}. Your CSV column names differ; send me the column list.")

for c in ["title", "abstract", "doi"]:
    if c in raw.columns:
        raw[c] = raw[c].fillna("").astype(str).str.strip()

raw = raw[(raw["title"] != "") & (raw["abstract"] != "")].copy()

if "year" in raw.columns:
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")

raw["doi_norm"] = raw["doi"].fillna("").astype(str).str.lower().str.strip() if "doi" in raw.columns else ""
raw["title_norm"] = raw["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

before = len(raw)
with_doi = raw[raw["doi_norm"] != ""].drop_duplicates(subset=["doi_norm"], keep="first")
no_doi   = raw[raw["doi_norm"] == ""].drop_duplicates(subset=["title_norm", "year"], keep="first")
df_clean = pd.concat([with_doi, no_doi], ignore_index=True)

print("After dedup rows:", len(df_clean), "| removed:", before - len(df_clean))

keep_cols = [
    "eid","doi","title","year","journal","authors",
    "affiliations","authors_with_affiliations",
    "abstract","author_keywords","index_keywords",
    "doc_type","publication_stage","open_access","link","__source_file"
]
keep_cols = [c for c in keep_cols if c in df_clean.columns]
df_clean = df_clean[keep_cols].copy()

out_path = os.path.join(OUT_DIR, "df_clean.csv")
df_clean.to_csv(out_path, index=False, encoding="utf-8-sig")
print("Saved:", out_path)