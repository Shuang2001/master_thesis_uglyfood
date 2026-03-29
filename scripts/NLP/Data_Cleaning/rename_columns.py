import pandas as pd

in_path  = r"data_processed/df_clean.csv"
out_path = r"data_processed/df_clean_en.csv"

df = pd.read_csv(in_path)

# 兼容：如果已经是英文列名，不会出问题
rename_map = {
    "EID": "eid",
    "eid": "eid",
    "DOI": "doi",
    "doi": "doi",
    "文献标题": "title",
    "title": "title",
    "年份": "year",
    "year": "year",
    "来源出版物名称": "journal",
    "journal": "journal",
    "作者": "authors",
    "authors": "authors",
    "归属机构": "affiliations",
    "affiliations": "affiliations",
    "带归属机构的作者": "authors_with_affiliations",
    "authors_with_affiliations": "authors_with_affiliations",
    "摘要": "abstract",
    "abstract": "abstract",
    "作者关键字": "author_keywords",
    "author_keywords": "author_keywords",
    "索引关键字": "index_keywords",
    "index_keywords": "index_keywords",
    "文献类型": "doc_type",
    "doc_type": "doc_type",
    "出版阶段": "publication_stage",
    "publication_stage": "publication_stage",
    "开放获取": "open_access",
    "open_access": "open_access",
    "链接": "link",
    "link": "link",
    "__source_file": "__source_file",
}

df = df.rename(columns=rename_map)

# 固定列顺序（存在才保留）
cols_order = [
    "eid","doi","title","year","journal","authors",
    "affiliations","authors_with_affiliations",
    "abstract","author_keywords","index_keywords",
    "doc_type","publication_stage","open_access","link","__source_file"
]
cols_order = [c for c in cols_order if c in df.columns]
df = df[cols_order].copy()

df.to_csv(out_path, index=False, encoding="utf-8-sig")
print("Saved:", out_path)
print("Columns:", list(df.columns))
print("Rows:", len(df))