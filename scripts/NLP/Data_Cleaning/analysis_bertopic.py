import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# ========= 提前下载NLTK所需数据 =========
try:
    stopwords.words('english')
    WordNetLemmatizer().lemmatize('test')
except:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# ========= Config =========
DATA_PATH = "data_processed/final_dataset.csv"
OUT_DIR = "data_processed"
OUT_TOPIC_TABLE = os.path.join(OUT_DIR, "bertopic_topic_info.csv")
OUT_DOC_TOPIC = os.path.join(OUT_DIR, "bertopic_doc_topics.csv")
OUT_PNG = os.path.join(OUT_DIR, "topic_keywords_clean.png")

MIN_TOPIC_SIZE = 3
TARGET_TOPICS = 8

# ========= 停用词：通用 + 学术 + 丑食领域通用词 =========
DOMAIN_STOPWORDS = {
    # 学术写作通用
    'study', 'research', 'paper', 'article', 'findings', 'result', 'results',
    'method', 'methods', 'analysis', 'sample', 'data', 'participants',
    'experiment', 'survey', 'investigate', 'examine', 'explore', 'address',
    'discuss', 'present', 'report', 'show', 'demonstrate', 'indicate',
    'suggest', 'conclude', 'reveal', 'find', 'test', 'evaluate', 'assess',
    'measure', 'analyze', 'use', 'used', 'using', 'also', 'may', 'well',
    'significantly', 'significant', 'positive', 'negative', 'high', 'higher',
    'low', 'lower', 'two', 'three', 'one', 'first', 'second', 'third',
    # 丑食/食品领域通用词（出现在几乎所有主题里，无区分度）
    'food', 'consumer', 'consumers', 'consumption', 'waste', 'product',
    'products', 'purchase', 'suboptimal', 'imperfect', 'ugly', 'produce',
    'fruit', 'vegetable', 'willingness', 'intention', 'behavior', 'behaviour',
    'attitude', 'perception', 'perceived', 'effect', 'factor', 'factors',
    'impact', 'influence', 'role', 'based', 'related', 'paper', 'toward',
    'towards', 'among', 'across', 'within', 'whether', 'however', 'although',
    'therefore', 'thus', 'furthermore', 'moreover', 'addition',
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    stop_words = set(stopwords.words('english')).union(DOMAIN_STOPWORDS)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    # 词形还原后再过滤一次
    words = [w for w in words if w not in stop_words and len(w) > 2]

    return ' '.join(words)

def pick_abstract_col(df):
    for c in ["abstract", "Abstract", "AB", "ABS"]:
        if c in df.columns:
            return c
    raise ValueError("Cannot find abstract column")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 读取数据
    df = pd.read_csv(DATA_PATH, low_memory=False)
    abstract_col = pick_abstract_col(df)

    # 2. 数据清洗
    df = df.dropna(subset=[abstract_col]).copy()
    df[abstract_col] = df[abstract_col].astype(str).str.strip()
    df = df[df[abstract_col].str.len() > 50].copy()
    print("Loaded raw documents:", len(df))

    # 3. 预处理
    df["abstract_processed"] = df[abstract_col].apply(preprocess_text)
    df = df[df["abstract_processed"].str.len() > 10].copy()
    documents = df["abstract_processed"].tolist()
    print("Processed documents:", len(documents))

    # 4. 模型组件配置
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    n_docs = len(documents)
    n_neighbors = min(10, n_docs - 1)
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    all_stopwords_list = list(set(stopwords.words('english')).union(DOMAIN_STOPWORDS))
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=all_stopwords_list,
        min_df=2,
    )

    # 5. 构建并训练BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=MIN_TOPIC_SIZE,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(documents)

    # 6. 检查初始主题数，若超过目标则reduce
    topic_info_raw = topic_model.get_topic_info()
    n_topics_raw = int(topic_info_raw[topic_info_raw["Topic"] != -1].shape[0])
    print(f"\nInitial topics (before reduce): {n_topics_raw}")

    if n_topics_raw > TARGET_TOPICS:
        print(f"Reducing to {TARGET_TOPICS} topics...")
        topic_model.reduce_topics(documents, nr_topics=TARGET_TOPICS)
        topics = topic_model.topics_
        print(f"Topics after reduce: {len(set(t for t in topics if t != -1))}")
    elif n_topics_raw < TARGET_TOPICS:
        print(f"Warning: Only got {n_topics_raw} topics (< target {TARGET_TOPICS}).")
        print("Consider lowering MIN_TOPIC_SIZE further or check data size.")

    # 7. 保存结果
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(OUT_TOPIC_TABLE, index=False, encoding="utf-8-sig")

    df_out = df.copy()
    df_out["topic"] = topics
    df_out.to_csv(OUT_DOC_TOPIC, index=False, encoding="utf-8-sig")

    # 8. 输出摘要
    n_outliers = int((pd.Series(topics) == -1).sum())
    n_topics_final = int(topic_info[topic_info["Topic"] != -1].shape[0])
    print("\n=== Summary ===")
    print("Total documents:", len(documents))
    print("Outliers (-1)  :", n_outliers)
    print("Final topics   :", n_topics_final)
    print("\nTopic keywords preview:")
    for _, row in topic_info[topic_info["Topic"] != -1].iterrows():
        print(f"  Topic {row['Topic']:2d} (n={row['Count']:3d}): {row['Representation'][:5]}")

    print("\nSaved:", OUT_TOPIC_TABLE)
    print("Saved:", OUT_DOC_TOPIC)

    # 9. 可视化
    if n_topics_final >= 3:
        # 图1：主题间距离图（HTML交互版）
        try:
            fig = topic_model.visualize_topics()
            fig.write_html(os.path.join(OUT_DIR, "Figure5_intertopic_distance.html"))
            print("Saved: Figure5_intertopic_distance.html")
        except Exception as e:
            print("Intertopic plot failed:", str(e))

        # 图2：关键词柱状图 → PNG（论文直接用）
        try:
            fig2 = topic_model.visualize_barchart(
                top_n_topics=n_topics_final,
                n_words=8,
                title="Topic Keywords"
            )
            fig2.write_image(OUT_PNG, scale=2)  # 需要 pip install kaleido
            print("Saved:", OUT_PNG)
        except Exception as e:
            print("Keywords bar plot (PNG) failed:", str(e))
            # 降级保存为HTML
            try:
                fig2.write_html(os.path.join(OUT_DIR, "topic_keywords_bar.html"))
                print("Fallback saved: topic_keywords_bar.html")
            except:
                pass

        # 图3：主题热力图（可选）
        try:
            fig3 = topic_model.visualize_heatmap()
            fig3.write_html(os.path.join(OUT_DIR, "topic_heatmap.html"))
            print("Saved: topic_heatmap.html")
        except Exception as e:
            print("Heatmap failed:", str(e))

if __name__ == "__main__":
    main()