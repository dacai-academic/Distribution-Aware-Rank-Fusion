import pandas as pd

# 1. 读取 parquet
df = pd.read_parquet("/home/cai/project/bm25_embedding/data/train-00000-of-00001.parquet")

# 2. 随机抽样 52,000 条（可复现）
sampled_df = df.sample(n=54000, random_state=42)

# 3. 保存为 jsonl
sampled_df.to_json(
    "/home/cai/project/bm25_embedding/data/part_pubmed.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)
