import pandas as pd
import json


csv_file_path = "/home/cai/project/bm25_embedding/data/valid_bio.csv"
df = pd.read_csv(csv_file_path, encoding="utf-8")
out = {}
for i in range(df.shape[0]):
    question = str(df.iloc[i, 0]).strip()
    context = str(df.iloc[i, 1]).split("<context>")[-1]
    if question not in out.keys():
        out[question] = [context]
    else:
        out[question].append(context)

with open("/home/cai/project/bm25_embedding/data/bioasq.jsonl", "w", encoding="utf-8") as f:
    for k, v in out.items():
        tmp = {"question": k, "contexts": list(set(v))}
        json_str = json.dumps(tmp, ensure_ascii=False)
        f.write(json_str + "\n")