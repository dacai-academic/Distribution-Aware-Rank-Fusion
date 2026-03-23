import json

input_file1 = "/home/cai/project/bm25_embedding/data/part_huatuo.jsonl"
input_file2 = "/home/cai/project/bm25_embedding/data/part_pubmed.jsonl"
output_file1 = "train.jsonl"
output_file2 = "val.jsonl"

train_content = []
val_content = []
with open(input_file1, "r", encoding="utf-8") as f1:
    m = 0
    for line in f1.readlines():
        m += 1
        tmp_dict = json.loads(line)
        if m <= 50000:
            train_content.append(tmp_dict)
        else:
            val_content.append(tmp_dict)

with open(input_file2, "r", encoding="utf-8") as f2:
    n = 0
    for line in f2.readlines():
        n += 1
        tmp_dict = json.loads(line)
        new_dict = {}
        for k, v in tmp_dict.items():
            if k == "long_answer":
                new_dict["answer"] = v
            else:
                new_dict[k] = v

        if n <= 50000:
            train_content.append(new_dict)
        else:
            val_content.append(new_dict)

with open(output_file1, "w", encoding="utf-8") as f:
    for each in train_content:
        f.write(json.dumps(each, ensure_ascii=False) + "\n")

with open(output_file2, "w", encoding="utf-8") as _f:
    for each in val_content:
        _f.write(json.dumps(each, ensure_ascii=False) + "\n")
