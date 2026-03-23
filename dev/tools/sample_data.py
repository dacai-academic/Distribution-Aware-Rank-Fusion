import random
import json

from core.datatset import MedicalDataset


if __name__ == '__main__':
    dataset = MedicalDataset("/home/cai/project/bm25_embedding/data/huatuo_lite.jsonl")
    data = dataset.content
    random.shuffle(data)
    with open("/home/cai/project/bm25_embedding/data/part_huato.jsonl", "w") as f:
        for each in data[:54000]:
            f.write(json.dumps(each, ensure_ascii=False) + "\n")

