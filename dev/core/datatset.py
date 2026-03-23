import json

from torch.utils.data import Dataset

from core.parse_docs import Documents


class MedicalDataset(Dataset):
    def __init__(self, file_path: str):
        self.content = []
        self.get_data(file_path)

    def __getitem__(self, index):
        question = self.content[index]["question"]
        answer = self.content[index]["answer"]
        return question, answer

    def __len__(self):
        return len(self.content)

    def get_data(self, path: str):
        with open(path) as f:
            for line in f.readlines():
                tmp_dict = json.loads(line)
                self.content.append(tmp_dict)
        return self.content


class HFDataset:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> Dataset:
        data = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    "sentence1": "为这个句子生成表示用于检索相关文章：" + item["question"],
                    "sentence2": item["answer"]
                })
        from datasets import Dataset
        return Dataset.from_list(data)


if __name__ == '__main__':
    a = MedicalDataset("/home/cai/project/bm25_embedding/data/train.jsonl")
    b = a.content
    pass