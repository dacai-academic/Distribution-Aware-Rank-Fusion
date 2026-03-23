import os
import torch
import threading

from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import Optional
from core.fusion_model import FusionModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("/home/cai/project/bm25_embedding/checkpoint/fusion_model_m3_epoch2.pt", map_location='cpu')

new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v

embedding_model = FusionModel()
embedding_model.load_state_dict(new_state_dict)
embedding_model.to(device)
embedding_model.eval()


class ChromaStore(object):
    def __init__(self, persist_dir: str, embedding_model_name=None):
        self.persist_dir = persist_dir

        self.chroma_settings = Settings(persist_directory=self.persist_dir, anonymized_telemetry=False)
        self.chroma_client = PersistentClient(path=self.persist_dir, settings=self.chroma_settings)
        self.collection = self.chroma_client.get_or_create_collection(
            name="sale_qa",
            embedding_function=None,
            metadata={"hnsw:space": "cosine"})
        self.embeddings = embedding_model
        self.mutex = threading.Lock()
        self._ids = self.get()["ids"]
        self.insert_id = int(self.get()["ids"][-1]) if len(self._ids) > 0  else 0

    def _split_list(self, tmp: list):
        list_len = len(tmp)
        assert list_len != 0, "make sure the $tmp(list) exist data!"
        out = []
        count = list_len // 5000 + 1
        for i in range(count):
            start_index = i * 5000
            if (start_index + 5000) >= list_len:
                out.append(tmp[start_index:])
                break

            out.append(tmp[start_index: start_index+5000])
        return out

    def _add_text(self, texts: list, source: list):
        meta_datas = [{"source": i} for i in source]
        ids = []
        with self.mutex:
            for _ in range(len(texts)):
                self.insert_id += 1
                ids.append(str(self.insert_id))

            _texts = self._split_list(texts)
            _meta_datas = self._split_list(meta_datas)
            _ids = self._split_list(ids)

            for index in range(len(_texts)):
                embeddings = self.embeddings.encode(_texts[index])
                self.collection.add(metadatas=_meta_datas[index],
                                    embeddings=embeddings,
                                    documents=_texts[index],
                                    ids=_ids[index])
        return dict(zip(ids, texts))

    def _query(self, text: str, top_k: int = 3):
        query_embedding = self.embeddings.encode(text)
        with self.mutex:
            return self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k)

    def similar_search(self, text: str, top_k: int = 5):
        chroma_results = self._query(text=text, top_k=top_k)
        return chroma_results

    def store(self, texts: list, source: list):
        tmp = self._add_text(texts, source)
        return tmp

    def get(self, ids: Optional[list[str]] = None, max_limit: Optional[int] = None):
        with self.mutex:
            return self.collection.get(ids=ids, limit=max_limit)

    def find(self, include_text: Optional[str] = None, max_limit: Optional[int] = None):
        with self.mutex:
            if include_text:
                return self.collection.get(where_document={"$contains": include_text}, limit=max_limit)
            else:
                return self.collection.get(limit=max_limit)

    def insert(self, texts: list, source: list):
        texts_len = len(texts)
        meta_datas = [{"source": i} for i in source]
        ids = []

        with self.mutex:
            for _ in range(texts_len):
                self.insert_id += 1
                ids.append(str(self.insert_id))

            _texts = self._split_list(texts)
            _meta_datas = self._split_list(meta_datas)
            _ids = self._split_list(ids)

            for index in range(len(_texts)):
                embeddings = self.embeddings.encode(_texts[index])
                self.collection.add(metadatas=_meta_datas[index],
                                    embeddings=embeddings,
                                    documents=_texts[index],
                                    ids=_ids[index])
            return dict(zip(ids, texts))

    def delete(self, ids: Optional[list[str]] = None, include_text: Optional[str] = None):
        with self.mutex:
            if include_text:
                self.collection.delete(ids=ids, where_document={"$contains": include_text})
            else:
                self.collection.delete(ids=ids)
        return None

    def update(self, ids: list[str], texts: list, source: list):
        assert len(ids) <= 1000, "in the once update process, records should lower than 1000!"
        meta_datas = [{"source": i} for i in source]
        embeddings = self.embeddings.encode(texts)
        with self.mutex:
            self.collection.upsert(metadatas=meta_datas,
                                   embeddings=embeddings,
                                   documents=texts,
                                   ids=ids)
        return dict(zip(ids, texts))


if __name__ == "__main__":
    import time
    from core.parse_docs import Documents
    docs = Documents(file_path="/home/cai/project/bm25_embedding/data/qa.xlsx")
    text = []
    source = []
    for i, j in docs.content:
        text.append(i)
        source.append("q")
        text.append(j)
        source.append("a")

    cs = ChromaStore(persist_dir="./chroma_db")
    cs.insert(text, source)
    cs.insert(["Diabetes", "what a nice weather!", "today is sunny!", "阳光明媚", "雨天", "安达唐"], ["天气怎么样", "what a nice weather!", "today is sunny!", "阳光明媚", "雨天", "rain"])
    res = cs.similar_search("好大的雨", top_k=10)
    pass
    # data = [str(i).zfill(8) for i in range(10001)]
    # t1 = time.time()
    # cs.insert(data, data)
    # t2 = time.time()
    # print(t2 - t1)
    # t3 = time.time()
    # bbb = cs.get()
    # t4 = time.time()
    # print(t4 - t3)


