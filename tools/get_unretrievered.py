import ast
import traceback
from sentence_transformers import CrossEncoder


class Reranker(object):
    def __init__(self, model_name=None):
        if not model_name:
            model_name = "/home/cai/project/bm25_embedding/models/bge-reranker-v2-m3"
        self.model = CrossEncoder(model_name)

    def rerank(self, documents: list, query):
        results = self.model.rank(query, list(set(documents)), return_documents=True)
        final_out = []
        for result in results:
            final_out.append(result["text"])

        return final_out


if __name__ == '__main__':
    import time
    import subprocess
    from tqdm import tqdm
    from core.datatset import MedicalDataset
    from core.chorma_store import ChromaStore
    from core.bm25 import BM25Retriever
    from core.fusion_model import FusionModel
    from core.softmax_rrf import RRF, SoftmaxRRF
    from eval import get_bge, get_embedding_model

    subprocess.run(["rm", "-rf", "./chroma_db_local"], check=True)

    data = MedicalDataset("/home/cai/project/bm25_embedding/data/val.jsonl")
    pubmed = data.content[4000:]
    contexts = []
    for item in pubmed:
        contexts.extend(item["context"]["contexts"])

    reranker = Reranker()
    bm25 = BM25Retriever()
    embedding_model = get_bge()
    # embedding_model = get_embedding_model()
    cs = ChromaStore(persist_dir="./chroma_db_local", embedding_model=embedding_model)

    bm25.insert(contexts)
    source_data = ["" for i in contexts]
    cs.store(texts=contexts, source=source_data)

    miss = 0
    for each in tqdm(pubmed):
        q = each["question"]
        top_k_result = cs.similar_search(q)
        top_k_docs = top_k_result["documents"][0]
        top_k_socres = [1.0 - d for d in top_k_result["distances"][0]]
        top_n_results = bm25.similar_search(q)
        top_n_docs = top_n_results["docs"]
        top_n_scores = top_n_results["scores"]

        rrf = RRF([top_k_docs, top_n_docs])
        fusion_docs = rrf.get_docs()
        # softmax_rrf = SoftmaxRRF([top_k_docs, top_n_docs, top_k_socres, top_n_scores])
        # fusion_docs = softmax_rrf.get_docs()
        final_docs = reranker.rerank(fusion_docs, q)[:5]

        flag = True
        for doc in final_docs:
            if doc in each["context"]["contexts"]:
                flag = False
                break
        if flag:
            miss += 1

    print({"miss": miss, "rate": miss/len(pubmed)})

