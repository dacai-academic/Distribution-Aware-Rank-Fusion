import subprocess
import torch
import json

from core.datatset import MedicalDataset
from core.chorma_store import ChromaStore
from core.bm25 import BM25Retriever
from core.metric import RetrieveMetric, MultiRetrieveMetric
from core.fusion_model import FusionModel
from core.model import DualFusionModel
from core.embedding_model import Embeddings
from core.softmax_rrf import RRF, SoftmaxRRF


def get_embedding_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load("/home/cai/project/bm25_embedding/checkpoint/fusion_model_m3_epoch6.pt", map_location='cpu')

    embedding_model = FusionModel()
    embedding_model.load_state_dict(state_dict)
    embedding_model.to(device)
    embedding_model.eval()
    return embedding_model


def get_dual_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load("/home/cai/project/bm25_embedding/checkpoint/mix_model_epoch3.pt", map_location='cpu')

    embedding_model = DualFusionModel()
    embedding_model.load_state_dict(state_dict)
    embedding_model.to(device)
    embedding_model.eval()
    return embedding_model


def get_bge():
    embedding_model = Embeddings("/home/cai/project/bm25_embedding/models/bge-m3").get_embedding_model()
    return embedding_model


def get_retrieve_metric():
    subprocess.run(["rm", "-rf", "./chroma_db"], check=True)
    # embedding_model = get_dual_model()
    embedding_model = get_embedding_model()
    # embedding_model = get_bge()
    cs = ChromaStore(persist_dir="./chroma_db", embedding_model=embedding_model)
    dataset = MedicalDataset("/home/cai/project/bm25_embedding/data/val.jsonl")
    retrieve_metric = RetrieveMetric()
    data = []
    docs = []
    ranked_docs = []
    for q, ans in dataset:
        data.append((q, ans))
        docs.append(ans)

    source_data = ["" for i in docs]
    cs.store(texts=docs, source=source_data)

    for q, _ in dataset:
        top_k_results = cs.similar_search(q, top_k=10)
        ranked_docs.append(top_k_results["documents"][0])

    metric = retrieve_metric.evaluate(data, ranked_docs)
    print(metric)


def get_retrieve_metric_mix():
    subprocess.run(["rm", "-rf", "./chroma_db"], check=True)
    # embedding_model = get_embedding_model()
    embedding_model = get_bge()
    # embedding_model = get_dual_model()

    bm25 = BM25Retriever()
    cs = ChromaStore(persist_dir="./chroma_db", embedding_model=embedding_model)

    content = []
    with open("/home/cai/project/bm25_embedding/data/bioasq.jsonl") as f:
        for line in f.readlines():
            tmp_dict = json.loads(line)
            content.append(tmp_dict)

    multi_retrieve_metric = MultiRetrieveMetric(k=3)
    contexts = []
    for item in content:
        contexts.extend(item["contexts"])

    bm25.insert(contexts)
    source_data = ["" for i in contexts]
    cs.store(texts=contexts, source=source_data)

    final_docs1 = []
    final_docs2 = []
    ground_truth = []
    for item_dict in content:
        q = item_dict["question"]
        c = item_dict["contexts"]
        top_k_result = cs.similar_search(q)
        top_k_docs = top_k_result["documents"][0]
        top_k_socres = [1.0 - d for d in top_k_result["distances"][0]]
        top_n_results = bm25.similar_search(q)
        top_n_docs = top_n_results["docs"]
        top_n_scores = top_n_results["scores"]

        ground_truth.append(c)

        softmax_rrf = SoftmaxRRF([top_k_docs, top_n_docs, top_k_socres, top_n_scores])
        final_docs1.append(softmax_rrf.get_docs())

        rrf = RRF([top_k_docs, top_n_docs])
        final_docs2.append(rrf.get_docs())

    print("------softmax rrf------")
    metric1 = multi_retrieve_metric.evaluate(final_docs1, ground_truth)
    print(metric1)

    print("------rrf------")
    metric2 = multi_retrieve_metric.evaluate(final_docs2, ground_truth)
    print(metric2)


if __name__ == '__main__':
    get_retrieve_metric_mix()

