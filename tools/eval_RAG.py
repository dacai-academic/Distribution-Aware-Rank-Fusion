import ast
import traceback
from openai import OpenAI
from sentence_transformers import CrossEncoder

SYSTEM_PROMPT = """You are a medical assistant, skilled at answering questions based on the available information.
Constraints:
    1.Answer only based on the provided information; do not fabricate any information.
    2.Respond to the user’s question by providing a judgment (yes / no / unknown) and the corresponding reason for that judgment.
    3.The reason for the judgment must be derived from the provided information.
    4.Reply with unknown only if the provided information is insufficient to answer the question.
    5.Ensure the output is strictly valid JSON, with no additional text or explanation, and can be parsed by Python’s json.loads method.
Please think step by step and respond in the following JSON format:
{"judgment": "unknown", "reason": "The available information can not support enough evidence."}
"""

PROMPT = """ User Query:
    [***user*query***]
Available Information:
[***available*information***]
"""
MAX_CACHE_DIALOGUE_ROUNDS = 1
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "put your key here"


def parse_out(out: str):
    standard_str = out.strip("\n").strip(' ')
    if standard_str[0] != '{' or standard_str[-1] != '}':
        ind1 = 0
        ind2 = len(standard_str) - 1
        for ind_, each_ in enumerate(standard_str):
            if each_ == '{':
                ind1 = ind_
                break

        for _ind, _each in enumerate(standard_str):
            if _each == '}':
                ind2 = _ind
        standard_str = standard_str[ind1:ind2 + 1]
    try:
        final_out = ast.literal_eval(standard_str)
        return final_out

    except Exception as e:
        print(str(e))
        raise RuntimeError("大模型幻觉,输出格式错误")


class ChatBot(object):
    def __init__(self, url=URL, api_key=API_KEY):
        self.base_url = url
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, query: str, info: str):
        query_content = PROMPT.replace("[***user*query***]", query)
        query_content = query_content.replace("[***available*information***]", info)
        try:
            completion = self.client.chat.completions.create(
                model="glm-4.7",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query_content}
                ],
                extra_body={"enable_thinking": True},
            )
            final_out = parse_out(completion.choices[0].message.content.strip())
            if final_out["judgment"].lower() not in ["yes", "no", "unknown"]:
                raise RuntimeError("大模型调回答幻觉!")

            print(final_out)
            final_out["code"] = 0
            return final_out

        except Exception as e:
            print(e)
            traceback.print_exc()
            return {"judgment": "大模型调用出现意外错误!", "reason": "error!", "code": -1}


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
    from core.softmax_rrf import RRF, SoftmaxRRF
    from eval import get_bge, get_embedding_model

    subprocess.run(["rm", "-rf", "./chroma_db"], check=True)

    data = MedicalDataset("/home/cai/project/bm25_embedding/data/val.jsonl")
    pubmed = data.content[4000:]
    contexts = []
    for item in pubmed:
        contexts.extend(item["context"]["contexts"])

    chatbot = ChatBot()
    reranker = Reranker()
    bm25 = BM25Retriever()
    # embedding_model = get_bge()
    embedding_model = get_embedding_model()
    cs = ChromaStore(persist_dir="./chroma_db", embedding_model=embedding_model)

    bm25.insert(contexts)
    source_data = ["" for i in contexts]
    cs.store(texts=contexts, source=source_data)

    hallucinate = 0
    reject = 0
    accuracy = 0
    GT = 0
    PT = 0
    TP = 0
    for each in tqdm(pubmed):
        q = each["question"]
        top_k_result = cs.similar_search(q)
        top_k_docs = top_k_result["documents"][0]
        top_k_socres = [1.0 - d for d in top_k_result["distances"][0]]
        top_n_results = bm25.similar_search(q)
        top_n_docs = top_n_results["docs"]
        top_n_scores = top_n_results["scores"]

        softmax_rrf = SoftmaxRRF([top_k_docs, top_n_docs, top_k_socres, top_n_scores])
        fusion_docs = softmax_rrf.get_docs()
        # rrf = RRF([top_k_docs, top_n_docs])
        # fusion_docs = rrf.get_docs()

        final_docs = reranker.rerank(fusion_docs, q)[:5]

        retrieved_info = ""
        for ind, text in enumerate(final_docs):
            retrieved_info += ("    (" + str(ind+1) + ") " + text.strip("\n") + "\n")

        llm_out = chatbot.chat(query=q, info=retrieved_info .strip("\n"))
        while llm_out["code"] != 0:
            llm_out = chatbot.chat(query=q, info=retrieved_info .strip("\n"))
        time.sleep(2)

        flag_h = True
        for doc in final_docs:
            if doc in each["context"]["contexts"]:
                flag_h = False
                break

        if flag_h and (llm_out["judgment"].lower() != "unknown"):
            hallucinate += 1

        if (not flag_h) and (llm_out["judgment"].lower() == "unknown"):
            hallucinate += 1

        if llm_out["judgment"].lower() == "unknown":
            reject += 1

        if (llm_out["judgment"].lower() == each["final_decision"].lower()) and (not flag_h):
            accuracy += 1

        if llm_out["judgment"].lower() == "yes":
            PT += 1

        if each["final_decision"].lower() == "yes":
            GT += 1
            if (llm_out["judgment"].lower() == "yes") and (not flag_h):
                TP += 1

    precision = TP / PT
    recall = TP / GT
    f1_score = 2 * (precision * recall) / (precision + recall)
    metric_dict = {
        "accuracy": accuracy / len(pubmed),
        "f1_score": f1_score,
        "Hallucination Rate": hallucinate / len(pubmed),
        "Rejection Rate": reject / len(pubmed)
    }
    print(metric_dict)
