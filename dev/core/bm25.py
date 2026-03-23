import re
import os
import jieba

import numpy as np

from rank_bm25 import BM25Okapi
from typing import Optional


def process_sentence(sentence: str):
    # with gpu device, jieba in paddle is better
    word_list = []
    tmp_list = jieba.lcut(sentence)
    for word in tmp_list:
        tmp = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5]+', '', word)
        if tmp:
            word_list.append(tmp)
    return word_list


class BM25(object):
    def __init__(self, docs: list):
        self.docs = docs
        self.tokenized_docs = []
        self.tokenized_docs = [process_sentence(doc) for doc in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def get_result(self, query: str, top_n: int = 5):
        tokenized_query = process_sentence(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return {
            "docs": [self.docs[i] for i in top_indices],
            "scores": [float(scores[i]) for i in top_indices]
        }


class BM25Retriever(object):
    def __init__(self):
        self.docs_list = []
        self.bm25 = None

    def update(self, docs_list: Optional[list[str]] = None):
        if docs_list is None:
            self.bm25 = None
            self.docs_list = []
            return []

        self.docs_list = docs_list
        self.bm25 = BM25(self.docs_list)
        return self.docs_list

    def insert(self, docs_list: list[str]):
        self.docs_list.extend(docs_list)
        self.bm25 = BM25(self.docs_list)
        return self.docs_list

    def similar_search(self, query: str, top_n: int = 5):
        if self.bm25 is None:
            return []

        return self.bm25.get_result(query, top_n=top_n)


if __name__ == '__main__':
    # process_sentence("漫步者@, 有线耳机metax-c500")
    bmr = BM25Retriever()
    b = bmr.similar_search("漫步者 有线耳机")
    c = 1
