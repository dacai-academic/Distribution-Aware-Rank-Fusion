import numpy as np

from typing import List


class RRF(object):
    def __init__(self,  docs: List[List[str]]):
        self.docs = docs

    def _fusion(self):
        fusion_docs = {}
        for doc_list in self.docs:
            for ind, doc in enumerate(doc_list):
                if doc not in fusion_docs.keys():
                    fusion_docs[doc] = (1.0 / (ind + 61))  # 60 + 1, k=60
                else:
                    fusion_docs[doc] += (1.0 / (ind + 61))
        rank_docs = self._sort(fusion_docs)

        return rank_docs

    def _sort(self, fusion_docs: dict):
        sorted_keys = sorted(fusion_docs, key=fusion_docs.get, reverse=True)
        return sorted_keys

    def get_docs(self):
        return self._fusion()


class SoftmaxRRF(object):
    def __init__(self,  docs: List[List]):
        self.docs = docs

    def _fusion(self):
        assert len(self.docs) == 4, "check length of docs!"
        fusion_docs = {}
        tmp_docs = self.docs[:2]
        tmp_scores = self.docs[2:]
        w = [0.7, 0.3]
        t = [0.8, 2.0]
        alpha = [0.02, 0.05]
        for i in range(len(tmp_scores)):
            for j in range(len(tmp_scores[i])):
                tmp_scores[i][j] = tmp_scores[i][j] / t[i] - alpha[i] * (j + 1)

        for ind1, doc_list in enumerate(tmp_docs):
            for ind2, doc in enumerate(doc_list):
                if doc not in fusion_docs.keys():
                    fusion_docs[doc] = (np.exp(tmp_scores[ind1][ind2]) / np.sum(np.exp(tmp_scores[ind1]))) * (w[ind1] / (61 + ind2))  # 60 + 1, k=60
                else:
                    fusion_docs[doc] += ((np.exp(tmp_scores[ind1][ind2]) / np.sum(np.exp(tmp_scores[ind1]))) * (w[ind1] / (61 + ind2)))  # 60 + 1, k=60

        rank_docs = self._sort(fusion_docs)

        return rank_docs

    def _sort(self, fusion_docs: dict):
        sorted_keys = sorted(fusion_docs, key=fusion_docs.get, reverse=True)
        return sorted_keys

    def get_docs(self):
        return self._fusion()


if __name__ == '__main__':
    rrf = RRF([["1", "3", "2", "4"], ["2", "3", "4", "1"]])
    a = rrf.get_docs()
    b = 1