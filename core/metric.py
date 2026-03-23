import math

from tqdm import tqdm


class RetrieveMetric(object):
    def __init__(self, top_k: list[int] = [3, 5, 10]):
        self.top_k = top_k

    def eval_one_query(self, ranked_docs: list, gt_doc: str):
        if gt_doc in ranked_docs:
            rank = ranked_docs.index(gt_doc) + 1
            rr = 1.0 / rank
            recalls = {k: int(rank <= k) for k in self.top_k}
        else:
            rr = 0.0
            recalls = {k: 0 for k in self.top_k}

        return rr, recalls

    def evaluate(self, data: list[tuple], ranked_docs: list[list]):
        assert len(data) == len(ranked_docs), "check your eval data"
        mrrs = []
        recalls = {k: [] for k in self.top_k}
        print("------eval data------")

        for ind, sample in tqdm(enumerate(data)):
            rr, rks = self.eval_one_query(ranked_docs[ind], sample[1])
            mrrs.append(rr)
            for k in rks:
                recalls[k].append(rks[k])

        return {
            "MRR": sum(mrrs) / len(mrrs),
            "Recall@3": sum(recalls[3]) / len(recalls[3]),
            "Recall@5": sum(recalls[5]) / len(recalls[5]),
            "Recall@10": sum(recalls[10]) / len(recalls[10]),
        }


class MultiRetrieveMetric(object):
    def __init__(self, k):
        self.top_k = k

    def average_precision(self, ranked_docs: list, relevant_docs: list):
        if not relevant_docs:
            return 0.0

        hit = 0
        score = 0.0
        cutoff = self.top_k

        for i, doc in enumerate(ranked_docs[:cutoff], start=1):
            if doc in relevant_docs:
                hit += 1
                score += hit / i
        return score / min(len(relevant_docs), self.top_k)

    def mean_average_precision(self, all_ranked_docs: list[list], all_relevant_docs: list[list]):
        ap_scores = []

        for ranked_docs, relevant_docs in zip(all_ranked_docs, all_relevant_docs):
            ap = self.average_precision(ranked_docs, relevant_docs)
            ap_scores.append(ap)

        return sum(ap_scores) / len(ap_scores)

    def dcg_at_k(self, rels):
        dcg = 0.0
        for i, rel in enumerate(rels[:self.top_k], start=1):
            dcg += (2 ** rel - 1) / math.log2(i + 1)
        return dcg

    def ndcg_at_k(self, ranked_docs: list, relevant_docs: list):
        rels = [1 if doc in relevant_docs else 0 for doc in ranked_docs]

        dcg = self.dcg_at_k(rels)

        # ideal relevance list
        ideal_rels = [1] * min(len(relevant_docs), self.top_k)
        idcg = self.dcg_at_k(ideal_rels)

        return dcg / idcg

    def mean_ndcg_at_k(self, all_ranked_docs: list[list], all_relevant_docs: list[list]):
        scores = []

        for ranked_docs, relevant_docs in zip(all_ranked_docs, all_relevant_docs):
            score = self.ndcg_at_k(ranked_docs, relevant_docs)
            scores.append(score)

        return sum(scores) / len(scores)

    def evaluate(self, ranked_docs: list[list], relevant_docs: list[list]):
        return {
            "MAP@3": self.mean_average_precision(ranked_docs, relevant_docs),
            "NDCG@3": self.mean_ndcg_at_k(ranked_docs, relevant_docs)
        }

