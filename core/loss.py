import torch
import torch.nn.functional as F


def multiple_negatives_loss(q_emb, p_emb, temperature=0.05):
    """
    q_emb: (B, D)
    p_emb: (B, D)
    """
    scores = torch.matmul(q_emb, p_emb.T) / temperature  # (B, B)
    labels = torch.arange(scores.size(0), device=scores.device)
    loss = F.cross_entropy(scores, labels)
    return loss

