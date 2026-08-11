import torch
import torch.nn as nn
import torch.nn.functional as F

from core.embedding_model import Embeddings


class GroupGatedFusion(nn.Module):
    def __init__(self, hidden_dim=1024, groups=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % groups == 0, "hidden_dim % groups should equals 0!"

        self.groups = groups
        self.group_dim = hidden_dim // groups

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, groups),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )

    def forward(self, bge_h, m3_h):
        """
        bge_h, m3_h: (B, hidden_dim)
        """
        concat = torch.cat([bge_h, m3_h], dim=-1)  # (B, 2*hidden_dim)
        g = self.gate(concat)                         # (B, groups)

        # expand to (B, hidden_dim)
        g = g.unsqueeze(-1).repeat(1, 1, self.group_dim)
        g = g.view(bge_h.size())

        fused = g * bge_h + (1.0 - g) * m3_h
        return fused


class FusionModel(nn.Module):
    def __init__(self, hidden_dim=1024, groups=8, dropout=0.1, freeze_encoders=True):
        super().__init__()
        self.bge = Embeddings("/home/cai/project/bm25_embedding/checkpoint/bge-large-zh-finetuned/checkpoint-4686").get_embedding_model()
        self.m3 = Embeddings("/home/cai/project/bm25_embedding/checkpoint/m3-finetuned/checkpoint-4686").get_embedding_model()
        self.bge_proj = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.m3_proj = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.gate = GroupGatedFusion(
            hidden_dim=hidden_dim,
            groups=groups,
            dropout=dropout
        )

        if freeze_encoders:
            for p in self.bge.parameters():
                p.requires_grad = False

            for p in self.m3.parameters():
                p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            bge_emb = self.bge.encode(x, convert_to_tensor=True).clone().detach()
            m3_emb = self.m3.encode(x, convert_to_tensor=True).clone().detach()

        bge_emb = self.bge_proj(bge_emb)
        m3_emb = self.m3_proj(m3_emb)

        fused = self.gate(bge_emb, m3_emb)
        fused = F.normalize(fused, dim=-1)
        return fused

    def encode(self, x):
        # encode
        return self.forward(x).detach().cpu().numpy()


if __name__ == '__main__':
    fm = FusionModel()
    pass
