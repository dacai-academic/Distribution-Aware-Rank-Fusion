import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


ROOT = os.path.abspath(__file__).split("core")[0]
tokenizer_bge = AutoTokenizer.from_pretrained(os.path.join(ROOT, "models/bge-large-zh"))
tokenizer_m3 = AutoTokenizer.from_pretrained(os.path.join(ROOT, "models/bge-m3"))


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


class DualFusionModel(nn.Module):
    def __init__(self, hidden_dim=1024, groups=8, dropout=0.1, freeze_encoders=False, normalize=True):
        super().__init__()
        self.freeze_encoders = freeze_encoders
        self.normalize = normalize

        self.bge = AutoModel.from_pretrained(os.path.join(ROOT, "models/bge-large-zh"))
        self.m3 = AutoModel.from_pretrained(os.path.join(ROOT, "models/bge-m3"))

        self.bge_proj = nn.Sequential(
            nn.Linear(self.bge.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.m3_proj = nn.Sequential(
            nn.Linear(self.m3.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.gate = GroupGatedFusion(hidden_dim=hidden_dim, groups=groups, dropout=dropout)

        if self.freeze_encoders:
            for p in self.bge.parameters():
                p.requires_grad = False
            for p in self.m3.parameters():
                p.requires_grad = False

    def forward(self, batch_bge, batch_m3):
        bge_emb = self.bge(**batch_bge).last_hidden_state[:, 0, :]  # [CLS] token
        m3_emb = self.m3(**batch_m3).last_hidden_state[:, 0, :]
        bge_emb = self.bge_proj(bge_emb)
        m3_emb = self.m3_proj(m3_emb)
        fused = self.gate(bge_emb, m3_emb)

        if self.normalize:
            fused = F.normalize(fused, dim=-1)
        return fused

    @torch.no_grad()
    def encode(self, question: str, device="cuda"):
        self.eval()
        if isinstance(question, str):
            texts = [question]
        elif isinstance(question, list):
            texts = question
        else:
            raise TypeError(f"question must be str or list[str], got {type(question)}")

        batch_bge = tokenizer_bge(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        batch_m3 = tokenizer_m3(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        return self.forward(batch_bge, batch_m3).detach().cpu().numpy()

