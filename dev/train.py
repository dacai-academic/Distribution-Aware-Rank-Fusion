import torch

from torch import nn
from torch.utils.data import DataLoader
# from sentence_transformers import InputExample
from core.fusion_model import FusionModel
# from core.parse_docs import Documents
from core.datatset import TextGenerator, MedicalDataset
from core.loss import multiple_negatives_loss
from torch.optim.lr_scheduler import CosineAnnealingLR


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_data = TextGenerator("/home/cai/project/bm25_embedding/data/qa.xlsx")
    train_data = MedicalDataset()
    train_dataloader = DataLoader(train_data, batch_size=4096, shuffle=True, drop_last=True)
    model = FusionModel().to(device)
    model.train()

    proj_params = []
    gate_params = []
    norm_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "gate" in name:
            gate_params.append(param)
        elif "LayerNorm" in name or "bias" in name:
            norm_params.append(param)
        else:
            proj_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": proj_params, "lr": 2e-4, "weight_decay": 1e-2},
            {"params": gate_params, "lr": 1e-4, "weight_decay": 0.0},
            {"params": norm_params, "lr": 2e-4, "weight_decay": 0.0},
        ]
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_dataloader) * num_epochs,
        eta_min=1e-6
    )

    for epoch in range(1000):
        for step, batch in enumerate(train_dataloader):
            sentences1 = batch[0]
            sentences2 = batch[1]

            emb1 = model(sentences1)
            emb2 = model(sentences2)

            # InfoNCE / CrossEntropy Loss
            loss = multiple_negatives_loss(emb1, emb2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), "fusion_model.pt")


if __name__ == '__main__':
    main()