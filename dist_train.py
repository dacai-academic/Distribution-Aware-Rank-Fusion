import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist

from core.fusion_model import FusionModel
from core.datatset import MedicalDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from core.loss import multiple_negatives_loss


NUM_EPOCHS = 3
BATCH_SIZE = 16
LOG_INTERVAL = 10


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    train_data = MedicalDataset("/home/cai/project/bm25_embedding/data/train.jsonl")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True, drop_last=True)
    model = FusionModel().to(device)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
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
            {"params": proj_params, "lr": 2e-5, "weight_decay": 1e-2},
            {"params": gate_params, "lr": 2e-5, "weight_decay": 0.0},
            {"params": norm_params, "lr": 2e-5, "weight_decay": 0.0},
        ]
    )

    num_training_steps = len(train_dataloader) * NUM_EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            sentences1 = batch[0]
            sentences2 = batch[1]

            emb1 = model(sentences1)
            emb2 = model(sentences2)

            # InfoNCE / CrossEntropy Loss
            loss = multiple_negatives_loss(emb1, emb2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            if (((step + 1) % LOG_INTERVAL == 0) or (step == len(train_dataloader) - 1)) and int(os.environ["RANK"]) == 0:
                print(f"Epoch: {epoch+1}/{NUM_EPOCHS}, step: {step+1}/{len(train_dataloader)}, loss: {loss.item():.4f}, learning rate: {scheduler.get_last_lr()[0]:.2e}")

        if (epoch % 1 == 0) and int(os.environ["RANK"]) == 0:
            torch.save(model.module.state_dict(), "./checkpoint/fusion_model_m3_epoch{}.pt".format(epoch+1))

    dist.barrier()  # 确保所有进程都停止于此
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
