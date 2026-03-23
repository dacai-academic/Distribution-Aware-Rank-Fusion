import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist

from torch.cuda.amp import autocast, GradScaler

from core.model import DualFusionModel, tokenizer_bge, tokenizer_m3
from core.datatset import MedicalDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from core.loss import multiple_negatives_loss


NUM_EPOCHS = 3
BATCH_SIZE = 4
LOG_INTERVAL = 10


def collate_fn(batch):
    questions, answers = zip(*batch)
    texts = list(questions) + list(answers)
    B = len(questions)

    batch_bge = tokenizer_bge(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    batch_m3 = tokenizer_m3(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    return {
        "bge": batch_bge,
        "m3": batch_m3,
        "batch_size": B
    }


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    train_data = MedicalDataset("/home/cai/project/bm25_embedding/data/train.jsonl")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    model = DualFusionModel().to(device)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    model.train()

    proj_params = []
    other_params = []
    for name, param in model.named_parameters():
        # if not param.requires_grad:
        #     continue

        if 'bge_proj' in name or 'm3_proj' in name:
            proj_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": proj_params, "lr": 2e-5, "weight_decay": 1e-2},
        {"params": other_params, "lr": 2e-5, "weight_decay": 0.0},
    ])

    num_training_steps = len(train_dataloader) * NUM_EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    scaler = GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            batch_bge = {k: v.to(device) for k, v in batch["bge"].items()}
            batch_m3 = {k: v.to(device) for k, v in batch["m3"].items()}

            optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                emb = model(batch_bge, batch_m3)  # [2B, D]
                B = batch["batch_size"]
                q_emb = emb[:B]
                a_emb = emb[B:]

                # InfoNCE / CrossEntropy Loss
                loss = multiple_negatives_loss(q_emb, a_emb)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            unused_params = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is None:
                    unused_params.append(name)

            if unused_params:
                print("Unused parameters in this batch:", unused_params)

            if (((step + 1) % LOG_INTERVAL == 0) or (step == len(train_dataloader) - 1)) and int(os.environ["RANK"]) == 0:
                print(f"Epoch: {epoch+1}/{NUM_EPOCHS}, step: {step+1}/{len(train_dataloader)}, loss: {loss.item():.4f}, learning rate: {scheduler.get_last_lr()[0]:.2e}")

        if (epoch % 1 == 0) and int(os.environ["RANK"]) == 0:
            torch.save(model.module.state_dict(), "./checkpoint/fusion_model_epoch{}.pt".format(epoch+1))

    dist.barrier()  # 确保所有进程都停止于此
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
