from sentence_transformers import SentenceTransformer
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers import losses

from core.datatset import   HFDataset


def main():
    training_args = SentenceTransformerTrainingArguments(
        output_dir="./checkpoint/m3-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=True,
        dataloader_drop_last=True,
        logging_steps=50,
        save_strategy="epoch"
    )

    model = SentenceTransformer("models/bge-m3")
    dataset = HFDataset("/home/cai/project/bm25_embedding/data/train.jsonl").load()
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=losses.MultipleNegativesRankingLoss(model)
    )

    trainer.train()


if __name__ == "__main__":
    main()
