import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

from torch_audit import Auditor, AuditConfig
from torch_audit.callbacks import HFAuditCallback


def run_demo():
    print("\n" + "=" * 60)
    print("🤗 TORCH-AUDIT: HUGGING FACE TRAINER DEMO")
    print("=" * 60)

    model_name = "distilbert-base-uncased"

    # 1. Setup Model & Tokenizer
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 2. Create Dummy Dataset
    # We create a dataset where some inputs are dangerously long or malformed
    data = [
        {"text": "This is a normal sentence.", "label": 0},
        {"text": "Another normal sentence for training.", "label": 1},
        {"text": "", "label": 0},
        # Issue: Very long nonsense to trigger compute waste
        {"text": "waste " * 500, "label": 1},
    ]

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    dataset = Dataset.from_list(data)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 3. Setup Auditor
    config = AuditConfig(
        monitor_nlp=True,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=tokenizer.vocab_size
    )
    auditor = Auditor(model, config=config)

    # 4. Define Trainer with Callback
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        report_to="none",
        use_cpu=not torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        # The Magic Line:
        callbacks=[HFAuditCallback(auditor)]
    )

    print("\n[Starting Fine-Tuning]...")
    trainer.train()


if __name__ == "__main__":
    run_demo()