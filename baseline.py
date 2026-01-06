import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score

# ===== 数据集 =====
class FraudDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["is_fraud"])
        df["is_fraud"] = df["is_fraud"].astype(str).str.upper()
        df = df[df["is_fraud"].isin(["TRUE", "FALSE"])]

        self.texts = df["specific_dialogue_content"].tolist()
        self.labels = df["is_fraud"].map({"TRUE": 1, "FALSE": 0}).tolist()

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }

# ===== 加载模型 =====
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "hfl/chinese-bert-wwm-ext"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
).to(device)

# ===== 测试 =====
test_set = FraudDataset("data/test.csv", tokenizer)
test_loader = DataLoader(test_set, batch_size=16)

model.eval()
preds, labels = [], []

with torch.no_grad():
    for i, batch in enumerate(test_loader):

        labels_batch = batch["labels"].to(device)

        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device)
        }

        out = model(**inputs)
        pred = torch.argmax(out.logits, dim=1)

        preds.extend(pred.cpu().numpy())
        labels.extend(labels_batch.cpu().numpy())


print("Original Accuracy:", accuracy_score(labels, preds))
