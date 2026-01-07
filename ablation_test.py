import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---------- 中文显示配置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 可用中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ---------- 数据集 ----------
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
        # 批量 encode，避免 __getitem__ 每次编码耗时
        self.encodings = tokenizer(self.texts,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=self.max_len,
                                   return_tensors='pt')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings['input_ids'][idx],
            "attention_mask": self.encodings['attention_mask'][idx],
            "labels": torch.tensor(self.labels[idx])
        }

# ---------- 测试函数 ----------
def test_model(csv_file, model, tokenizer, device, batch_size=16):
    try:
        dataset = FraudDataset(csv_file, tokenizer)
    except FileNotFoundError:
        print(f"{csv_file} 数据集不存在，跳过。")
        return None

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            out = model(**inputs)
            pred = torch.argmax(out.logits, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(labels, preds)
    return acc

# ---------- 主程序 ----------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    model_name = "hfl/chinese-bert-wwm-ext"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    ).to(device)

    results = {}
    # 原始测试集
    acc_orig = test_model("data/test.csv", model, tokenizer, device)
    if acc_orig is not None:
        results["原始测试集"] = acc_orig
        print(f"原始测试集 准确率: {acc_orig:.3f}")

    # 改写测试集
    acc_adv = test_model("data/test_adv.csv", model, tokenizer, device)
    if acc_adv is not None:
        results["改写测试集"] = acc_adv
        print(f"改写测试集 准确率: {acc_adv:.3f}")

    # 同义词替换
    acc_syn = test_model("data/test_synonym.csv", model, tokenizer, device)
    if acc_syn is not None:
        results["同义词替换"] = acc_syn
        print(f"同义词替换 准确率: {acc_syn:.3f}")

    # 整句改写
    acc_sent = test_model("data/test_sentence.csv", model, tokenizer, device)
    if acc_sent is not None:
        results["整句改写"] = acc_sent
        print(f"整句改写 准确率: {acc_sent:.3f}")

    # ---------- 可视化 ----------
    if results:
        plt.figure(figsize=(8,5))
        plt.bar(results.keys(), results.values(), color=['skyblue', 'salmon', 'lightgreen', 'orange'])
        plt.ylim(0,1)
        plt.ylabel("准确率")
        plt.title("不同测试集上的模型准确率对比")
        for i, v in enumerate(results.values()):
            plt.text(i, v+0.02, f"{v:.3f}", ha='center', fontweight='bold')
        plt.show()

if __name__ == "__main__":
    main()
