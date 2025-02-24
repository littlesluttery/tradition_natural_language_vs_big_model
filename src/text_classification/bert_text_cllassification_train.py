"""
基于bert实现中文文本分类任务
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

now_path = os.getcwd()


def build_ori_data(
        file_path: str,
        file_type: str
):
    # 数据加载
    with open(file_path, encoding="utf-8") as file:
        file_data = file.readlines()

    data = []
    for line in file_data:
        middle_dict = {}
        if file_type != "test":
            label = line.split("	")[0]
            text = line.split("	")[1]
            middle_dict['features'] = text
            middle_dict['label'] = label
            data.append(middle_dict)

    df = pd.DataFrame(data)
    return df


def preprocess_for_bert(
        tokenizer: BertTokenizer,
        data,
        labels,
        max_length=256
):
    input_ids = []
    attention_mask = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
            truncation=True
        )
        input_ids.append(encoded_sent["input_ids"])
        attention_mask.append(encoded_sent["attention_mask"])
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    return input_ids, attention_mask, labels


# 计算准确率
def flat_accuracy(
        preds,
        labels
):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train_and_evaluate(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        epochs
):
    best_val_accuracy = 0
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels
            )
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, 0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} / {epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        for batch in val_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device).long()
            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
            loss = outputs.loss
            total_eval_loss += loss.item()
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        avg_val_loss = total_eval_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {avg_val_accuracy:.4f}")

        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print("Model saved with best validation accuracy.")


if __name__ == "__main__":
    file_path = "../../data/text_classification/cnews.train.txt"
    file = os.path.join(now_path, file_path)
    train_df = build_ori_data(file, file_type="train")
    X = train_df['features'].values
    y = train_df['label'].values

    # 对标签进行编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # print(y_encoded)

    # 分割数据
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.1, random_state=42)
    base_model_path = "../../pre_model/bert-base-chinese"
    bert_base_chinese_path = os.path.join(now_path, base_model_path)
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(bert_base_chinese_path)
    # print(tokenizer)

    # 数据预处理
    # 训练数据
    train_inputs, train_masks, train_labels = preprocess_for_bert(
        tokenizer,
        X_train,
        y_train,
    )
    # 验证数据
    val_inputs, val_masks, val_labels = preprocess_for_bert(
        tokenizer,
        X_val,
        y_val
    )
    # 创建DataLoader
    train_data = TensorDataset(
        train_inputs,
        train_masks,
        train_labels
    )
    val_data = TensorDataset(
        val_inputs,
        val_masks,
        val_labels
    )
    train_dataloader = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=8
    )
    val_dataloader = DataLoader(
        val_data,
        sampler=SequentialSampler(val_data),
        batch_size=8
    )

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        bert_base_chinese_path,
        num_labels=len(label_encoder.classes_)
    ).to(device)

    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * 5
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    # 模型保存路径
    best_model_path = '../../train_model/text_classification_model/bert_model'
    best_model_path = os.path.join(now_path, best_model_path)
    # 训练和评估
    train_and_evaluate(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        epochs=5
    )
