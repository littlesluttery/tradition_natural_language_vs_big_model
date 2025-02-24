"""文本分类模型测评"""
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch


class Traditional_Predict():
    def __init__(
            self,
            model_name_or_path,
            test_file_path
    ):
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.model = BertForSequenceClassification.from_pretrained(model_name_or_path)
        self.test_file_path = test_file_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_data(self):
        # 数据加载
        with open(self.test_file_path, encoding="utf-8") as file:
            file_data = file.readlines()

        data = []
        for line in file_data:
            middle_dict = {}
            label = line.split("	")[0]
            text = line.split("	")[1]
            middle_dict['text'] = text
            middle_dict['label'] = label
            data.append(middle_dict)

        test_df = pd.DataFrame(data)
        X_test = test_df["text"].values
        y_test = test_df["label"].values

        # 加载标签编码器
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)
        y_test_encoded = label_encoder.transform(y_test)
        return X_test, y_test_encoded, label_encoder

    def predict_dataset(
            self,
            texts,
    ):
        """对整个数据集进行预测。"""
        self.model.to(self.device)
        self.model.eval()
        predictions = []

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
                predictions.append(pred)

        return predictions

    def evaluate_model(
            self,
            predictions,
            true_labels,
            label_encoder
    ):
        """评估模型性能，计算准确率、召回率和F1值。"""
        # 将类别索引映射回类别名称
        true_labels = label_encoder.inverse_transform(true_labels)
        predictions = label_encoder.inverse_transform(predictions)

        # 计算分类报告
        report = classification_report(true_labels, predictions, target_names=label_encoder.classes_, digits=4)
        return report

    def predict(self):
        # 处理测试数据
        X_test, y_test_encoded, label_encoder = self.prepare_data()
        # 对测试集进行预测
        predictions = self.predict_dataset(X_test)

        # 评估模型
        report = self.evaluate_model(predictions, y_test_encoded, label_encoder)

        # 打印分类报告
        print("分类报告：")
        print(report)


if __name__ == "__main__":
    import os

    now_file = os.getcwd()
    test_file = os.path.join(now_file, "../..//data/text_classification/cnews.val.txt")
    model_name_or_path = os.path.join(now_file, "../../train_model/text_classification_model/bert_model")
    tp = Traditional_Predict(model_name_or_path, test_file)
    tp.predict()
