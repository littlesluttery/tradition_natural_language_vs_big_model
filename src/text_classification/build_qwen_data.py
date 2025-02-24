"""
构造qwen模型微调所需要的数据
[
    {
        'id': 0,
        'conversations':
            [
                {'from': 'human','value': ""},
               {'from': 'gpt', 'value': '体育'}
            ],
        'system': 'You are a helpful assistant.'
    }
]

"""
import json


def get_human_prompt(text):
    """构造输入的提示"""
    prompt = """你是一个文本分类专家，你具备超强的语义理解能力，能对大量信息进行总结提取，请根据给出的内容和分类标签对内容进行分类。
请你直接输出具体的类型，不需要过多解释。
以下为分类类型：
{'教育', '时尚', '时政', '科技', '体育', '娱乐', '财经', '家居', '房产', '游戏'}
内容：
    """ + text
    return prompt


def get_gpt_prompt(label):
    """构造输出的提示，直接返回结果"""
    return label


def save_json(data, output_file_path):
    """保存训练数据集为json格式"""
    with open(output_file_path, 'w', encoding='utf-8')  as file:
        json.dump(data, file, ensure_ascii=False)


def build_data(file_path, output_file_path):
    """对于输入的txt文本进行处理，构造训练所需数据格式"""
    # 数据加载
    with open(file_path, encoding="utf-8") as file:
        file_data = file.readlines()

    res_list = []
    i = 0
    label_set = set()
    for line in file_data:
        middle_dict = {}
        conversations = []
        human_dict = {}
        gpt_dict = {}

        label = line.split("	")[0]
        text = line.split("	")[1]
        label_set.add(label)

        human_dict["from"] = "human"
        human_dict["value"] = get_human_prompt(text)

        gpt_dict["from"] = "gpt"
        gpt_dict["value"] = get_gpt_prompt(label)

        conversations.append(human_dict)
        conversations.append(gpt_dict)

        middle_dict["id"] = i
        middle_dict["conversations"] = conversations
        middle_dict["system"] = "You are a helpful assistant."

        i += 1
        res_list.append(middle_dict)
    save_json(res_list, output_file_path)
    print("数据保存完成")


if __name__ == "__main__":
    import os
    now_file = os.getcwd()
    file_path = os.path.join(now_file,"../../data/text_classification/cnews.train.txt")
    output_file_path = os.path.join(now_file,"../../data/text_classification/train_classification.json")
    build_data(file_path, output_file_path)
