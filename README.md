### 传统自然语言处理任务 VS 大模型
```markdown
1、文本分类
2、命名实体识别
3、关系抽取
4、情感分析
5、事件抽取
```

#### 中文文本分类
```markdown
数据集：THUCNews数据集

# 基于bert的文本分类模型
    1、数据集和模型: 
        在huggingface上下载THUCNews数据集，格式为txt，下载完成后将数据放在data/text_classification/下；
        在huggingface上下载bert模型，模型放在pre_model/bert-base-chinese/下；    
    2、bert模型训练：python bert_text_classification_train.py   
    3、测评：python predict.py

# 基于qwen2.5的文本分类
    1、数据集构造：python build_qwen_data.py
       
    2、模型微调： sh train_qwen.sh
            MODEL_NAME_OR_PATH：在huggingface或者modelscope上下载qwen模型，放在pre_model/下；
    3、测评
        3.1 拉起训练后的模型(采用fastchat拉起服务):
            下载fastchat源码到本地，替换其中vllm_worker.py 
            sh run_model.sh
            BEST_MODEL_CHECKPOINT:修改为你训练后的模型路径(需要merge)
        3.2 运行测评脚本: python predict_qwen.py 
            url:拉起的模型服务
```

#### 中文命名实体识别
```markdown

# 基于bert的中文命名实体识别
    1、数据集下载：下载中文人民日报数据集
    2、模型训练：python bert_ner_train.py
    3、模型测评：python predict.py

# 基于qwen2.5的中文命名实体识别
    1、数据集构造： python build_qwen_data.py
    2、模型微调： sh train_qwen.sh
    3、测评
       3.1 拉起训练后的模型：sh run_model.sh
       3.2 运行测评脚本


```


### 备注
```markdown
# 利用大模型写代码的提示词。
利用transformers和bert模型完成中文命名实体识别代码开发，数据集为中文人名日报，格式为txt，采用BIO标注。
包括模型训练，测试和模型保存，以及单条文本的测评。要求和transformers的trainer.py格式相同。不要直接调用，需要实现模型训练过程和验证。

# 核心包及其版本
transformers 4.47.1
torch 2.5.1
vllm 0.6.4
ms-swift 3.1.0
```
