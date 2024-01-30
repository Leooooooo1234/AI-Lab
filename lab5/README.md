# README

```bash
pip install -r requirements.txt
```

## 训练

多模态融合模型

```bash
python ./main.py --multi_or_single multi
```

消融实验(图像)

```bash
python ./main.py --multi_or_single image
```

消融实验(文本)

```bash
python ./main.py --multi_or_single text
```

实验结果（验证集）：

|    指标    | Accuracy |   F1   | Recall | Precision |
| :--------: | :------: | :----: | ------ | --------- |
|   Image    |  63.37%  | 48.93% | 47.53% | 55.27%    |
|    Text    |   70%    | 62.25% | 55.5%  | 57.21%    |
| MultiModal |  72.38%  | 62.14% | 60.52% | 64.95%    |

## 测试

```bash
python ./main.py --test_or_train test
```







