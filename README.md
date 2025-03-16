# LLM微调项目

这个项目使用[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)在Google Colab上对大型语言模型进行微调。

## 项目结构

- `llama_factory_finetune.ipynb`: 在Google Colab上运行的主要笔记本文件
- `data/`: 存放训练数据的目录
- `scripts/`: 辅助脚本目录，用于数据准备和处理
- `models/`: 存放微调后模型的目录

## 使用方法

1. 打开`llama_factory_finetune.ipynb`笔记本
2. 上传到Google Colab
3. 按照笔记本中的说明执行单元格
4. 完成微调后，可以从`models/`目录下载微调后的模型

## 依赖

- Python 3.8+
- PyTorch
- Transformers
- 其他依赖项会在笔记本中自动安装 