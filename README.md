## Introduction
一个使用 NumPy 从零手写实现 Transformer 的练手项目。
项目目标是：不依赖 PyTorch / TensorFlow 自动求导框架，完整实现 Transformer 的前向传播、反向传播、训练、保存、加载与推理翻译。

## Author
Mingwei Zhang - 13337649640@163.com

## System requirements:
`Windows` (tested on `Windows 11`).

## Model Parameters
layers = 2;
d_model = 128;
heads = 4;
d_ff = 256;

## Run code

训练模型:

```
python app.py train --epochs 15 --save_dir model_128_4_256
```
其中 --epochs 代表训练轮数，--save_dir 表示保存路径

加载模型并进行翻译:

```
python app.py translate --model_dir model_128_4_256
```
其中 --model_dir 代表加载模型的路径