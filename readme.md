# OmniCAD

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

### Introduction

OmniCAD is an AI framework for generating 3D CAD models from various multi-modal inputs, including B-Rep data, point clouds, multi-view images, and text descriptions. It leverages diffusion models to produce CAD outputs, utilizing a pre-trained VAE for latent space representation of CAD sequences.

### Project Structure

- `data/`: Contains the dataset
  - `sketchacad/`: Processed dataset with JSON, vector representations, STEP files, STL meshes, and PLY point clouds
- `src/`: Source code
  - `CadSeqProc/`: CAD sequence processing modules
    - Data conversion from JSON to vector representation
    - STEP/STL/PLY file generation
    - Utilities for CAD operations
  - `models/`: Neural network models
    - `MultiModalOmniCAD`: The core multi-modal diffusion model.
    - `VAE`: Variational Autoencoder for latent space representation of CAD sequences (pre-trained and frozen for OmniCAD).
    - Modality Encoders:
        - `BRepEncoder`: Processes B-Rep data using UVNet components.
        - `PointCloudEncoder`: Processes point cloud data using PointNet++.
        - `MultiViewImageEncoder`: Processes multi-view images using DINOv2 or a similar Vision Transformer.
        - `TextEncoder`: Processes textual descriptions using BERT.
    - `OminiCADDiffusion`: Diffusion model (e.g., DiT-based) for the generative process in the latent space.
    - `AE`: Autoencoder (can be used for pre-training or simpler tasks).
  - `train/`: Training scripts and utilities

### Data Processing Workflow

The data processing pipeline includes:

1.  **Data Collection**: Source CAD models in JSON format (or other raw formats convertible to CAD sequences).
2.  **Data Processing**: Converting source JSON to standardized CAD sequence formats (vector representations).
    ```bash
    python src/CadSeqProc/data_processor.py --input /path/to/json/files --output data/sketchacad --bit 8 --max_workers 12
    ```
3.  **Data Conversion**:
    - JSON to vector representation for model training (target for VAE, and subsequently for OmniCAD).
    - JSON to STEP/STL/PLY for visualization and validation.
4.  **Dataset Creation**: Splitting into train/test sets (80%/20%).

### Training Workflow

The training scripts for OmniCAD are currently under development and not yet implemented.

Previously, individual components like AE and VAE were trained as follows (these are for reference and pre-training of VAE):

#### 1. Autoencoder (AE) Training (Optional Pre-training/Simpler Tasks)
```bash
Example command for AE training (adjust parameters as needed)
python src/train/train_ae.py \
     --data_dir data/sketchacad \
     --embed_dim 256 \
     --latent_dim 256 \
     --enc_layers 4 \
     --dec_layers 4 \
     --num_heads 8 \
     --save_dir ./checkpoints/ae \
     --log_dir ./logs/ae \
     --use_tensorboard
```

#### 2. Variational Autoencoder (VAE) Training (To obtain the frozen VAE for OmniCAD)
```bash
Example command for VAE training (adjust parameters as needed)
python src/train/train_vae.py \
    --data_dir data/sketchacad \
    --embed_dim 256 \
    --latent_dim 256 \
    --enc_layers 4 \
    --dec_layers 4 \
    --num_heads 8 \
    --kl_weight 0.1 \
    --save_dir ./checkpoints/vae \
    --log_dir ./logs/vae \
    --use_tensorboard
```

### Testing and Visualization
Testing and visualization scripts for OmniCAD are under development.

### Requirements

The main dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```
You will also need to install `pythonocc-core` separately.

<a name="chinese"></a>
## 中文

### 简介

OmniCAD 是一个人工智能框架，旨在从多种模态输入（包括 B-Rep 数据、点云、多视图图像和文本描述）生成 3D CAD 模型。它利用扩散模型来生成 CAD 输出，并使用预训练的 VAE 进行 CAD 序列的潜空间表示。

### 项目结构

- `data/`: 存放数据集
  - `sketchacad/`: 处理后的数据集，包含 JSON、向量表示、STEP 文件、STL 网格和 PLY 点云
- `src/`: 源代码
  - `CadSeqProc/`: CAD 序列处理模块
    - 从 JSON 到向量表示的数据转换
    - STEP/STL/PLY 文件生成
    - CAD 操作的实用工具
  - `models/`: 神经网络模型
    - `VAE`: 用于 CAD 序列潜空间表示的变分自编码器（为 OmniCAD 预训练并冻结参数）。
    - `OmniCAD`: 核心的多模态扩散模型。
    - 模态编码器:
        - `BRepEncoder`: 使用 UVNet 组件处理 B-Rep 数据。
        - `PointCloudEncoder`: 使用 PointNet++ 处理点云数据。
        - `MultiViewImageEncoder`: 使用 DINOv2 或类似的 Vision Transformer 处理多视图图像。
        - `TextEncoder`: 使用 BERT 处理文本描述。
        - `OminiCADDiffusion`: 基于扩散模型（例如 DiT）的生成过程，在潜空间中操作。
    - `AE`: 自编码器（可用于预训练或较简单的任务）。
  - `train/`: 训练脚本和工具

### 数据处理流程

数据处理流程包括：

1.  **数据收集**：JSON 格式的源 CAD 模型（或其他可转换为 CAD 序列的原始格式）。
2.  **数据处理**：将源 JSON 转换为标准化的 CAD 序列格式（向量表示）。
    ```bash
    python src/CadSeqProc/data_processor.py --input /path/to/json/files --output data/sketchacad
    ```
3.  **数据转换**：
    上述脚本会完成
    - JSON 到向量表示，用于模型训练（作为 VAE 的目标，并随后用于 OmniCAD）。
    - JSON 到 STEP/STL/PLY，用于可视化和验证。
4.  **数据集创建**：分割为训练/测试集（80%/20%）。

### 训练流程

OmniCAD 的训练脚本目前正在开发中，尚未实现。

作为参考，之前像 AE 和 VAE 这样的独立组件是这样训练的（这些命令用于 VAE 的预训练）：

#### 1. 自编码器 (AE) 训练 (可选的预训练/较简单任务)
```bash
AE 训练示例命令 (根据需要调整参数)
python src/train/train_ae.py \
    --data_dir data/sketchacad \
    --embed_dim 256 \
    --latent_dim 256 \
    --enc_layers 4 \
    --dec_layers 4 \
    --num_heads 8 \
    --save_dir ./checkpoints/ae \
    --log_dir ./logs/ae \
    --use_tensorboard
```

#### 2. 变分自编码器 (VAE) 训练 (为 OmniCAD 获取冻结的 VAE)
```bash
VAE 训练示例命令 (根据需要调整参数)
python src/train/train_vae.py \
    --data_dir data/sketchacad \
    --embed_dim 256 \
    --latent_dim 256 \
    --enc_layers 4 \
    --dec_layers 4 \
    --num_heads 8 \
    --kl_weight 0.1 \
    --save_dir ./checkpoints/vae \
    --log_dir ./logs/vae \
    --use_tensorboard
```


### 测试和可视化

OmniCAD 的测试和可视化脚本有待开发。
```

### 依赖项

主要依赖项列在 `requirements.txt` 中。使用以下命令安装它们：

```bash
pip install -r requirements.txt
```
需要额外安装 `pythonocc-core` `occwl`


#从向量转换到
python src/CadSeqProc/json2vec_converter.py -i /root/autodl-tmp/SketchACAD/data/sketchacad/json -o /root/autodl-tmp/SketchACAD/data/sketchacad --max_workers 15

python src/CadSeqProc/split_dataset.py -i /root/autodl-tmp/SketchACAD/data/sketchacad -o /root/autodl-tmp/SketchACAD/data/sketchacad --train_ratio 0.8 --stratified

python src/train/train_vae.py --data_dir data/sketchacad --use_tensorboard