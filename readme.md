# SketchACAD

[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

### Introduction

SketchACAD is an AI framework for automatically generating and understanding 3D CAD models from 2D sketches. It leverages deep learning techniques to bridge the gap between 2D sketch inputs and precise 3D CAD models, supporting various CAD operations like extrusion and revolve.

### Project Structure

- `data/`: Contains the dataset 
  - `sketchacad/`: Processed dataset with JSON, vector representations, STEP files, STL meshes, and PLY point clouds 
- `src/`: Source code
  - `CadSeqProc/`: CAD sequence processing modules
    - Data conversion from JSON to vector representation
    - STEP/STL/PLY file generation
    - Utilities for CAD operations
  - `models/`: Neural network models
    - VQVAE (Vector Quantized Variational Autoencoder)
    - Sketch2CAD diffusion models
  - `train/`: Training scripts and utilities

### Data Processing Workflow

The data processing pipeline includes:

1. **Data Collection**: Source CAD models in JSON format
2. **Data Processing**: Converting source JSON to standardized formats
   ```bash
   python src/CadSeqProc/data_processor.py --input /path/to/json/files --output data/sketchacad --bit 8 --max_workers 12
   ```
3. **Data Conversion**: 
   - JSON to vector representation for model training
   - JSON to STEP/STL/PLY for visualization and validation
4. **Dataset Creation**: Splitting into train/test sets (80%/20%)

### Training Workflow

#### 1. VQVAE Training

Train the Vector Quantized VAE model to compress CAD sequences into a latent space:

```bash
python src/train/train_vqvae.py \
    --data_dir data/sketchacad \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --embed_dim 256 \
    --num_embeddings 1024 \
    --enc_layers 4 \
    --dec_layers 4 \
    --ca_level_start 0 \
    --num_heads 8 \
    --commitment_cost 0.25 \
    --decay 0.99 \
    --save_dir ./checkpoints \
    --log_dir ./logs \
    --use_tensorboard \
    --use_wandb
```

#### 2. Sketch2CAD Training

Train the Sketch2CAD model using the pretrained VQVAE:

```bash
python src/train/train_sketch2cad.py \
    --data_dir data/sketchacad \
    --vqvae_path /path/to/vqvae_checkpoints/best_model.pth \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --embed_dim 256 \
    --num_embeddings 1024 \
    --dit_model DiT-B/2 \
    --save_dir ./checkpoints \
    --log_dir ./logs \
    --use_tensorboard \
    --use_wandb
```

### Testing and Visualization

Test the model pipeline and visualize results:

```bash
python src/CadSeqProc/test_json2vec2json2step.py \
    --input /path/to/test/jsons \
    --output ./test_output \
    --bit 8
```

### Requirements

The main dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```
You will also need to install `pythonocc-core` separately.

<a name="chinese"></a>
## 中文

### 简介

SketchACAD 是一个基于人工智能的框架，用于从 2D 草图自动生成和理解 3D CAD 模型。它利用深度学习技术来桥接 2D 草图输入与精确 3D CAD 模型之间的差距，支持拉伸和旋转等各种 CAD 操作。

### 项目结构

- `data/`: 存放数据集（有点大）
  - `sketchacad/`: 处理后的数据集，包含 JSON、向量表示、STEP 文件、STL 网格和 PLY 点云
- `src/`: 源代码
  - `CadSeqProc/`: CAD 序列处理模块
    - 从 JSON 到向量表示的数据转换
    - STEP/STL/PLY 文件生成
    - CAD 操作的实用工具
  - `models/`: 神经网络模型
    - VQVAE (矢量量化变分自编码器)
    - Sketch2CAD 扩散模型
  - `train/`: 训练脚本和工具

### 数据处理流程

数据处理流程包括：

1. **数据收集**：JSON 格式的源 CAD 模型
2. **数据处理**：将源 JSON 转换为标准化格式
   ```bash
   python src/CadSeqProc/data_processor.py --input /path/to/json/files --output data/sketchacad
   ```
3. **数据转换**：
上述脚本会完成
   - JSON 到向量表示，用于模型训练
   - JSON 到 STEP/STL/PLY，用于可视化和验证，目前点云和mesh导出还有问题，为了节省空间，这部分先不做
4. **数据集创建**：分割为训练/测试集（80%/20%）

### 训练流程

#### 1. VQVAE 训练

训练矢量量化 VAE 模型，将 CAD 序列压缩到潜在空间：

```bash
python src/train/train_vqvae.py \
    --data_dir data/sketchacad \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --embed_dim 256 \
    --num_embeddings 1024 \
    --enc_layers 4 \
    --dec_layers 4 \
    --ca_level_start 0 \
    --num_heads 8 \
    --commitment_cost 0.25 \
    --decay 0.99 \
    --save_dir ./checkpoints \
    --log_dir ./logs \
    --use_tensorboard \
    --use_wandb
```

#### 2. Sketch2CAD 训练

使用预训练的 VQVAE 训练 Sketch2CAD 模型：

```bash
python src/train/train_sketch2cad.py \
    --data_dir data/sketchacad \
    --vqvae_path /path/to/vqvae_checkpoints/best_model.pth \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --embed_dim 256 \
    --num_embeddings 1024 \
    --dit_model DiT-B/2 \
    --save_dir ./checkpoints \
    --log_dir ./logs \
    --use_tensorboard \
    --use_wandb
```

### 测试和可视化

未完成，可以使用Vec重建CAD模型导出STEP文件然后对比

### 依赖项

主要依赖项列在 `requirements.txt` 中。使用以下命令安装它们：

```bash
pip install -r requirements.txt
```
需要额外安装 `pythonocc-core`

