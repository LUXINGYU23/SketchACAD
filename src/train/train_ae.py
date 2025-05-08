import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

# 导入自定义模块
from src.models.ae import AE
from src.CadSeqProc.utility.macro import *
from src.CadSeqProc.utility.utils import ensure_dir

# 尝试导入wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("未安装wandb，将不使用wandb进行日志记录")


# 重用CADSequenceDataset类
class CADSequenceDataset(Dataset):
    """CAD序列数据集类，用于加载经过预处理的CAD序列向量"""
    
    def __init__(self, data_dir, file_list):
        """
        初始化数据集，并过滤掉超过MAX_CAD_SEQUENCE_LENGTH长度的样本
        
        Args:
            data_dir: 数据根目录，包含vec子目录
            file_list: 文件ID列表
        """
        self.data_dir = Path(data_dir)
        
        # 过滤超出MAX_CAD_SEQUENCE_LENGTH的样本
        filtered_list = []
        filtered_count = 0
        
        print(f"开始过滤超过{MAX_CAD_SEQUENCE_LENGTH}长度的样本...")
        for file_id in tqdm(file_list, desc="过滤样本"):
            # 加载CAD序列向量
            cad_data_path = self.data_dir / "vec" / f"{file_id}.pth"
            try:
                cad_data = torch.load(cad_data_path, map_location='cpu')
                cad_vec_length = cad_data["vec"]["cad_vec"].shape[0]
                flag_vec_length = cad_data["vec"]["flag_vec"].shape[0]
                index_vec_length = cad_data["vec"]["index_vec"].shape[0]
                
                if cad_vec_length <= MAX_CAD_SEQUENCE_LENGTH and \
                   flag_vec_length <= MAX_CAD_SEQUENCE_LENGTH and \
                   index_vec_length <= MAX_CAD_SEQUENCE_LENGTH:
                    filtered_list.append(file_id)
                else:
                    filtered_count += 1
            except Exception as e:
                print(f"加载文件 {file_id} 时出错: {e}")
                filtered_count += 1
                
        self.file_list = filtered_list
        print(f"过滤完成: 共过滤掉 {filtered_count} 个样本，保留 {len(filtered_list)} 个样本")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        
        # 加载CAD序列向量
        cad_data_path = self.data_dir / "vec" / f"{file_id}.pth"
        cad_data = torch.load(cad_data_path, map_location='cpu')
        
        # 构建返回字典
        return {
            "file_id": file_id,
            "vec_dict": {
                "cad_vec": cad_data["vec"]["cad_vec"],
                "flag_vec": cad_data["vec"]["flag_vec"],
                "index_vec": cad_data["vec"]["index_vec"]
            },
            "mask_dict": {
                "attn_mask": cad_data["mask_cad_dict"].get("attn_mask"),
                "key_padding_mask": cad_data["mask_cad_dict"]["key_padding_mask"]
            }
        }


def compute_cad_reconstruction_loss(output, target_vec, mask=None): # mask 预期为 (B, L, 2)
    """
    计算CAD序列的重建损失
    
    Args:
        output: 模型输出，形状为[B, L, 2, C]
        target_vec: 目标CAD向量，形状为[B, L, 2]
        mask: 可选的掩码，形状为[B, L, 2]，True表示填充
        
    Returns:
        loss: 重建损失值
    """
    # 分别计算X和Y坐标的交叉熵损失
    loss_x_all = F.cross_entropy(
        output[:, :, 0].contiguous().view(-1, output.size(-1)).float(),
        target_vec[:, :, 0].contiguous().view(-1).long(),
        reduction='none'
    ) # 形状: (B*L)
    
    loss_y_all = F.cross_entropy(
        output[:, :, 1].contiguous().view(-1, output.size(-1)).float(),
        target_vec[:, :, 1].contiguous().view(-1).long(),
        reduction='none'
    ) # 形状: (B*L)
    
    if mask is not None:
        # mask 是 (B, L, 2), True 表示填充
        # 我们只计算非填充位置的损失
        mask_x_flat = mask[:, :, 0].reshape(-1)  # (B*L)
        mask_y_flat = mask[:, :, 1].reshape(-1)  # (B*L)

        # 对填充位置的损失置零
        loss_x_masked = loss_x_all * (~mask_x_flat).float()
        loss_y_masked = loss_y_all * (~mask_y_flat).float()

        # 计算有效（非填充）位置的数量
        n_valid_x = (~mask_x_flat).sum()
        n_valid_y = (~mask_y_flat).sum()

        # 计算平均损失，防止除以零
        loss_x = loss_x_masked.sum() / n_valid_x.clamp(min=1.0)
        loss_y = loss_y_masked.sum() / n_valid_y.clamp(min=1.0)
    else:
        # 无掩码，直接平均
        loss_x = loss_x_all.mean()
        loss_y = loss_y_all.mean()
    
    # 组合最终损失
    loss = loss_x + loss_y
    
    return loss


def train_ae(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    scheduler=None, 
    num_epochs=100, 
    device='cuda', 
    save_dir='./checkpoints',
    log_dir='./logs',
    exp_name=None,
    log_interval=10,
    save_interval=5,
    use_wandb=True,
    use_tensorboard=True,
    early_stopping_patience=20,  # 早停的耐心值
    early_stopping_delta=0.0001,  # 最小改善阈值
    gradient_accumulation_steps=1,  # 梯度累积步数
):
    """
    训练AE模型并记录训练过程
    
    Args:
        model: AE模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮数
        device: 训练设备
        save_dir: 保存模型的目录
        log_dir: 日志目录
        exp_name: 实验名称，用于日志和保存模型
        log_interval: 记录日志的间隔（batch数）
        save_interval: 保存模型的间隔（epoch数）
        use_wandb: 是否使用wandb记录训练过程
        use_tensorboard: 是否使用TensorBoard记录训练过程
        early_stopping_patience: 早停的耐心值，连续多少个epoch验证损失没有改善则停止训练
        early_stopping_delta: 验证损失改善的最小值，小于此值不算作改善
        gradient_accumulation_steps: 梯度累积步数，每多少步更新一次参数
    """
    model.train()
    
    # 实验名称，如果未指定则使用时间戳
    if exp_name is None:
        exp_name = f"ae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建保存目录
    model_save_dir = os.path.join(save_dir, exp_name)
    ensure_dir(model_save_dir)
    
    # 设置日志记录
    if use_tensorboard:
        tb_log_dir = os.path.join(log_dir, 'tensorboard', exp_name)
        ensure_dir(tb_log_dir)
        writer = SummaryWriter(log_dir=tb_log_dir)
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project="SketchACAD", name=exp_name, config={
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size if hasattr(train_loader, 'batch_size') else 'unknown',
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "model": model.__class__.__name__,
            "scheduler": scheduler.__class__.__name__ if scheduler else "None",
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_delta": early_stopping_delta,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": train_loader.batch_size * gradient_accumulation_steps if hasattr(train_loader, 'batch_size') else 'unknown',
        })
    
    # 记录训练开始时间
    start_time = time.time()
    best_val_loss = float('inf')
    
    # 早停相关变量
    early_stopping_counter = 0
    early_stopped = False
    
    # 训练指标记录
    train_losses = []
    val_losses = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        total_loss = 0.0
        num_batches = 0
        
        # 创建进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # 重置优化器的梯度
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 获取输入数据
            vec_dict = {
                'cad_vec': batch['vec_dict']['cad_vec'].to(device),
                'flag_vec': batch['vec_dict']['flag_vec'].to(device),
                'index_vec': batch['vec_dict']['index_vec'].to(device)
            }
            
            mask_dict = {
                'attn_mask': batch['mask_dict']['attn_mask'].to(device) if batch['mask_dict']['attn_mask'] is not None else None,
                'key_padding_mask': batch['mask_dict']['key_padding_mask'].to(device)
            }
            
            # 前向传播
            output, metrics = model(vec_dict, mask_dict)
            
            # 计算重建损失
            recon_loss = compute_cad_reconstruction_loss(
                output, vec_dict['cad_vec'], mask_dict['key_padding_mask']
            )
            
            # 总损失就是重建损失，并按梯度累积步数缩放
            loss = recon_loss / gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 仅在累积了指定步数的梯度后更新参数
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            
            # 记录损失（使用未缩放的损失值）
            total_loss += recon_loss.item()
            num_batches += 1
            
            # 更新进度条显示当前批次损失
            pbar.set_postfix({
                'loss': recon_loss.item(),
                'acc_step': (batch_idx % gradient_accumulation_steps) + 1
            })
            
            # 记录训练过程指标
            if batch_idx % log_interval == 0:
                global_step = epoch * len(train_loader) + batch_idx
                
                if use_tensorboard:
                    writer.add_scalar('train/batch_loss', recon_loss.item(), global_step)
                    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'train/batch_loss': recon_loss.item(),
                        'train/learning_rate': optimizer.param_groups[0]['lr']
                    }, step=global_step)
        
        # 计算平均训练损失
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 保存当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                # 获取输入数据
                val_vec_dict = {
                    'cad_vec': val_batch['vec_dict']['cad_vec'].to(device),
                    'flag_vec': val_batch['vec_dict']['flag_vec'].to(device),
                    'index_vec': val_batch['vec_dict']['index_vec'].to(device)
                }
                
                val_mask_dict = {
                    'attn_mask': val_batch['mask_dict']['attn_mask'].to(device) if val_batch['mask_dict']['attn_mask'] is not None else None,
                    'key_padding_mask': val_batch['mask_dict']['key_padding_mask'].to(device)
                }
                
                # 前向传播
                val_output, val_metrics = model(val_vec_dict, val_mask_dict)
                
                # 计算重建损失
                val_recon = compute_cad_reconstruction_loss(
                    val_output, val_vec_dict['cad_vec'], val_mask_dict['key_padding_mask']
                )
                
                val_loss += val_recon.item()
                val_batch_count += 1
        
        # 计算验证集平均损失
        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        
        # 学习率调度器
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # 计算每个epoch的耗时
        epoch_time = time.time() - epoch_start_time
        
        # 记录每个epoch的指标
        if use_tensorboard:
            writer.add_scalar('train/epoch_loss', avg_train_loss, epoch)
            writer.add_scalar('val/epoch_loss', avg_val_loss, epoch)
            writer.add_scalar('train/learning_rate', current_lr, epoch)
            writer.add_scalar('time/epoch_seconds', epoch_time, epoch)
            writer.add_scalar('early_stopping/counter', early_stopping_counter, epoch)
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'train/epoch_loss': avg_train_loss,
                'val/epoch_loss': avg_val_loss,
                'train/learning_rate': current_lr,
                'time/epoch_seconds': epoch_time,
                'early_stopping/counter': early_stopping_counter,
            }, step=epoch)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss - early_stopping_delta:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(model_save_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_model_path)
            
            print(f"保存最佳模型，验证损失: {avg_val_loss:.6f}")
            # 重置早停计数器
            early_stopping_counter = 0
        else:
            # 验证损失没有改善，早停计数器加1
            early_stopping_counter += 1
            print(f"验证损失未改善，早停计数器: {early_stopping_counter}/{early_stopping_patience}")
            
            # 检查是否应该早停
            if early_stopping_counter >= early_stopping_patience:
                print(f"早停! 验证损失已经 {early_stopping_patience} 个epoch没有改善")
                early_stopped = True
                break
        
        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(model_save_dir, f"checkpoint_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.8f} | "
              f"Time: {epoch_time:.2f}s")
    
    # 训练结束，保存最终模型
    final_model_path = os.path.join(model_save_dir, f"final_model.pth")
    torch.save({
        'epoch': epoch,  # 使用实际训练的轮数
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'early_stopped': early_stopped,
    }, final_model_path)
    
    # 将所有张量转换为Python标量
    train_losses = [x if isinstance(x, (int, float)) else x.cpu().item() for x in train_losses]
    val_losses = [x if isinstance(x, (int, float)) else x.cpu().item() for x in val_losses]
    learning_rates = [x if isinstance(x, (int, float)) else x.cpu().item() for x in learning_rates]
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"\n训练完成，总耗时: {total_time/60:.2f} 分钟")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"模型已保存到: {model_save_dir}")
    
    # 返回训练历史
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
    }


def main():
    """主函数，处理命令行参数并启动训练"""
    parser = argparse.ArgumentParser(description="训练AE模型")
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径，包含vec子目录')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型')
    parser.add_argument('--embed_dim', type=int, default=256, help='嵌入维度')
    parser.add_argument('--latent_dim', type=int, default=256, help='潜在空间维度')
    parser.add_argument('--enc_layers', type=int, default=4, help='编码器层数')
    parser.add_argument('--dec_layers', type=int, default=4, help='解码器层数')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout概率')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称')
    parser.add_argument('--log_interval', type=int, default=10, help='日志记录间隔')
    parser.add_argument('--save_interval', type=int, default=50, help='模型保存间隔')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb')
    parser.add_argument('--use_tensorboard', action='store_true', help='是否使用tensorboard')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点文件')
    parser.add_argument('--early_stopping_patience', type=int, default=400, help='早停的耐心值')
    parser.add_argument('--early_stopping_delta', type=float, default=5e-9, help='最小改善阈值')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='梯度累积步数')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 确保保存目录存在
    ensure_dir(args.save_dir)
    ensure_dir(args.log_dir)
    
    # 加载训练和验证集文件列表
    train_list_path = os.path.join(args.data_dir, 'train.json')
    test_list_path = os.path.join(args.data_dir, 'test.json')
    
    with open(train_list_path, 'r') as f:
        train_files = json.load(f)['files']
    
    with open(test_list_path, 'r') as f:
        test_files = json.load(f)['files']
    
    print(f"训练集: {len(train_files)}个文件, 验证集: {len(test_files)}个文件")
    
    # 创建数据集
    train_dataset = CADSequenceDataset(args.data_dir, train_files)
    val_dataset = CADSequenceDataset(args.data_dir, test_files)
    
    print(f"已创建训练集和验证集")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"已创建数据加载器")
    
    # 获取CAD类别信息
    cad_class_info = CAD_CLASS_INFO
    
    # 创建AE模型
    model = AE(
        cad_class_info=cad_class_info,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        device=device
    )
    
    # 移动模型到设备
    model = model.to(device)
    
    # 从检查点恢复（如果提供）
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载检查点: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"从epoch {start_epoch}继续训练")
        else:
            print(f"未找到检查点: {args.resume}")
    
    # 打印模型参数总数
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {num_params:,}")
    print(f"模型参数总数 (百万): {num_params/1e6:.2f}M")
    
    # 设置优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )
    
    # 如果恢复训练，加载优化器状态
    if args.resume and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # 如果恢复训练，加载调度器状态
    if args.resume and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"开始训练，共{args.epochs}个轮次")
    
    # 训练模型
    history = train_ae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        exp_name=args.exp_name,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    print("训练完成")


if __name__ == "__main__":
    main()