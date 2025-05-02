import os, sys

sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

import torch.nn as nn
import torch
import torch.nn.functional as F
import json
import argparse
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import io
from datetime import datetime
import wandb
from utility.macro import *
from utility.utils import ensure_dir



# 添加项目根目录到Python路径
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

# 导入自定义模块
from src.models.sketch2cad import Sketch2CAD
from src.models.vqvae import VQVAE
from src.train.train_sketch2cad import train_sketch2cad, plot_training_history
from utility.macro import *
from utility.utils import ensure_dir

def train_sketch2cad(
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
):
    """训练Sketch2CAD模型 - 使用直接预测z_0的损失，并记录可视化训练过程指标
    
    Args:
        model: Sketch2CAD模型
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
    """
    model.train()
    
    # 实验名称，如果未指定则使用时间戳
    if exp_name is None:
        exp_name = f"sketch2cad_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建保存目录
    model_save_dir = os.path.join(save_dir, exp_name)
    ensure_dir(model_save_dir)
    
    # 设置日志记录
    if use_tensorboard:
        tb_log_dir = os.path.join(log_dir, 'tensorboard', exp_name)
        ensure_dir(tb_log_dir)
        writer = SummaryWriter(log_dir=tb_log_dir)
    
    if use_wandb:
        wandb.init(project="SketchACAD", name=exp_name, config={
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size if hasattr(train_loader, 'batch_size') else 'unknown',
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "model": model.__class__.__name__,
            "scheduler": scheduler.__class__.__name__ if scheduler else "None"
        })
    
    # 记录训练开始时间
    start_time = time.time()
    best_val_loss = float('inf')
    
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
        
        for batch_idx, batch in enumerate(pbar):
            # 获取输入数据
            views = batch['views'].to(device)
            cad_data = batch['cad_data']
            
            vec_dict = {
                'cad_vec': cad_data['cad_vec'].to(device),
                'flag_vec': cad_data['flag_vec'].to(device),
                'index_vec': cad_data['index_vec'].to(device)
            }
            
            mask_dict = {
                'attn_mask': cad_data['attn_mask'].to(device) if 'attn_mask' in cad_data else None,
                'key_padding_mask': cad_data['key_padding_mask'].to(device)
            }
            
            # 使用VQVAE获取目标潜在向量 (z_0)
            with torch.no_grad():
                encoded = model.vqvae.encode(vec_dict, mask_dict)
                _, quantized, _, _ = model.vqvae.vq_layer(encoded)
                z_0 = quantized  # 这是我们要预测的原始信号
            
            # 随机选择时间步
            batch_size = views.shape[0]
            t = torch.randint(0, 1000, (batch_size,), device=device)
            
            # 加入噪声获取z_t
            noise = torch.randn_like(z_0)
            alphas_t = model.diffusion.alphas_cumprod[t].view(-1, 1, 1)
            z_t = torch.sqrt(alphas_t) * z_0 + torch.sqrt(1 - alphas_t) * noise
            
            # 预测原始信号z_0
            optimizer.zero_grad()
            pred_z0 = model(views, z_t, t)
            
            # 计算损失 (直接与原始信号比较)
            loss = F.mse_loss(pred_z0, z_0)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条显示当前批次损失
            pbar.set_postfix({'loss': loss.item()})
            
            # 记录训练过程指标
            if batch_idx % log_interval == 0:
                global_step = epoch * len(train_loader) + batch_idx
                
                if use_tensorboard:
                    writer.add_scalar('train/batch_loss', loss.item(), global_step)
                    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                
                if use_wandb:
                    wandb.log({
                        'train/batch_loss': loss.item(),
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
                val_views = val_batch['views'].to(device)
                val_cad_data = val_batch['cad_data']
                
                val_vec_dict = {
                    'cad_vec': val_cad_data['cad_vec'].to(device),
                    'flag_vec': val_cad_data['flag_vec'].to(device),
                    'index_vec': val_cad_data['index_vec'].to(device)
                }
                
                val_mask_dict = {
                    'attn_mask': val_cad_data['attn_mask'].to(device) if 'attn_mask' in val_cad_data else None,
                    'key_padding_mask': val_cad_data['key_padding_mask'].to(device)
                }
                
                # 获取目标潜在向量
                val_encoded = model.vqvae.encode(val_vec_dict, val_mask_dict)
                _, val_quantized, _, _ = model.vqvae.vq_layer(val_encoded)
                val_z_0 = val_quantized
                
                # 随机时间步
                val_batch_size = val_views.shape[0]
                val_t = torch.randint(0, 1000, (val_batch_size,), device=device)
                
                # 加噪声
                val_noise = torch.randn_like(val_z_0)
                val_alphas_t = model.diffusion.alphas_cumprod[val_t].view(-1, 1, 1)
                val_z_t = torch.sqrt(val_alphas_t) * val_z_0 + torch.sqrt(1 - val_alphas_t) * val_noise
                
                # 预测
                val_pred_z0 = model(val_views, val_z_t, val_t)
                val_batch_loss = F.mse_loss(val_pred_z0, val_z_0)
                
                val_loss += val_batch_loss.item()
                val_batch_count += 1
        
        # 计算验证集平均损失
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # 学习率调整
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
        
        if use_wandb:
            wandb.log({
                'train/epoch_loss': avg_train_loss,
                'val/epoch_loss': avg_val_loss,
                'train/learning_rate': current_lr,
                'time/epoch_seconds': epoch_time,
            }, step=epoch)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
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
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.8f} | Time: {epoch_time:.2f}s")
    
    # 训练结束，保存最终模型
    final_model_path = os.path.join(model_save_dir, f"final_model.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, final_model_path)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    
    # 训练和验证损失
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 学习率
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), learning_rates)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    
    # 保存图表
    plt.savefig(os.path.join(model_save_dir, 'training_plots.png'))
    
    # 保存到tensorboard
    if use_tensorboard:
        figure_buf = io.BytesIO()
        plt.savefig(figure_buf, format='png')
        figure_buf.seek(0)
        img = Image.open(figure_buf)
        img_tensor = to_tensor(img)
        writer.add_image('training/loss_plots', img_tensor)
        writer.close()
    
    # 保存到wandb
    if use_wandb:
        wandb.log({"training/loss_plots": wandb.Image(plt)})
        wandb.finish()
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to {model_save_dir}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
    }


def plot_training_history(history, save_path=None):
    """绘制训练历史指标"""
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制学习率曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['learning_rates'])
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    
    # 记录最佳验证损失
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, f"Best Validation Loss: {history['best_val_loss']:.6f}\n"
                       f"Total Training Time: {history['total_time']/60:.2f} min",
             horizontalalignment='center', verticalalignment='center',
             fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()
# 数据集类
class SketchCADDataset(Dataset):
    """Sketch2CAD数据集类"""
    def __init__(self, data_dir, file_list, view_transform=None):
        """
        初始化数据集
        
        Args:
            data_dir: 数据根目录
            file_list: 文件ID列表
            view_transform: 视图变换
        """
        self.data_dir = Path(data_dir)
        self.file_list = file_list
        self.view_transform = view_transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        
        # 加载CAD序列向量
        cad_data_path = self.data_dir / "vec" / f"{file_id}.pth"
        cad_data = torch.load(cad_data_path, map_location='cpu')
        
        # 加载视图图像
        views_path = self.data_dir / "views" / f"{file_id}.pth"
        
        try:
            # 尝试加载为torch张量
            views = torch.load(views_path, map_location='cpu')
        except:
            # 如果不成功，则尝试加载为图像
            views_folder = self.data_dir / "views" / file_id
            view_files = sorted([f for f in os.listdir(views_folder) if f.endswith(('.jpg', '.png'))])
            
            if not view_files:
                raise FileNotFoundError(f"未找到视图文件: {views_folder}")
            
            views = []
            for view_file in view_files:
                img_path = views_folder / view_file
                img = Image.open(img_path).convert('RGB')
                img = to_tensor(img)  # 转换为[0, 1]范围的张量
                if self.view_transform:
                    img = self.view_transform(img)
                views.append(img)
            
            # 将视图列表转换为张量 [num_views, C, H, W]
            views = torch.stack(views)
        
        # 如果加载成功且需要应用变换
        if self.view_transform and len(views.shape) == 4:  # [num_views, C, H, W]
            transformed_views = []
            for i in range(views.shape[0]):
                transformed_views.append(self.view_transform(views[i]))
            views = torch.stack(transformed_views)
        
        # 构建返回字典
        return {
            "views": views,
            "cad_data": {
                "cad_vec": cad_data["vec"]["cad_vec"],
                "flag_vec": cad_data["vec"]["flag_vec"],
                "index_vec": cad_data["vec"]["index_vec"],
                "attn_mask": cad_data["mask_cad_dict"].get("attn_mask"),
                "key_padding_mask": cad_data["mask_cad_dict"]["key_padding_mask"]
            }
        }
# 主函数
def main():
    """主函数，处理命令行参数并启动训练"""
    parser = argparse.ArgumentParser(description="训练Sketch2CAD模型")
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型')
    parser.add_argument('--vqvae_path', type=str, required=True, help='预训练VQVAE模型路径')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称')
    parser.add_argument('--log_interval', type=int, default=10, help='日志记录间隔')
    parser.add_argument('--save_interval', type=int, default=5, help='模型保存间隔')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb')
    parser.add_argument('--use_tensorboard', action='store_true', help='是否使用tensorboard')
    parser.add_argument('--embed_dim', type=int, default=256, help='嵌入维度')
    parser.add_argument('--num_embeddings', type=int, default=1024, help='码本大小')
    parser.add_argument('--dit_model', type=str, default='DiT-B/2', help='使用的DiT模型')
    
    args = parser.parse_args()
    
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
    
    # 定义视图变换
    view_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = SketchCADDataset(args.data_dir, train_files, view_transform)
    val_dataset = SketchCADDataset(args.data_dir, test_files, view_transform)
    
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
    
    # 创建VQVAE模型
    vqvae_model = VQVAE(
        cad_class_info=cad_class_info,
        embed_dim=args.embed_dim,
        num_embeddings=args.num_embeddings,
        enc_layers=4,
        dec_layers=4,
        ca_level_start=0,
        num_heads=8,
        commitment_cost=0.25,
        decay=0.99,
        dropout=0.1,
        device=device
    )
    
    # 加载预训练权重
    print(f"加载预训练VQVAE模型: {args.vqvae_path}")
    vqvae_checkpoint = torch.load(args.vqvae_path, map_location=device)
    if 'model_state_dict' in vqvae_checkpoint:
        vqvae_model.load_state_dict(vqvae_checkpoint['model_state_dict'])
    else:
        vqvae_model.load_state_dict(vqvae_checkpoint)
    
    print("已加载预训练VQVAE模型")
    
    # 创建Sketch2CAD模型
    model = Sketch2CAD(
        vqvae=vqvae_model,
        latent_dim=args.embed_dim,
        dit_model_name=args.dit_model
    )
    
    # 移动模型到设备
    model = model.to(device)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 设置优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs
    )
    
    print(f"开始训练，共{args.epochs}个轮次")
    
    # 训练模型
    history = train_sketch2cad(
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
        use_tensorboard=args.use_tensorboard
    )
    
    print("训练完成")
    
    # 绘制训练历史
    if args.exp_name:
        plot_path = os.path.join(args.save_dir, args.exp_name, "training_history.png")
    else:
        plot_path = os.path.join(args.save_dir, "training_history.png")
    
    plot_training_history(history, save_path=plot_path)
    print(f"训练历史已保存到: {plot_path}")

if __name__ == "__main__":
    main()