import os, sys

sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

import torch.nn as nn
import torch
import timm
from models.layers.dit import DiT_models, TimestepEmbedder
from utility.macro import *
from models.vqvae import VQVAE
class MultiViewEncoder(nn.Module):
    def __init__(self, vit_model_name="vit_base_patch16_224", output_dim=256):
        super().__init__()
        # 加载预训练ViT
        self.vit = timm.create_model(vit_model_name, pretrained=True)
        # 移除分类头
        self.vit.head = nn.Identity()
        
        # 多视图融合
        self.fusion = nn.Sequential(
            nn.Linear(self.vit.embed_dim, output_dim*2),
            nn.LayerNorm(output_dim*2),
            nn.GELU(),
            nn.Linear(output_dim*2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 冻结ViT主干
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def forward(self, views):
        # views: [batch_size, num_views, 3, H, W]
        B, V, C, H, W = views.shape
        views_flat = views.reshape(B*V, C, H, W)
        
        # 提取每个视图特征
        features = self.vit(views_flat)  # [B*V, vit_dim]
        features = features.reshape(B, V, -1)  # [B, V, vit_dim]
        
        # 视图聚合 (简单平均池化)
        pooled = torch.mean(features, dim=1)  # [B, vit_dim]
        
        # 投影到目标维度
        embedding = self.fusion(pooled)  # [B, output_dim]
        
        return embedding
class SketchCADDiffusion(nn.Module):
    def __init__(self, vae_latent_dim=256, dit_model_name='DiT-CAD', sequence_length=128):
        super().__init__()
        
        # 加载DiT模型
        self.diffusion = DiT_models[dit_model_name](
            input_size=sequence_length, 
            in_channels=1,  # 序列中每个位置是一维向量
            num_classes=1000,  # 保持默认
            learn_sigma=False  # 我们直接预测原始信号
        )
        
        # 条件编码投影器
        self.condition_proj = nn.Sequential(
            nn.Linear(vae_latent_dim, self.diffusion.hidden_size),
            nn.LayerNorm(self.diffusion.hidden_size),
            nn.GELU(),
            nn.Linear(self.diffusion.hidden_size, self.diffusion.hidden_size)
        )
        
        # 时间步嵌入
        self.time_embed = TimestepEmbedder(self.diffusion.hidden_size)
        
        # 扩散参数
        self.beta_schedule = torch.linspace(0.0001, 0.02, 1000)
        self.alphas = 1.0 - self.beta_schedule
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def forward(self, x, t, condition):
        """训练时前向传播：直接预测原始信号z_0"""
        # 处理条件嵌入
        cond_emb = self.condition_proj(condition)
        # 时间步嵌入
        t_emb = self.time_embed(t)
        # 合并条件
        combined_cond = cond_emb + t_emb
        for block in self.diffusion.blocks:
            x = block(x, combined_cond)                      # (N, T, D)
        x = self.diffusion.final_layer(x, combined_cond)     # (N, T, patch_size ** 2 * out_channels)
        pred_z0 = x
        return pred_z0

    
    def diffusion_step(self, x_t, t, pred_z0, next_t):
        """执行一步扩散采样，基于预测的z_0进行"""
        alpha_t = self.alphas_cumprod[t]
        alpha_next = self.alphas_cumprod[next_t] if next_t < 1000 else torch.tensor(0.0)
        
        # 预测噪声 (反向计算)
        pred_noise = (x_t - torch.sqrt(alpha_t) * pred_z0) / torch.sqrt(1 - alpha_t)
        sigma_t = 0.0
        x_next = torch.sqrt(alpha_next) * pred_z0 + torch.sqrt(1 - alpha_next - sigma_t**2) * pred_noise + sigma_t * torch.randn_like(x_t)
        
        return x_next

class VQVAEMapper(nn.Module):
    def __init__(self, vqvae_model):
        super().__init__()
        self.vqvae = vqvae_model
        
        # 从VQVAE获取码本
        self.codebook = self.vqvae.vq_layer._embedding.weight  # [num_embeddings, embed_dim]
        self.num_embeddings = self.vqvae.vq_layer._num_embeddings
    
    def continuous_to_indices(self, latents):
        """将连续向量映射到最近的码本索引"""
        # latents: [B, seq_len, embed_dim]
        B, L, D = latents.shape
        latents_flat = latents.reshape(-1, D)  # [B*L, D]
        
        # 计算与码本的距离
        distances = torch.cdist(latents_flat, self.codebook)  # [B*L, num_embeddings]
        
        # 找到最近的码本索引
        indices = torch.argmin(distances, dim=1)  # [B*L]
        indices = indices.reshape(B, L)  # [B, L]
        
        return indices
    
class Sketch2CAD(nn.Module):
    def __init__(self, vqvae, latent_dim=256, dit_model_name='DiT-CAD'):
        super().__init__()
        
        # 加载预训练的VQVAE
        self.vqvae = vqvae
        for param in self.vqvae.parameters():
            param.requires_grad = False  # 冻结VQVAE
        
        # 多视图编码器
        self.view_encoder = MultiViewEncoder(output_dim=latent_dim)
        
        # 序列长度与VQVAE一致
        seq_length = MAX_CAD_SEQUENCE_LENGTH
        
        # 扩散模型
        self.diffusion = SketchCADDiffusion(
            vae_latent_dim=latent_dim,
            dit_model_name=dit_model_name,
            sequence_length=seq_length
        )
        
        # VQVAE映射器
        self.vq_mapper = VQVAEMapper(vqvae)
        
        # 扩散参数
        self.num_sampling_steps = 50  # 采样步数
    
    def forward(self, views, z_t, timesteps):
        """训练阶段前向传播"""
        # 编码视图
        view_embedding = self.view_encoder(views)
        
        # 扩散模型前向传播，预测原始信号z_0
        pred_z0 = self.diffusion(z_t, timesteps, view_embedding)
        
        return pred_z0
    
    @torch.no_grad()
    def generate(self, views, vec_dict, mask_dict, sampling_steps=None):
        """从视图生成CAD序列 - 使用直接预测z_0的方法"""
        if sampling_steps is None:
            sampling_steps = self.num_sampling_steps
        
        # 编码视图获取条件嵌入
        view_embedding = self.view_encoder(views)
    
        batch_size = views.shape[0]
        seq_len = MAX_CAD_SEQUENCE_LENGTH
        latent_dim = self.vqvae.embed_dim
        device = views.device
    
        # 从标准正态分布采样初始噪声
        x_t = torch.randn(batch_size, seq_len, latent_dim, device=device)
    
        # 从T到0逐步去噪
        time_steps = torch.linspace(999, 0, sampling_steps).long().to(device)
    
        for i in range(len(time_steps)-1):
            t = time_steps[i]
            next_t = time_steps[i+1]
        
            # 计算当前时间步的z_0预测
            t_batch = torch.full((batch_size,), t, device=device)
            pred_z0 = self.diffusion(x_t, t_batch, view_embedding)
        
            # 应用去噪步骤(基于预测的z_0)
            x_t = self.diffusion.diffusion_step(x_t, t, pred_z0, next_t)
    
        # 最后一步可以直接使用预测的z_0
        t_batch = torch.full((batch_size,), time_steps[-1], device=device)
        final_pred_z0 = self.diffusion(x_t, t_batch, view_embedding)
    
        # 映射到VQVAE码本索引
        indices = self.vq_mapper.continuous_to_indices(final_pred_z0)
    
        # 从索引解码为CAD序列
        output = self.vqvae.decode_from_indices(indices, vec_dict, mask_dict)
    
        return output, indices