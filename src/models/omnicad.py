import os, sys
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import BertModel, BertConfig
import dgl
from src.CadSeqProc.utility.macro import *
from models.layers.dit import DiT_models, TimestepEmbedder
from models.vae import VAE
from src.models.uvnetencoders import UVNetCurveEncoder, UVNetSurfaceEncoder, UVNetGraphEncoder

class ModalityProjector(nn.Module):
    """将各种模态的特征投影到统一维度的嵌入空间"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.projection(x)

class BRepEncoder(nn.Module):
    """使用UVNet处理BRep数据"""
    def __init__(self, output_dim=256):
        super().__init__()
        self.crv_emb_dim = 64
        self.srf_emb_dim = 64
        self.graph_emb_dim = 128
        
        self.curv_encoder = UVNetCurveEncoder(
            in_channels=6, output_dims=self.crv_emb_dim
        )
        self.surf_encoder = UVNetSurfaceEncoder(
            in_channels=7, output_dims=self.srf_emb_dim
        )
        self.graph_encoder = UVNetGraphEncoder(
            self.srf_emb_dim, self.crv_emb_dim, self.graph_emb_dim
        )
        
        # 投影到目标维度
        self.projector = ModalityProjector(self.graph_emb_dim, output_dim)
    
    def forward(self, batched_graph):
        # 处理BRep输入
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        
        # 计算特征
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        
        # Message passing
        _, graph_emb = self.graph_encoder(
            batched_graph, hidden_srf_feat, hidden_crv_feat
        )
        
        # 投影到统一维度
        embedding = self.projector(graph_emb)
        return embedding

class PointCloudEncoder(nn.Module):
    """使用PointNet++处理点云数据"""
    def __init__(self, output_dim=256):
        super().__init__()
        # 导入PointNet++
        try:
            from pointnet2_ops.pointnet2_modules import PointnetSAModule
        except ImportError:
            raise ImportError("PointNet++ not installed. Please install pointnet2_ops.")
            
        # PointNet++ 参数
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=True
            )
        )
        
        # 冻结PointNet++参数
        for m in self.SA_modules.parameters():
            m.requires_grad = False
        
        # 投影到目标维度
        self.projector = ModalityProjector(1024, output_dim)
    
    def forward(self, pointcloud):
        """
        输入: B x N x 3 的点云
        输出: B x output_dim 的特征
        """
        xyz = pointcloud
        batch_size = xyz.shape[0]
        
        # PointNet++ 处理
        l_xyz, l_features = [xyz], [None]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        # 获取全局特征
        global_features = l_features[-1].view(batch_size, -1)
        
        # 投影到统一维度
        embedding = self.projector(global_features)
        return embedding

class MultiViewImageEncoder(nn.Module):
    """使用DINOv2处理多视图图像"""
    def __init__(self, output_dim=256, model_name="dinov2_vits14"):
        super().__init__()
        # 导入DINOv2
        try:
            import torch.hub
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        except:
            # 备选：使用timm中的DeiT作为替代
            self.backbone = timm.create_model("vit_deit_base_patch16_224", pretrained=True)
            self.backbone.head = nn.Identity()  # 移除分类头
        
        # 获取特征维度
        self.feature_dim = self.backbone.embed_dim
        
        # 冻结backbone参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 投影到目标维度
        self.projector = ModalityProjector(self.feature_dim, output_dim)
    
    def forward(self, images):
        """
        输入: B x V x 3 x H x W 的图像
        输出: B x output_dim 的特征
        """
        B, V, C, H, W = images.shape
        images_flat = images.reshape(B*V, C, H, W)
        
        # 提取每个视图的特征
        with torch.no_grad():
            features = self.backbone(images_flat)  # [B*V, feature_dim]
        
        # 重塑为[B, V, feature_dim]
        features = features.reshape(B, V, -1)
        
        # 平均池化
        pooled = torch.mean(features, dim=1)  # [B, feature_dim]
        
        # 投影到统一维度
        embedding = self.projector(pooled)
        return embedding

class TextEncoder(nn.Module):
    """使用BERT处理文本输入"""
    def __init__(self, output_dim=256, model_name="bert-base-uncased"):
        super().__init__()
        # 加载BERT
        self.bert_config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        
        # 冻结BERT参数
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # 投影到目标维度
        self.projector = ModalityProjector(self.bert.config.hidden_size, output_dim)
    
    def forward(self, input_ids, attention_mask=None):
        """
        输入: 文本的token ids和attention mask
        输出: B x output_dim 的特征
        """
        # 使用BERT提取特征
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # 使用[CLS]令牌的输出作为文本表示
            text_features = outputs.pooler_output  # [B, hidden_size]
        
        # 投影到统一维度
        embedding = self.projector(text_features)
        return embedding

class MultiModalController(nn.Module):
    """处理和整合多种模态输入"""
    def __init__(self, output_dim=256):
        super().__init__()
        
        # 初始化各模态编码器
        self.brep_encoder = BRepEncoder(output_dim)
        self.pc_encoder = PointCloudEncoder(output_dim)
        self.image_encoder = MultiViewImageEncoder(output_dim)
        self.text_encoder = TextEncoder(output_dim)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    
    def forward(self, modality_inputs):
        """
        输入: 包含各种模态输入的字典，可能包含:
            - 'brep': DGL图
            - 'pointcloud': 点云 [B, N, 3]
            - 'images': 多视图图像 [B, V, 3, H, W]
            - 'text': {'input_ids': tensor, 'attention_mask': tensor}
        
        输出: B x output_dim 的融合特征
        """
        embeddings = []
        modality_count = 0
        
        # 处理BRep输入
        if 'brep' in modality_inputs and modality_inputs['brep'] is not None:
            brep_emb = self.brep_encoder(modality_inputs['brep'])
            embeddings.append(brep_emb)
            modality_count += 1
        
        # 处理点云输入
        if 'pointcloud' in modality_inputs and modality_inputs['pointcloud'] is not None:
            pc_emb = self.pc_encoder(modality_inputs['pointcloud'])
            embeddings.append(pc_emb)
            modality_count += 1
        
        # 处理图像输入
        if 'images' in modality_inputs and modality_inputs['images'] is not None:
            img_emb = self.image_encoder(modality_inputs['images'])
            embeddings.append(img_emb)
            modality_count += 1
        
        # 处理文本输入
        if 'text' in modality_inputs and modality_inputs['text'] is not None:
            text_emb = self.text_encoder(**modality_inputs['text'])
            embeddings.append(text_emb)
            modality_count += 1
        
        # 如果没有输入，返回零向量
        if modality_count == 0:
            batch_size = 1  # 默认批次大小
            for v in modality_inputs.values():
                if v is not None:
                    if isinstance(v, dict) and 'input_ids' in v:
                        batch_size = v['input_ids'].shape[0]
                    elif hasattr(v, 'shape'):
                        batch_size = v.shape[0]
                    break
            return torch.zeros(batch_size, self.fusion[0].out_features).to(next(self.parameters()).device)
        
        # 特征融合策略：平均
        if modality_count > 1:
            fused_emb = torch.stack(embeddings, dim=0).mean(dim=0)
        else:
            fused_emb = embeddings[0]
        
        # 应用融合层
        output = self.fusion(fused_emb)
        return output

class OminiCADDiffusion(nn.Module):
    def __init__(self, latent_dim=256, dit_model_name='DiT-CAD', sequence_length=128):
        super().__init__()
        
        # 加载DiT模型
        self.diffusion = DiT_models[dit_model_name](
            input_size=sequence_length, 
            in_channels=latent_dim//sequence_length,  # 调整通道数量以匹配潜在空间维度
            num_classes=1000,  # 保持默认
            learn_sigma=False  # 我们直接预测原始信号
        )
        
        # 条件编码投影器
        self.condition_proj = nn.Sequential(
            nn.Linear(latent_dim, self.diffusion.hidden_size),
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
        
        # 潜在空间维度
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
    
    def forward(self, x, t, condition):
        """训练时前向传播：直接预测原始信号z_0"""
        # 调整x的形状以适应DiT (B, latent_dim) -> (B, seq_len, latent_dim/seq_len)
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.sequence_length, -1)
        
        # 处理条件嵌入
        cond_emb = self.condition_proj(condition)
        
        # 时间步嵌入
        t_emb = self.time_embed(t)
        
        # 合并条件
        combined_cond = cond_emb + t_emb
        
        # 通过DiT
        for block in self.diffusion.blocks:
            x_reshaped = block(x_reshaped, combined_cond)
        
        # 最终层
        pred_z0_reshaped = self.diffusion.final_layer(x_reshaped, combined_cond)
        
        # 重新调整形状为原始潜在空间维度
        pred_z0 = pred_z0_reshaped.reshape(batch_size, self.latent_dim)
        
        return pred_z0

    def diffusion_step(self, x_t, t, pred_z0, next_t):
        """执行一步扩散采样，基于预测的z_0进行"""
        alpha_t = self.alphas_cumprod[t]
        alpha_next = self.alphas_cumprod[next_t] if next_t < 1000 else torch.tensor(0.0).to(x_t.device)
        
        # 预测噪声 (反向计算)
        pred_noise = (x_t - torch.sqrt(alpha_t) * pred_z0) / torch.sqrt(1 - alpha_t)
        sigma_t = 0.0
        x_next = torch.sqrt(alpha_next) * pred_z0 + torch.sqrt(1 - alpha_next - sigma_t**2) * pred_noise + sigma_t * torch.randn_like(x_t)
        
        return x_next

class MultiModalOmniCAD(nn.Module):
    def __init__(self, vae, latent_dim=256, dit_model_name='DiT-CAD'):
        super().__init__()
        
        # 加载预训练的VAE
        self.vae = vae
        # 冻结VAE参数
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # 多模态控制器
        self.controller = MultiModalController(output_dim=latent_dim)
        
        # 序列长度与VAE一致
        seq_length = MAX_CAD_SEQUENCE_LENGTH
        
        # 扩散模型
        self.diffusion = OminiCADDiffusion(
            latent_dim=latent_dim,
            dit_model_name=dit_model_name,
            sequence_length=seq_length
        )
        
        # 扩散采样参数
        self.num_sampling_steps = 50
    
    def forward(self, modality_inputs, z_t, timesteps):
        """训练阶段前向传播"""
        # 编码多模态输入
        controller_embedding = self.controller(modality_inputs)
        
        # 扩散模型前向传播，预测原始信号z_0
        pred_z0 = self.diffusion(z_t, timesteps, controller_embedding)
        
        return pred_z0
    
    @torch.no_grad()
    def generate(self, modality_inputs, vec_dict, mask_dict, sampling_steps=None):
        """从多模态输入生成CAD序列"""
        if sampling_steps is None:
            sampling_steps = self.num_sampling_steps
        
        # 编码多模态输入获取条件嵌入
        controller_embedding = self.controller(modality_inputs)
        
        batch_size = controller_embedding.shape[0]
        device = controller_embedding.device
        
        # 从标准正态分布采样初始噪声
        x_t = torch.randn(batch_size, self.diffusion.latent_dim, device=device)
        
        # 从T到0逐步去噪
        time_steps = torch.linspace(999, 0, sampling_steps).long().to(device)
        
        for i in range(len(time_steps)-1):
            t = time_steps[i]
            next_t = time_steps[i+1]
            
            # 计算当前时间步的z_0预测
            t_batch = torch.full((batch_size,), t, device=device)
            pred_z0 = self.diffusion(x_t, t_batch, controller_embedding)
            
            # 应用去噪步骤(基于预测的z_0)
            x_t = self.diffusion.diffusion_step(x_t, t, pred_z0, next_t)
        
        # 最后一步可以直接使用预测的z_0
        t_batch = torch.full((batch_size,), time_steps[-1], device=device)
        final_pred_z0 = self.diffusion(x_t, t_batch, controller_embedding)
        
        # 通过VAE解码器生成CAD序列
        final_pred_z0_expanded = final_pred_z0.unsqueeze(0)  # [1, B, latent_dim] 适配VAE解码器
        output = self.vae.decode(final_pred_z0_expanded)
        
        # 处理输出为CAD序列格式
        pred_x = torch.argmax(output[:, :, 0], dim=-1)
        pred_y = torch.argmax(output[:, :, 1], dim=-1)
        cad_vec = torch.stack([pred_x, pred_y], dim=-1)
        
        # 创建输出字典
        result = {
            "cad_vec": cad_vec,
            "flag_vec": vec_dict["flag_vec"] if "flag_vec" in vec_dict else None,
            "index_vec": vec_dict["index_vec"] if "index_vec" in vec_dict else None,
        }
        
        return result, final_pred_z0