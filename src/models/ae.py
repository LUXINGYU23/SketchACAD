import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
import sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))

from src.models.layers.attention import MultiHeadAttention
from src.models.layers.functional import FeedForwardLayer
from src.models.layers.embedder import CADSequenceEmbedder, PositionalEncodingSinCos,PositionalEncodingLUT
from src.models.utils import count_parameters
from src.CadSeqProc.utility.macro import *
from src.models.layers.transformer import *
from src.models.layers.improved_transformer import *
from src.CadSeqProc.utility.utils import (
    create_flag_vec,
    create_index_vec,
    top_p_sampling,
)
from typing import Dict, Optional, Tuple

class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        # src_key_padding_mask = src_key_padding_mask.transpose(0, 1)
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
                           )
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, tgt_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class ConstEmbedding(nn.Module):
    """学习的常量嵌入，为解码器提供初始输入"""
    def __init__(self, d_model, seq_len, device="cuda"):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.device = device
        self.pos_encoding = PositionalEncodingLUT(d_model, max_len=seq_len)
        # # 可学习的常量嵌入
        # self.learned_embed = nn.Parameter(torch.randn(seq_len, 1, d_model) * 0.02)

    def forward(self, z):
        batch_size = z.size(1)
        # # 扩展到批次大小
        # embed = self.learned_embed.expand(-1, batch_size, -1)
        # 添加位置编码
        src = self.pos_encoding(z.new_zeros(self.seq_len, batch_size, self.d_model))
        return src


class FCN(nn.Module):
    """输出投影层"""
    def __init__(self, d_model, cad_class_info):
        super().__init__()
        # 输出x和y的坐标
        self.output_x = nn.Linear(d_model, cad_class_info["one_hot_size"])
        self.output_y = nn.Linear(d_model, cad_class_info["one_hot_size"])

    def forward(self, out):
        S, N, _ = out.shape
        x_logits = self.output_x(out)  # [S, N, one_hot_size]
        y_logits = self.output_y(out)  # [S, N, one_hot_size]
        
        # 重新整形为目标格式
        return x_logits, y_logits


class Bottleneck(nn.Module):
    """将编码器输出投影到潜在空间"""
    def __init__(self, d_model, latent_dim):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.bottleneck(z)


# 自动编码器（AE）模型
class AE(nn.Module):
    def __init__(
        self,
        cad_class_info: Dict[str, int],
        embed_dim: int = 256,
        latent_dim: int = 256,
        enc_layers: int = 4,
        dec_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        
        # 保留原始的序列嵌入和位置编码
        self.seq_embed = CADSequenceEmbedder(
            one_hot_size=cad_class_info["one_hot_size"],
            flag_size=cad_class_info["flag_size"],
            index_size=cad_class_info["index_size"],
            d_model=embed_dim,
            device=device,
        )
        
        self.pe = PositionalEncodingSinCos(
            embedding_size=embed_dim, 
            max_seq_len=MAX_CAD_SEQUENCE_LENGTH, 
            device=device
        )
        
        # 使用改进的编码器
        encoder_layer = TransformerEncoderLayerImproved(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout
        )
        encoder_norm = LayerNorm(embed_dim)
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=enc_layers,
            norm=encoder_norm
        )
        
        # 瓶颈层
        self.bottleneck = Bottleneck(embed_dim, latent_dim)
        
        # 常量嵌入
        self.const_embedding = ConstEmbedding(
            d_model=embed_dim,
            seq_len=MAX_CAD_SEQUENCE_LENGTH,
            device=device
        )
        
        # 使用改进的解码器
        decoder_layer = TransformerDecoderLayerGlobalImproved(
            d_model=embed_dim,
            d_global=latent_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout
        )
        decoder_norm = LayerNorm(embed_dim)
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=dec_layers,
            norm=decoder_norm
        )
        
        # 输出投影
        self.fcn = FCN(embed_dim, cad_class_info)
    
    def _prepare_input_masks(self, mask_dict: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """准备编码器和解码器的掩码. 返回 (B,L) 形状的掩码或 None."""
        key_padding_mask = mask_dict.get("key_padding_mask")
        if key_padding_mask is not None:
            if key_padding_mask.dim() > 2:  # 例如 (B, L, 2)
                # 如果序列元素的两个分量都标记为填充，则认为该元素已填充
                key_padding_mask = torch.all(key_padding_mask, dim=-1) # 转换为 (B,L)
            return key_padding_mask
        return None
    
    def encode(self, vec_dict: Dict[str, torch.Tensor], mask_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """编码CAD序列到潜在空间"""
        # 序列嵌入 + 位置编码
        seq_len = vec_dict["cad_vec"].shape[1]
        # 处理掩码
        prepared_key_padding_mask = self._prepare_input_masks(mask_dict)
        x = self.pe(seq_len) + self.seq_embed(vec_dict, prepared_key_padding_mask)
        x = x.transpose(0, 1)
        # 编码
        memory = self.encoder(x, src_key_padding_mask=prepared_key_padding_mask)
        
        # 创建padding_mask用于池化
        if prepared_key_padding_mask is not None:
            # prepared_key_padding_mask 是 (B, L), True 表示已填充
            # 我们需要一个有效token的掩码, 即 (~prepared_key_padding_mask)
            # pool_mask 应该是 (L, B, 1) 以便与 memory (L, B, D) 进行逐元素乘法
            pool_mask = (~prepared_key_padding_mask).float().unsqueeze(-1) # (B, L, 1)
            pool_mask = pool_mask.transpose(0, 1) # (L, B, 1)
            
            # 池化 - 为分母添加 clamp 以防止除以零
            z = (memory * pool_mask).sum(dim=0, keepdim=True) / pool_mask.sum(dim=0, keepdim=True).clamp(min=1e-9)
        else:
            # 没有掩码时简单平均
            z = memory.mean(dim=0, keepdim=True)
        
        # 瓶颈投影
        latent = self.bottleneck(z)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """解码潜在表示为CAD序列"""
        # 获取常量嵌入作为解码器输入
        tgt = self.const_embedding(latent)
        
        # 解码
        out = self.decoder(tgt, latent)
        
        # 输出投影
        x_logits, y_logits = self.fcn(out)
        
        # 转回批次优先格式 [batch_size, seq_len, feature_dim]
        x_logits = x_logits.transpose(0, 1)
        y_logits = y_logits.transpose(0, 1)
        
        # 创建CAD向量格式输出
        output = torch.stack([x_logits, y_logits], dim=2)
        
        return output
    
    def forward(self, vec_dict: Dict[str, torch.Tensor], mask_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """前向传播过程"""
        # 编码
        latent = self.encode(vec_dict, mask_dict)
        
        # 解码
        output = self.decode(latent)
        
        # 指标
        metrics = {
            "latent_norm": torch.norm(latent, dim=2).mean().item(),
            "latent": latent.detach().transpose(0, 1),  # [batch_size, latent_dim]
        }
        
        return output, metrics
    
    def reconstruct(self, vec_dict: Dict[str, torch.Tensor], mask_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """重建CAD序列"""
        with torch.no_grad():
            # 编码 + 解码
            latent = self.encode(vec_dict, mask_dict)
            output = self.decode(latent)
            
            # 获取预测结果
            pred_x = torch.argmax(output[:, :, 0], dim=-1)
            pred_y = torch.argmax(output[:, :, 1], dim=-1)
            
            # 组合成CAD向量
            reconstructed_cad_vec = torch.stack([pred_x, pred_y], dim=-1)
            
            return {
                "cad_vec": reconstructed_cad_vec,
                "flag_vec": vec_dict["flag_vec"],
                "index_vec": vec_dict["index_vec"]
            }
    
    def sample(self, batch_size: int, seq_len: int, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """从随机潜在向量采样生成CAD序列"""
        with torch.no_grad():
            # 从随机分布采样潜在向量
            z = torch.randn(1, batch_size, self.latent_dim).to(self.device) * temperature
            
            # 解码
            output = self.decode(z)
            
            # 初始化序列(起始标记)
            new_cad_seq_dict = {
                "cad_vec": torch.tensor([[[1, 0]]]).repeat(batch_size, 1, 1).to(self.device),
                "flag_vec": torch.zeros(batch_size, 1, dtype=torch.int32).to(self.device),
                "index_vec": torch.zeros(batch_size, 1, dtype=torch.int32).to(self.device),
            }
            
            # 获取预测结果
            pred_x = torch.argmax(output[:, :, 0], dim=-1)
            pred_y = torch.argmax(output[:, :, 1], dim=-1)
            
            # 组合成CAD向量
            gen_cad_vec = torch.stack([pred_x, pred_y], dim=-1)
            
            # 截断到最大长度
            max_len = min(seq_len, gen_cad_vec.shape[1])
            gen_cad_vec = gen_cad_vec[:, :max_len]
            
            # 生成flag向量和index向量
            flag_vec = torch.zeros(batch_size, max_len, dtype=torch.int32).to(self.device)
            index_vec = torch.zeros(batch_size, max_len, dtype=torch.int32).to(self.device)
            
            for t in range(1, max_len):
                new_flag = create_flag_vec(gen_cad_vec[:, :t+1], flag_vec[:, :t])
                flag_vec[:, t] = new_flag[:, -1]
                
                new_index = create_index_vec(gen_cad_vec[:, :t+1], index_vec[:, :t])
                index_vec[:, t] = new_index[:, -1]
            
            return {
                "cad_vec": gen_cad_vec,
                "flag_vec": flag_vec,
                "index_vec": index_vec
            }
    
    def generate(self, latent=None, batch_size=1, seq_len=None, temperature=1.0):
        """从给定潜在向量生成CAD序列"""
        if seq_len is None:
            seq_len = MAX_CAD_SEQUENCE_LENGTH
        
        with torch.no_grad():
            if latent is None:
                latent = torch.randn(1, batch_size, self.latent_dim).to(self.device) * temperature
            elif latent.dim() == 1:
                latent = latent.unsqueeze(0).unsqueeze(0)
            elif latent.dim() == 2:
                latent = latent.unsqueeze(0)
                
            # 解码
            output = self.decode(latent)
            
            # 初始化序列(起始标记)
            new_cad_seq_dict = {
                "cad_vec": torch.tensor([[[1, 0]]]).repeat(batch_size, 1, 1).to(self.device),
                "flag_vec": torch.zeros(batch_size, 1, dtype=torch.int32).to(self.device),
                "index_vec": torch.zeros(batch_size, 1, dtype=torch.int32).to(self.device),
            }
            
            # 获取预测结果
            pred_x = torch.argmax(output[:, :, 0], dim=-1)
            pred_y = torch.argmax(output[:, :, 1], dim=-1)
            
            # 组合成CAD向量
            gen_cad_vec = torch.stack([pred_x, pred_y], dim=-1)
            
            # 截断到最大长度
            max_len = min(seq_len, gen_cad_vec.shape[1])
            gen_cad_vec = gen_cad_vec[:, :max_len]
            
            # 生成flag向量和index向量
            flag_vec = torch.zeros(batch_size, max_len, dtype=torch.int32).to(self.device)
            index_vec = torch.zeros(batch_size, max_len, dtype=torch.int32).to(self.device)
            
            for t in range(1, max_len):
                new_flag = create_flag_vec(gen_cad_vec[:, :t+1], flag_vec[:, :t])
                flag_vec[:, t] = new_flag[:, -1]
                
                new_index = create_index_vec(gen_cad_vec[:, :t+1], index_vec[:, :t])
                index_vec[:, t] = new_index[:, -1]
            
            return {
                "cad_vec": gen_cad_vec,
                "flag_vec": flag_vec,
                "index_vec": index_vec
            }
    
    def total_parameters(self, description: bool = False, in_millions: bool = False) -> int:
        num_params = count_parameters(self, description)
        if in_millions:
            num_params_million = num_params / 1_000_000
            print(f"参数总数: {num_params_million:.1f}M")
        return num_params