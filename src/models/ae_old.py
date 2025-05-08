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
from src.models.layers.embedder import CADSequenceEmbedder, PositionalEncodingSinCos
from src.models.utils import count_parameters
from src.CadSeqProc.utility.macro import *
from src.CadSeqProc.utility.utils import (
    create_flag_vec,
    create_index_vec,
    top_p_sampling,
)
from typing import Dict, Optional, Tuple


# 常量嵌入层
class ConstEmbedding(nn.Module):
    """提供解码器的初始输入"""
    def __init__(self, seq_len: int, embed_dim: int, device: str = "cuda"):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.device = device
        self.pos_encoding = PositionalEncodingSinCos(embed_dim, max_seq_len=seq_len, device=device)
        self.learned_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

    def forward(self, batch_size: int):
        embeddings = self.learned_embed.expand(batch_size, -1, -1)
        embeddings = embeddings + self.pos_encoding(self.seq_len)
        return embeddings


# Transformer解码器层
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_latent, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.latent_proj = nn.Linear(d_latent, d_model)
        
        self.ff_net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, latent, tgt_mask=None, tgt_key_padding_mask=None):
        # 第一个子层: 自注意力
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask, 
                             key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        # 第二个子层: 潜在向量条件
        tgt2 = self.latent_proj(latent)
        tgt = tgt + self.dropout2(tgt2)
        
        # 第三个子层: 前馈网络
        tgt2 = self.norm2(tgt)
        tgt = tgt + self.dropout2(self.ff_net(tgt2))
        
        return tgt


# Transformer解码器
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) 
                                    for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt, latent, tgt_mask=None, tgt_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, latent, tgt_mask, tgt_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)
            
        return output


# CAD解码器
class CADDecoder(nn.Module):
    def __init__(
        self,
        cad_class_info: dict,
        d_model: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_seq_len: int = MAX_CAD_SEQUENCE_LENGTH,
        device: str = "cuda"
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device
        self.cad_class_info = cad_class_info
        
        # 常量嵌入层
        self.const_embedding = ConstEmbedding(max_seq_len, d_model, device)
        
        # 潜在向量投影
        self.latent_proj = nn.Linear(latent_dim, d_model) if latent_dim != d_model else None
                
        # 解码器层
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            d_latent=d_model,  
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # 解码器
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # 输出投影
        self.output_x = nn.Linear(d_model, cad_class_info["one_hot_size"])
        self.output_y = nn.Linear(d_model, cad_class_info["one_hot_size"])
        
    def forward(self, latent: torch.Tensor):
        batch_size = latent.shape[0]
            
        # 准备潜在向量
        if self.latent_proj is not None:
            latent = self.latent_proj(latent)
            
        # 确保潜在向量是 [1, batch_size, d_model] 维度
        if latent.dim() == 2:  # [batch_size, latent_dim]
            latent = latent.unsqueeze(0)
            
        # 获取常量嵌入并转为序列优先格式
        tgt = self.const_embedding(batch_size).transpose(0, 1)
        
        # 解码并恢复批次优先格式
        out = self.decoder(tgt, latent).transpose(0, 1)
        
        # 映射到输出空间
        x_logits = self.output_x(out)
        y_logits = self.output_y(out)
        
        # 合并X和Y的logits [batch_size, seq_len, 2, one_hot_size]
        output = torch.stack([x_logits, y_logits], dim=2)
        
        return output
    
    def decode(self, latent, maxlen, nucleus_prob=0.0, topk_index=1, device="cuda"):
        self.eval()
        batch_size = latent.shape[0]
        
        # 初始化序列(起始标记)
        new_cad_seq_dict = {
            "cad_vec": torch.tensor([[[1, 0]]]).repeat(batch_size, 1, 1).to(device),
            "flag_vec": torch.zeros(batch_size, 1, dtype=torch.int32).to(device),
            "index_vec": torch.zeros(batch_size, 1, dtype=torch.int32).to(device),
        }
        
        # 迭代生成序列
        for t in range(1, min(maxlen, self.max_seq_len)):
            # 前向传播预测下一个token
            with torch.no_grad():
                output = self(latent)
                cad_pred = output[:, t-1:t]
            
            # 采样获取新的标记
            if nucleus_prob == 0:  # top-k 采样
                pred_x = torch.argmax(cad_pred[:, 0, 0], dim=-1).unsqueeze(1)
                pred_y = torch.argmax(cad_pred[:, 0, 1], dim=-1).unsqueeze(1)
                new_token = torch.stack([pred_x, pred_y], dim=-1)
            else:  # 核采样
                pred_x = top_p_sampling(cad_pred[:, 0, 0], nucleus_prob)
                pred_y = top_p_sampling(cad_pred[:, 0, 1], nucleus_prob)
                new_token = torch.stack([pred_x, pred_y], dim=-1)
            
            # 添加新标记到序列
            new_cad_seq_dict["cad_vec"] = torch.cat(
                [new_cad_seq_dict["cad_vec"], new_token], dim=1
            )
            
            # 生成flag向量和index向量
            new_flag = create_flag_vec(new_cad_seq_dict["cad_vec"], new_cad_seq_dict["flag_vec"])
            new_cad_seq_dict["flag_vec"] = torch.cat([new_cad_seq_dict["flag_vec"], new_flag], dim=1)
            
            new_index = create_index_vec(new_cad_seq_dict["cad_vec"], new_cad_seq_dict["index_vec"])
            new_cad_seq_dict["index_vec"] = torch.cat([new_cad_seq_dict["index_vec"], new_index], dim=1)
            
            # 检查终止条件
            end_tokens = torch.logical_or(
                new_cad_seq_dict["cad_vec"][:, -1, 0] <= END_TOKEN.index("END_REVOLVE"),
                new_cad_seq_dict["flag_vec"][:, -1] > 0
            )
            if end_tokens.all():
                break
                
        return new_cad_seq_dict


# Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            input_dim=dim, embed_dim=dim, dropout=0, num_heads=num_heads
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff_layer = FeedForwardLayer(input_dim=dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # 自注意力
        x2 = self.norm1(x)
        attn_mask = mask.get("attn_mask") if mask else None
        key_padding_mask = mask.get("key_padding_mask") if mask else None
        
        x2, _ = self.self_attn(
            x2, x2, x2, 
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        
        x = x + self.dropout1(x2)
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff_layer(x2))
        
        return x


# Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# 瓶颈模块
class Bottleneck(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对序列维度进行全局池化后投影到潜在空间
        return self.fc(torch.mean(x, dim=1))


# 自动编码器（AE）模型
class AE(nn.Module):
    def __init__(
        self,
        cad_class_info: Dict[str, int],
        embed_dim: int = 256,
        latent_dim: int = 128,
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
        
        # 序列嵌入和位置编码
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
        
        # 编码器、瓶颈和解码器
        self.encoder = TransformerEncoder(
            dim=embed_dim,
            num_layers=enc_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.bottleneck = Bottleneck(embed_dim, latent_dim)
        
        self.decoder = CADDecoder(
            cad_class_info=cad_class_info,
            d_model=embed_dim,
            latent_dim=latent_dim,
            num_layers=dec_layers,
            num_heads=num_heads,
            dim_feedforward=embed_dim * 4,  
            dropout=dropout,
            max_seq_len=MAX_CAD_SEQUENCE_LENGTH,
            device=device
        )
    
    def encode(self, vec_dict: Dict[str, torch.Tensor], mask_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 序列嵌入 + 位置编码
        seq_len = vec_dict["cad_vec"].shape[1]
        x = self.pe(seq_len) + self.seq_embed(vec_dict, mask_dict["key_padding_mask"])
        
        # 处理掩码
        if "key_padding_mask" in mask_dict and mask_dict["key_padding_mask"] is not None:
            if mask_dict["key_padding_mask"].dim() > 2:
                mask_dict = mask_dict.copy()  
                mask_dict["key_padding_mask"] = torch.all(mask_dict["key_padding_mask"], dim=-1)
        
        # 编码器 + 瓶颈
        encoded = self.encoder(x, mask_dict)
        latent = self.bottleneck(encoded)
        
        return latent
    
    def forward(self, vec_dict: Dict[str, torch.Tensor], mask_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        # 编码获取潜在表示
        latent = self.encode(vec_dict, mask_dict)
        
        # 解码
        output = self.decoder(latent)
        
        metrics = {
            "latent_norm": torch.norm(latent, dim=1).mean().item(),
            "latent": latent.detach(),
        }
        
        return output, metrics
    
    def reconstruct(self, vec_dict: Dict[str, torch.Tensor], mask_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            # 编码 + 解码
            latent = self.encode(vec_dict, mask_dict)
            output = self.decoder(latent)
            
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
        with torch.no_grad():
            # 从随机分布采样潜在向量
            z = torch.randn(batch_size, self.latent_dim).to(self.device) * temperature
            
            # 解码生成
            return self.decoder.decode(
                latent=z,
                maxlen=seq_len,
                nucleus_prob=0.0,
                topk_index=1,
                device=self.device
            )
    
    def generate(self, latent=None, batch_size=1, seq_len=None, temperature=1.0):
        if seq_len is None:
            seq_len = MAX_CAD_SEQUENCE_LENGTH
        
        with torch.no_grad():
            if latent is None:
                latent = torch.randn(batch_size, self.latent_dim).to(self.device) * temperature
            elif latent.dim() == 1:
                latent = latent.unsqueeze(0)
                
            return self.decoder.decode(
                latent=latent,
                maxlen=seq_len,
                nucleus_prob=0.0,
                topk_index=1,
                device=self.device
            )
    
    def total_parameters(self, description: bool = False, in_millions: bool = False) -> int:
        num_params = count_parameters(self, description)
        if in_millions:
            num_params_million = num_params / 1_000_000
            print(f"参数总数: {num_params_million:.1f}M")
        return num_params