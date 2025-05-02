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
from src.models.layers.embedder import VectorQuantizerEMA, CADSequenceEmbedder, PositionalEncodingSinCos
from src.models.decoder import CADDecoder, CADDecoderLayer
from src.models.utils import count_parameters
from utility.macro import *
from typing import Dict, Optional, Tuple


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        dropout: float = 0.1
    ):
        super(TransformerEncoderLayer, self).__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(
            input_dim=dim,
            embed_dim=dim,
            dropout=0,
            num_heads=num_heads,
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 前馈网络
        self.ff_layer = FeedForwardLayer(input_dim=dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        x: 形状为 [batch_size, seq_len, dim] 的张量
        mask: 包含 "attn_mask" 和 "key_padding_mask" 的字典
        """
        # 应用LayerNorm
        x2 = self.norm1(x)
        
        # 自注意力
        attn_mask = mask.get("attn_mask") if mask else None
        key_padding_mask = mask.get("key_padding_mask") if mask else None
        
        x2, _ = self.self_attn(
            x2, x2, x2, 
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        
        # 残差连接和dropout
        x = x + self.dropout1(x2)
        x2 = self.norm2(x)
        
        # 前馈网络
        x = x + self.dropout2(self.ff_layer(x2))
        
        return x


class TransformerEncoder(nn.Module):
    """
    由多个TransformerEncoderLayer组成的编码器
    """
    def __init__(
        self, 
        dim: int, 
        num_layers: int, 
        num_heads: int, 
        dropout: float = 0.1
    ):
        super(TransformerEncoder, self).__init__()
        
        # 编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层归一化
        self.norm = nn.LayerNorm(dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        x: 形状为 [batch_size, seq_len, dim] 的张量
        mask: 包含 "attn_mask" 和 "key_padding_mask" 的字典
        """
        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, mask)
            
        # 最后的归一化
        return self.norm(x)


class VQVAE(nn.Module):
    """
    基于Transformer的VQVAE模型，用于CAD序列的向量量化

    Args:
        cad_class_info: CAD类别信息字典，包含one_hot_size、flag_size和index_size
        embed_dim: 编码器和解码器的嵌入维度
        num_embeddings: 码本大小（离散潜在表示的数量）
        enc_layers: 编码器层数
        dec_layers: 解码器层数
        ca_level_start: 解码器中交叉注意力开始的层数
        num_heads: 注意力头数
        commitment_cost: 承诺损失权重
        decay: EMA衰减率
        dropout: Dropout概率
    """
    
    def __init__(
        self,
        cad_class_info: Dict[str, int],
        embed_dim: int = 256,
        num_embeddings: int = 1024,
        enc_layers: int = 4,
        dec_layers: int = 4,
        ca_level_start: int = 0,
        num_heads: int = 8,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(VQVAE, self).__init__()
        
        self.embed_dim = embed_dim
        self.device = torch.device(device)
        
        # CAD序列嵌入
        self.seq_embed = CADSequenceEmbedder(
            one_hot_size=cad_class_info["one_hot_size"],
            flag_size=cad_class_info["flag_size"],
            index_size=cad_class_info["index_size"],
            d_model=embed_dim,
            device=device,
        )
        
        # 位置编码
        self.pe = PositionalEncodingSinCos(
            embedding_size=embed_dim, 
            max_seq_len=MAX_CAD_SEQUENCE_LENGTH, 
            device=device
        )
        
        # 编码器
        self.encoder = TransformerEncoder(
            dim=embed_dim,
            num_layers=enc_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 向量量化层
        self.vq_layer = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
            commitment_cost=commitment_cost,
            decay=decay
        )
        
        # 解码器
        self.decoder = CADDecoder(
            cad_class_info=cad_class_info,
            tdim=embed_dim,  # 编码器输出维度
            cdim=embed_dim,  # 解码器输入维度
            num_layers=dec_layers,
            num_heads=num_heads,
            dropout=dropout,
            ca_level_start=ca_level_start,
            device=device,
        )
    
    def encode(
        self, 
        vec_dict: Dict[str, torch.Tensor], 
        mask_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        编码CAD序列到连续潜在空间

        Args:
            vec_dict: 包含cad_vec、flag_vec和index_vec的字典
            mask_dict: 包含attn_mask和key_padding_mask的字典

        Returns:
            encoded: 编码后的表示
        """
        # 序列长度
        seq_len = vec_dict["cad_vec"].shape[1]
        
        # CAD序列嵌入 + 位置编码
        x = self.pe(seq_len) + self.seq_embed(vec_dict, mask_dict["key_padding_mask"])
        
        # 编码器
        encoded = self.encoder(x, mask_dict)
        
        return encoded
    
    def forward(
        self, 
        vec_dict: Dict[str, torch.Tensor], 
        mask_dict: Dict[str, torch.Tensor],
        metadata: bool = False
    ) -> Tuple[torch.Tensor, float, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            vec_dict: 包含cad_vec、flag_vec和index_vec的字典
            mask_dict: 包含attn_mask和key_padding_mask的字典
            metadata: 是否返回元数据

        Returns:
            output: 解码后的CAD序列
            vq_loss: 向量量化损失
            perplexity: 困惑度，衡量码本的使用程度
            metrics: 包含编码索引和其他指标的字典
        """
        # 编码
        encoded = self.encode(vec_dict, mask_dict)
        
        # 向量量化
        vq_loss, quantized, perplexity, encoding_indices = self.vq_layer(encoded)
        
        # 解码
        output, decoder_metadata = self.decoder(
            vec_dict,
            quantized,  # 使用量化后的表示作为条件
            mask_dict,
            metadata=metadata
        )
        
        # 汇总指标
        metrics = {
            "encoding_indices": encoding_indices,
            "perplexity": perplexity
        }
        
        if metadata and decoder_metadata:
            metrics.update(decoder_metadata)
        
        return output, vq_loss, perplexity, metrics
    
    def reconstruct(
        self, 
        vec_dict: Dict[str, torch.Tensor], 
        mask_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        重建输入CAD序列

        Args:
            vec_dict: 包含cad_vec、flag_vec和index_vec的字典
            mask_dict: 包含attn_mask和key_padding_mask的字典

        Returns:
            reconstructed_dict: 重建后的CAD序列字典
        """
        with torch.no_grad():
            # 前向传播
            output, _, _, _ = self.forward(vec_dict, mask_dict)
            
            # 获取预测结果
            pred_x = torch.argmax(output[:, :, 0], dim=-1)
            pred_y = torch.argmax(output[:, :, 1], dim=-1)
            
            # 组合成CAD向量
            reconstructed_cad_vec = torch.stack([pred_x, pred_y], dim=-1)
            
            # 返回重建结果
            reconstructed_dict = {
                "cad_vec": reconstructed_cad_vec,
                "flag_vec": vec_dict["flag_vec"],  # 保留原始flag_vec
                "index_vec": vec_dict["index_vec"],  # 保留原始index_vec
            }
            
            return reconstructed_dict
    
    def encode_to_indices(
        self, 
        vec_dict: Dict[str, torch.Tensor], 
        mask_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        将CAD序列编码为离散索引

        Args:
            vec_dict: 包含cad_vec、flag_vec和index_vec的字典
            mask_dict: 包含attn_mask和key_padding_mask的字典

        Returns:
            encoding_indices: 编码索引
        """
        with torch.no_grad():
            # 编码
            encoded = self.encode(vec_dict, mask_dict)
            
            # 向量量化并获取索引
            _, _, _, encoding_indices = self.vq_layer(encoded)
            
            return encoding_indices
    
    def decode_from_indices(
        self, 
        indices: torch.Tensor, 
        vec_dict: Dict[str, torch.Tensor], 
        mask_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        从离散索引解码CAD序列

        Args:
            indices: 编码索引
            vec_dict: 包含cad_vec、flag_vec和index_vec的原始字典（用于辅助解码）
            mask_dict: 包含attn_mask和key_padding_mask的字典

        Returns:
            output: 解码后的CAD序列
        """
        with torch.no_grad():
            # 从码本中查找嵌入
            quantized = self.vq_layer._embedding(indices)
            
            # 重塑为正确的形状
            batch_size = vec_dict["cad_vec"].shape[0]
            seq_len = vec_dict["cad_vec"].shape[1]
            quantized = quantized.reshape(batch_size, seq_len, self.embed_dim)
            
            # 解码
            output, _ = self.decoder(vec_dict, quantized, mask_dict)
            
            return output
    
    def autoregressively_generate(
        self,
        encoded_latent: torch.Tensor,  
        cross_attn_mask_dict: Dict[str, torch.Tensor],
        maxlen: int,
        nucleus_prob: float,
        topk_index: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        从潜在表示自回归生成CAD序列

        Args:
            encoded_latent: 编码后的潜在表示
            cross_attn_mask_dict: 交叉注意力掩码
            maxlen: 生成的最大序列长度
            nucleus_prob: 核采样概率
            topk_index: top-k采样索引
            device: 设备

        Returns:
            cad_seq_dict: 生成的CAD序列字典
        """
        # 使用解码器的自回归生成方法
        return self.decoder.decode(
            encoded_latent,
            cross_attn_mask_dict,
            maxlen,
            nucleus_prob,
            topk_index,
            device
        )
    
    def total_parameters(self, description: bool = False, in_millions: bool = False) -> int:
        """计算模型参数总数"""
        num_params = count_parameters(self, description)
        if in_millions:
            num_params_million = num_params / 1_000_000
            print(f"参数总数: {num_params_million:.1f}M")
        return num_params