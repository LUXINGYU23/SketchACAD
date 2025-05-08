from loguru import logger
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class PositionalEncodingSinCos(nn.Module):
    def __init__(self, embedding_size: int, max_seq_len: int, device: str):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len

        # create a matrix of shape (max_seq_len, embedding_size/2)
        self.pos_enc = torch.zeros(max_seq_len, embedding_size)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size)
        )
        self.pos_enc[:, 0::2] = torch.sin(pos * div_term)
        self.pos_enc[:, 1::2] = torch.cos(pos * div_term)

        self.pos_enc = self.pos_enc.unsqueeze(0)

    def forward(self, seq_len):
        # x has shape (batch_size, seq_len, embedding_size)
        # add positional encoding to x
        x = self.pos_enc[:, :seq_len, :]
        return x.to(self.device)


class PositionalEncodingLUT(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncodingLUT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer("position", position)

        self.pos_embed = nn.Embedding(max_len, d_model)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[: x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class CADSequenceEmbedder(nn.Module):
    """
    CAD Sequence token embedding.

    parameters:

    one_hot_size: int. The dimension of one-hot embedding (default 4102).
    flag_size: Type of token
    index_size: Model index. Maximum is MAX_EXTRUSION
    d_model: Embedding dimension
    """

    def __init__(self, one_hot_size, flag_size, index_size, d_model, device="cpu"):
        super(CADSequenceEmbedder, self).__init__()

        # self.pe = PositionalEncodingSinCos(d_model,MAX_CAD_SEQUENCE_LENGTH, device=device)  # Positional encoding
        self.si = Embedder(index_size, d_model)  # CAD Sequence Index Encoding
        self.sf = Embedder(flag_size, d_model)  # CAD Sequence Flag Encoding
        self.cx = Embedder(one_hot_size, d_model)  # x-coordinate embedding
        self.cy = Embedder(one_hot_size, d_model)  # y-coordinate embedding
        self.device = torch.device(device)

    def forward(self, vec_dict, key_padding_mask):
        """
        vec_dict: contains key "cad_vec","flag_vec" and "index_vec"
        key_padding_mask: Tensor. Shape (batch_size, seq_len). True for padded elements.
        """
        # vec_dict["cad_vec"] shape: (batch_size, seq_len, 2)
        # key_padding_mask shape: (batch_size, seq_len), True for padded elements

        if key_padding_mask is not None:
            # Invert mask: True for non-padded elements, False for padded elements.
            # active_elements_mask will have shape (batch_size, seq_len)
            active_elements_mask = ~key_padding_mask

            # Apply the mask to x and y sequences.
            # Multiply by active_elements_mask.float() to zero out padded elements.
            # vec_dict["cad_vec"][:, :, 0] has shape (batch_size, seq_len)
            # active_elements_mask.float() has shape (batch_size, seq_len)
            x_seq = vec_dict["cad_vec"][:, :, 0] * active_elements_mask.float()
            y_seq = vec_dict["cad_vec"][:, :, 1] * active_elements_mask.float()
        else:
            x_seq = vec_dict["cad_vec"][:, :, 0]
            y_seq = vec_dict["cad_vec"][:, :, 1]

        # Embed x and y sequences
        # Ensure x_seq and y_seq are long type for embedding lookup
        x_embed = self.cx(x_seq.long())
        y_embed = self.cy(y_seq.long())
        flag_embed = self.sf(vec_dict["flag_vec"])
        index_embed = self.si(vec_dict["index_vec"])

        return flag_embed + index_embed + x_embed + y_embed


class VectorQuantizerEMA(nn.Module):
    """
    Code from SkexGen: Autoregressive Generation of CAD Construction Sequences with Disentangled Codebooks.
    https://github.com/samxuxiang/SkexGen
    """

    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        seqlen, bs = inputs.shape[0], inputs.shape[1]

        # Flatten input
        flat_input = inputs.reshape(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).reshape(
            seqlen, bs, self._embedding_dim
        )

        encodings_flat = encodings.reshape(inputs.shape[0], inputs.shape[1], -1)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), encodings_flat, encoding_indices
    
    def reset_codebook_with_kmeans(self, encoded_inputs, usage_threshold=0.01):
        """
        使用K-means重新初始化未使用或使用率低的码书向量
        
        Args:
            encoded_inputs: 编码器输出的嵌入向量 [seq_len, batch_size, embedding_dim]
            usage_threshold: 码书向量使用率阈值，低于此值的向量将被重新初始化
        """
        # 获取当前设备
        device = self._embedding.weight.device
        
        # 将输入展平为 [seq_len*batch_size, embedding_dim]
        flat_inputs = encoded_inputs.reshape(-1, self._embedding_dim).detach()
        
        # 计算当前码书使用率
        n = torch.sum(self._ema_cluster_size.data)
        usage = self._ema_cluster_size / n if n > 0 else torch.zeros_like(self._ema_cluster_size)
        
        # 找出未使用或低使用率的码书向量
        unused_indices = torch.where(usage < usage_threshold)[0].cpu().numpy()
        
        if len(unused_indices) > 0:
            # 有需要重置的码书向量
            print(f"重置 {len(unused_indices)}/{self._num_embeddings} 个低使用率码书向量")
            
            # 从当前批次中随机选择样本作为新码书向量
            num_samples = min(len(flat_inputs), len(unused_indices) * 10)
            if num_samples < len(unused_indices):
                return  # 样本不足，跳过本次重置
                
            # 随机选择样本
            perm = torch.randperm(len(flat_inputs))
            selected_inputs = flat_inputs[perm[:num_samples]]
            
            # 使用K-means聚类
            try:
                import numpy as np
                from sklearn.cluster import KMeans
                
                # 将张量转换为numpy进行聚类
                inputs_np = selected_inputs.cpu().numpy()
                kmeans = KMeans(n_clusters=len(unused_indices), n_init=10)
                kmeans.fit(inputs_np)
                
                # 获取聚类中心作为新的码书向量
                new_embeddings = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
                
                # 更新未使用的码书向量
                for i, idx in enumerate(unused_indices):
                    self._embedding.weight.data[idx] = new_embeddings[i]
                    # 重置EMA统计
                    self._ema_cluster_size[idx] = n / self._num_embeddings  # 初始化为平均使用率
                    self._ema_w.data[idx] = new_embeddings[i] * self._ema_cluster_size[idx]
                return True
            except ImportError:
                print("请安装numpy和sklearn以使用K-means聚类")
                return True
        return False

    def get_perplexity(self):
        n = torch.sum(self._ema_cluster_size.data)
        usage = self._ema_cluster_size / n if n > 0 else torch.zeros_like(self._ema_cluster_size)
        perplexity = torch.exp(-torch.sum(usage * torch.log(usage + 1e-10)))
        return perplexity.item() / self._num_embeddings

if __name__ == "__main__":
    pass
