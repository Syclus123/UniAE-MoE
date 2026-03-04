"""
Adapter & Q-Former instruction-tuning strategy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# v1
class EncoderAdapter(nn.Module):
    """
    Encoder Adapter: 2 x MLP + GELU
    structure: Linear -> GELU -> Linear
    
    Args:
        input_dim: input dimension (Encoder output dimension)
        hidden_dim: hidden dimension, default: input_dim * 4
        output_dim: output dimension, default: input_dim
        dropout: dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim * 4
        if output_dim is None:
            output_dim = input_dim
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """initialize weights"""
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass
        
        Args:
            x: input tensor, shape (batch_size, seq_len, input_dim)
            
        Returns:
            output tensor, shape (batch_size, seq_len, output_dim)
        """
        return self.adapter(x)

# v2
class ResidualEncoderAdapter(nn.Module):
    """
    Residual Encoder Adapter
    
    structure: x + Adapter(x)
    
    Args:
        input_dim: input dimension
        hidden_dim: hidden dimension
        dropout: dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim * 4
            
        self.adapter = EncoderAdapter(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            dropout=dropout,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass (with residual connection)
        """
        return x + self.adapter(x)
    

# Utils
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN: (xW1) * silu(xW2) -> W3"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = F.silu(self.w2(x))
        out = x1 * x2
        out = self.drop(out)
        out = self.w3(out)
        return self.drop(out)


class DropPath(nn.Module):
    """Stochastic Depth (per sample)"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


# QFormer++ Layer
class QFormerPlusLayer(nn.Module):
    """
    Query self-attn + (split cross-attn to global/local) + SwiGLU FFN
    with gated cross-attn residual (stable)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.self_norm = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_norm_q = RMSNorm(d_model)
        self.cross_norm_x = RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # gated cross-attn residual: tanh(gate) * cross_attn
        self.cross_gate = nn.Parameter(torch.zeros(1))

        self.ffn_norm = RMSNorm(d_model)
        ffn_hidden = int(d_model * ffn_ratio)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout=dropout)

        self.drop_path = DropPath(drop_path)

    def _cross_attend(self, q: torch.Tensor, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        q: [B, K, D]
        x: [B, T, D]
        x_mask: [B, T] 1=valid, 0=pad
        """
        qn = self.cross_norm_q(q)
        xn = self.cross_norm_x(x)

        # key_padding_mask: True => ignore
        key_padding_mask = (x_mask == 0)
        out = self.cross_attn(
            query=qn,
            key=xn,
            value=xn,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return out

    def forward(
        self,
        q_global: torch.Tensor,
        q_local: torch.Tensor,
        x_full: torch.Tensor,
        x_full_mask: torch.Tensor,
        x_pool: torch.Tensor,
        x_pool_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q_global: [B, Kg, D]
        q_local : [B, Kl, D]
        x_full  : [B, T,  D]
        x_pool  : [B, Tp, D]
        """
        # self-attn on concatenated queries
        q = torch.cat([q_global, q_local], dim=1)  # [B, K, D]
        qn = self.self_norm(q)
        q = q + self.drop_path(self.self_attn(qn, qn, qn, need_weights=False)[0])

        # split back
        Kg = q_global.size(1)
        q_global = q[:, :Kg]
        q_local = q[:, Kg:]

        # cross-attn: global queries attend pooled audio, local attend full audio
        cross_global = self._cross_attend(q_global, x_pool, x_pool_mask)
        cross_local  = self._cross_attend(q_local,  x_full, x_full_mask)

        gate = torch.tanh(self.cross_gate)  # scalar gate
        q_global = q_global + self.drop_path(gate * cross_global)
        q_local  = q_local  + self.drop_path(gate * cross_local)

        # FFN on concatenated queries
        q = torch.cat([q_global, q_local], dim=1)
        q = q + self.drop_path(self.ffn(self.ffn_norm(q)))

        # split back
        q_global = q[:, :Kg]
        q_local  = q[:, Kg:]
        return q_global, q_local


# QFormer++ Resampler
class QFormerPlusResampler(nn.Module):
    """
    Strong Q-Former for mixed tasks (classification + ASR + caption)

    Input:
        audio_frames: [B, T, 2048] from Qwen3 audio encoder :contentReference[oaicite:2]{index=2}
        audio_mask  : [B, T] 1=valid, 0=pad
    Output:
        q_tokens: [B, K, d_model]
        q_mask  : [B, K]
    """
    def __init__(
        self,
        audio_dim: int = 2048,
        d_model: int = 1024,
        num_global_queries: int = 32,
        num_local_queries: int = 96,     # total K = 128 (good for ASR/caption)
        n_layers: int = 6,
        n_heads: int = 16,
        pool_stride: int = 4,            # multi-scale pooling
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.05,
        num_tasks: int = 12,             # downstream tasks
        task_cond: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.audio_dim = audio_dim
        self.d_model = d_model
        self.num_global_queries = num_global_queries
        self.num_local_queries = num_local_queries
        self.pool_stride = pool_stride
        self.task_cond = task_cond

        # project audio frames to d_model (saves compute vs keeping 2048)
        self.audio_proj = nn.Linear(audio_dim, d_model, bias=False)
        self.audio_norm = RMSNorm(d_model)

        # learnable queries
        self.global_queries = nn.Parameter(torch.randn(num_global_queries, d_model) * 0.02)
        self.local_queries  = nn.Parameter(torch.randn(num_local_queries,  d_model) * 0.02)

        # task-conditioned query bias (lightweight, but very effective for multi-task)
        if task_cond:
            self.task_embed = nn.Embedding(num_tasks, d_model)
            self.task_proj  = nn.Linear(d_model, d_model, bias=False)

        # layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dp = drop_path * (i / max(1, n_layers - 1))  # linearly increase
            self.layers.append(QFormerPlusLayer(
                d_model=d_model,
                n_heads=n_heads,
                ffn_ratio=ffn_ratio,
                dropout=dropout,
                drop_path=dp,
            ))

        self.out_norm = RMSNorm(d_model)

    def _pool(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        simple strided pooling: x[:, ::stride]
        """
        s = self.pool_stride
        x_pool = x[:, ::s, :]
        m_pool = mask[:, ::s]
        return x_pool, m_pool

    def forward(
        self,
        audio_frames: torch.Tensor,
        audio_mask: torch.Tensor,
        task_id: torch.Tensor | None = None,
        query_budget: str = "auto",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        task_id: [B] int64 in [0, num_tasks-1], optional
        query_budget:
            - "cls"  : fewer queries (faster, good for classification)
            - "gen"  : full queries (better for ASR/caption)
            - "auto" : if task_id provided you can map task types outside and pass "cls"/"gen"
        """
        B, T, _ = audio_frames.shape
        assert audio_mask.shape[:2] == (B, T)

        # audio proj + norm
        x = self.audio_proj(audio_frames)
        x = self.audio_norm(x)

        # pooled audio for global queries
        x_pool, m_pool = self._pool(x, audio_mask)

        # select query budget
        if query_budget == "cls":
            # keep less local queries for classification
            Kl = max(32, self.num_local_queries // 3)  # 32
        else:
            Kl = self.num_local_queries               # 96 for ASR/caption
        Kg = self.num_global_queries

        qg = self.global_queries[:Kg].unsqueeze(0).expand(B, -1, -1).contiguous()
        ql = self.local_queries[:Kl].unsqueeze(0).expand(B, -1, -1).contiguous()

        # task-conditioned bias
        if self.task_cond and (task_id is not None):
            te = self.task_proj(self.task_embed(task_id))  # [B, D]
            qg = qg + te.unsqueeze(1)
            ql = ql + te.unsqueeze(1)

        # QFormer++ layers
        for layer in self.layers:
            qg, ql = layer(qg, ql, x, audio_mask, x_pool, m_pool)

        q = torch.cat([qg, ql], dim=1)  # [B, K, D]
        q = self.out_norm(q)

        q_mask = torch.ones((B, q.shape[1]), device=q.device, dtype=audio_mask.dtype)
        return q, q_mask
