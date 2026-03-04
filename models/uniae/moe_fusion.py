"""
UniAE-MoE 
MoE-based audio encoder fusing Audio Flamingo 3 and Qwen2-Audio

mode:
- "moe": standard MoE fusion – uses MLP experts with SiLU
- "moe_3layer": multi-layer MoE fusion – fuses 8/16/32 layer
- "moe_swiglu": improved MoE fusion – uses SwiGLU Expert (SOTA)

usage:
- eval:
    CUDA_VISIBLE_DEVICES=0 python -m xares_llm.run models/uniae/moe_fusion.py all all \
        --model_args '{
        "fusion_mode": "moe_swiglu",
        "checkpoint_path": "/path/checkpoints"
    }'
    
- train:
    - Single-GPU:
        CUDA_VISIBLE_DEVICES=0 python -m xares_llm.run \
            models/uniae/moe_fusion.py --stage full \
            --model_args '{"fusion_mode": "moe_swiglu"}' \
            --config config/train.yaml
    - Multi-GPU:
        CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src accelerate launch \
            --mixed_precision="bf16" \
            --num_processes=4 \
            --main_process_port="29501" \
            -m xares_llm.run_encoder_tuning \
            models/uniae/moe_fusion.py \
            --stage full \
            --model_args '{"fusion_mode": "moe_swiglu"}' \
            --config config/train.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file, save_file
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import snapshot_download

from transformers import (
    AudioFlamingo3ForConditionalGeneration, 
    AudioFlamingo3Processor,
    Qwen2AudioForConditionalGeneration,
    Qwen2AudioProcessor,
)


def length_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """ attention mask """
    if max_len is None:
        max_len = lengths.amax()
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = idx < lengths.unsqueeze(1)
    return mask.long()


# ========== Normalization ==========
class RMSNorm(nn.Module):
    """RMSNorm for stable training"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


# ========== Multi-Layer Encoder Wrappers ==========
def _clear_meta_device_context():
    original_device = torch.get_default_device()
    if original_device is not None and original_device.type == "meta":
        torch.set_default_device(None)


class AudioFlamingo3MultiLayerEncoders(nn.Module):
    """
    Audio Flamingo 3 Encoder (Support intermediate feature outputs)
    """
    def __init__(
        self, 
        model_name: str = "./audio-flamingo-3-hf",
        layer_indices: list[int] | None = None,
    ):
        super().__init__()
        
        if layer_indices is None:
            layer_indices = [8, 16, 32]
        self.layer_indices = layer_indices
        
        _clear_meta_device_context()
        
        print(f"[AudioFlamingo3] Loading model from {model_name}...")
        full_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float
        )
        
        # Audio Encoder
        self.encoder = full_model.audio_tower
        self.output_dim = self.encoder.config.d_model  # 1280
        self.num_layers = self.encoder.config.num_hidden_layers  # 32
        
        print(f"[AudioFlamingo3] num_layers={self.num_layers}, hidden_size={self.output_dim}")
        print(f"[AudioFlamingo3] Will extract layers: {layer_indices}")
        
        del full_model.language_model
        del full_model.multi_modal_projector
        del full_model
        
        self.processor = AudioFlamingo3Processor.from_pretrained(model_name, trust_remote_code=True)
        self.SAMPLE_RATE = 16000
        self.CHUNK_LENGTH = 30 * 16000
        
    def forward(
        self,
        audio: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Return:
            dict: {
                "last_hidden_state": [B, T, D],
                "hidden_states": tuple of [B, T, D] for each requested layer,
                "attention_mask": [B, T]
            }
        """
        device = audio.device
        batch_size = audio.size(0)
        audio = audio.to(self.encoder.dtype)
        
        # chunk
        all_chunks_list = []
        sample_to_chunk_indices = [[] for _ in range(batch_size)]
        
        audio_cpu = audio.detach().cpu()
        mask_cpu = audio_attention_mask.detach().cpu() if audio_attention_mask is not None else torch.ones_like(audio_cpu)
        
        global_chunk_idx = 0
        for i in range(batch_size):
            valid_length = mask_cpu[i].sum().int().item()
            valid_wav = audio_cpu[i, :valid_length].float().numpy()
            
            if len(valid_wav) == 0:
                chunks = [valid_wav]
            else:
                chunks = [
                    valid_wav[offset : offset + self.CHUNK_LENGTH] 
                    for offset in range(0, len(valid_wav), self.CHUNK_LENGTH)
                ]
            
            for chunk in chunks:
                all_chunks_list.append(chunk)
                sample_to_chunk_indices[i].append(global_chunk_idx)
                global_chunk_idx += 1
        
        # features
        inputs = self.processor.feature_extractor(
            all_chunks_list, 
            sampling_rate=self.SAMPLE_RATE,
            padding="max_length", 
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        input_features = inputs.input_features.to(device).to(self.encoder.dtype)
        input_features_mask = inputs.attention_mask.to(device)
        
        # Encoder Forward (hidden_states)
        encoder_outputs = self.encoder(
            input_features=input_features,
            input_features_mask=input_features_mask,
            output_hidden_states=output_hidden_states,
        )
        
        last_hidden_state = encoder_outputs.last_hidden_state
        all_hidden_states = encoder_outputs.hidden_states if output_hidden_states else None
        
        conv_output_lengths = (input_features_mask.sum(dim=-1) - 1) // 2 + 1
        post_lengths = (conv_output_lengths - 2) // 2 + 1
        max_len = last_hidden_state.shape[1]
        post_lengths = torch.clamp(post_lengths, min=0, max=max_len).long()
        
        def process_layer_output(hidden_state):
            """Merge chunk outputs back to batch outputs"""
            merged_list = []
            for i in range(batch_size):
                chunk_indices = sample_to_chunk_indices[i]
                sample_parts = []
                for idx in chunk_indices:
                    raw_embed = hidden_state[idx]
                    valid_len = post_lengths[idx]
                    valid_embed = raw_embed[:valid_len]
                    sample_parts.append(valid_embed)
                
                if len(sample_parts) > 0:
                    merged_sample = torch.cat(sample_parts, dim=0)
                else:
                    merged_sample = torch.zeros((1, self.output_dim), device=device)
                merged_list.append(merged_sample)
            
            padded = pad_sequence(merged_list, batch_first=True, padding_value=0.0)
            return padded
        
        # last_hidden_state
        padded_last = process_layer_output(last_hidden_state)
        
        # mask
        output_mask = torch.zeros((batch_size, padded_last.size(1)), dtype=torch.long, device=device)
        for i in range(batch_size):
            chunk_indices = sample_to_chunk_indices[i]
            total_len = sum(post_lengths[idx].item() for idx in chunk_indices)
            output_mask[i, :total_len] = 1
        
        # hidden states
        layer_hidden_states = {}
        if all_hidden_states is not None:
            for layer_idx in self.layer_indices:
                actual_idx = min(layer_idx, len(all_hidden_states) - 1)
                layer_output = all_hidden_states[actual_idx]
                layer_hidden_states[layer_idx] = process_layer_output(layer_output)
        
        return {
            "last_hidden_state": padded_last,
            "layer_hidden_states": layer_hidden_states,
            "attention_mask": output_mask,
        }


class Qwen2AudioMultiLayerEncoders(nn.Module):
    """
    Qwen2-Audio Encoder
    """
    def __init__(
        self, 
        model_id: str = "./Qwen2-Audio-7B",
        layer_indices: list[int] | None = None,
    ):
        super().__init__()
        
        if layer_indices is None:
            layer_indices = [8, 16, 32]
        self.layer_indices = layer_indices
        
        _clear_meta_device_context()
        
        print(f"[Qwen2Audio] Loading model from {model_id}...")
        full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        
        self.encoder = full_model.audio_tower
        self.processor = Qwen2AudioProcessor.from_pretrained(model_id)
        
        self.output_dim = self.encoder.config.d_model  # 1280
        self.num_layers = self.encoder.config.encoder_layers  # 32
        
        print(f"[Qwen2Audio] num_layers={self.num_layers}, hidden_size={self.output_dim}")
        print(f"[Qwen2Audio] Will extract layers: {layer_indices}")
        
        del full_model
        
        self.SAMPLE_RATE = 16000
        self.CHUNK_LENGTH = 30
        self.SAMPLES_PER_CHUNK = self.SAMPLE_RATE * self.CHUNK_LENGTH
        
    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Return:
            dict: {
                "last_hidden_state": [B, T, D],
                "layer_hidden_states": {layer_idx: [B, T, D]},
                "attention_mask": [B, T]
            }
        """
        device = audio.device
        batch_size = audio.size(0)
        total_samples = audio.shape[-1]
        
        batch_embeddings = []
        batch_layer_embeddings = {idx: [] for idx in self.layer_indices}
        
        for i, waveform in enumerate(audio):
            waveform_np = waveform.detach().cpu().numpy()
            total_len = waveform_np.shape[0]
            
            # chunk
            chunks = []
            if total_len > self.SAMPLES_PER_CHUNK:
                for start in range(0, total_len, self.SAMPLES_PER_CHUNK):
                    end = min(start + self.SAMPLES_PER_CHUNK, total_len)
                    chunk = waveform_np[start:end]
                    chunks.append(chunk)
            else:
                chunks.append(waveform_np)
            
            # features
            inputs = self.processor.feature_extractor(
                chunks, 
                sampling_rate=self.SAMPLE_RATE, 
                padding="max_length",
                return_tensors="pt"
            )
            
            chunk_features = inputs.input_features.to(device)
            
            # hidden_states
            chunk_outputs = self.encoder(
                chunk_features, 
                output_hidden_states=output_hidden_states
            )
            
            last_hidden = chunk_outputs.last_hidden_state
            all_hidden = chunk_outputs.hidden_states if output_hidden_states else None
            
            output_len = int(total_len / 16000 * 25)
            last_hidden = last_hidden[:, :output_len, :]
            
            # concat
            full_embedding = torch.cat([c for c in last_hidden], dim=0)
            batch_embeddings.append(full_embedding)
            
            if all_hidden is not None:
                for layer_idx in self.layer_indices:
                    actual_idx = min(layer_idx, len(all_hidden) - 1)
                    layer_hidden = all_hidden[actual_idx][:, :output_len, :]
                    layer_full = torch.cat([c for c in layer_hidden], dim=0)
                    batch_layer_embeddings[layer_idx].append(layer_full)
        
        # Pad batch
        max_len = max([e.size(0) for e in batch_embeddings])
        final_output = torch.zeros(batch_size, max_len, self.output_dim, device=device)
        
        for i, emb in enumerate(batch_embeddings):
            final_output[i, :emb.size(0), :] = emb
        
        # Pad hidden states
        layer_hidden_states = {}
        for layer_idx in self.layer_indices:
            layer_output = torch.zeros(batch_size, max_len, self.output_dim, device=device)
            for i, emb in enumerate(batch_layer_embeddings[layer_idx]):
                layer_output[i, :emb.size(0), :] = emb
            layer_hidden_states[layer_idx] = layer_output
        
        # mask
        output_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        for i, emb in enumerate(batch_embeddings):
            output_mask[i, :emb.size(0)] = 1
        
        return {
            "last_hidden_state": final_output,
            "layer_hidden_states": layer_hidden_states,
            "attention_mask": output_mask,
        }


# ========== Expert Networks ==========
class SiLUExpert(nn.Module):
    """ SiLU MLP Expert """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLUExpert(nn.Module):
    """
    SwiGLU Expert
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.swiglu_hidden = int(hidden_dim * 2 / 3)
        
        self.w_gate = nn.Linear(input_dim, self.swiglu_hidden, bias=False)
        self.w_up = nn.Linear(input_dim, self.swiglu_hidden, bias=False)
        self.w_down = nn.Linear(self.swiglu_hidden, output_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        hidden = F.silu(up) * gate
        hidden = self.dropout(hidden)
        output = self.w_down(hidden)
        output = self.dropout(output)
        return output


# ========== MoE Fusion Modules ==========
class MoEFusion(nn.Module):
    """ Standard MoE fusion module """
    def __init__(
        self,
        enc1_dim: int,
        enc2_dim: int,
        output_dim: int | None = None,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.enc1_dim = enc1_dim
        self.enc2_dim = enc2_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        if output_dim is None:
            output_dim = max(enc1_dim, enc2_dim)
        self.output_dim = output_dim
        
        if hidden_dim is None:
            hidden_dim = output_dim * 4
        self.hidden_dim = hidden_dim
        
        self.proj_enc1 = nn.Linear(enc1_dim, output_dim)
        self.proj_enc2 = nn.Linear(enc2_dim, output_dim)
        self.input_norm = RMSNorm(output_dim * 2)
        
        self.router = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, num_experts),
        )
        
        self.experts = nn.ModuleList([
            SiLUExpert(
                input_dim=output_dim * 2,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])
        
        self.out_norm = RMSNorm(output_dim)
        
    def forward(
        self,
        enc1_features: torch.Tensor,
        enc2_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.proj_enc1(enc1_features)
        enc2 = self.proj_enc2(enc2_features)
        
        combined = torch.cat([enc1, enc2], dim=-1)
        combined = self.input_norm(combined)
        
        router_logits = self.router(combined)
        topk_weights, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)
        
        expert_outputs = torch.stack([
            expert(combined) for expert in self.experts
        ], dim=2)
        
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.output_dim)
        selected_outputs = torch.gather(expert_outputs, 2, topk_indices_expanded)
        
        output = (selected_outputs * topk_weights.unsqueeze(-1)).sum(dim=2)
        output = self.out_norm(output)
        
        return output, attention_mask


class MoESwiGLUFusion(nn.Module):
    """ Improved MoE fusion module – SwiGLU experts + shared expert """
    def __init__(
        self,
        enc1_dim: int,
        enc2_dim: int,
        output_dim: int | None = None,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        use_shared_expert: bool = True,
    ):
        super().__init__()
        self.enc1_dim = enc1_dim
        self.enc2_dim = enc2_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert
        
        if output_dim is None:
            output_dim = max(enc1_dim, enc2_dim)
        self.output_dim = output_dim
        
        if hidden_dim is None:
            hidden_dim = output_dim * 4
        
        self.proj_enc1 = nn.Linear(enc1_dim, output_dim)
        self.proj_enc2 = nn.Linear(enc2_dim, output_dim)
        self.input_norm = RMSNorm(output_dim * 2)
        
        self.router = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(output_dim, num_experts),
        )
        
        self.experts = nn.ModuleList([
            SwiGLUExpert(
                input_dim=output_dim * 2,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])
        
        if use_shared_expert:
            self.shared_expert = SwiGLUExpert(
                input_dim=output_dim * 2,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
            )
            self.shared_gate = nn.Parameter(torch.ones(1) * 0.5)
        
        self.out_norm = RMSNorm(output_dim)
        
    def forward(
        self,
        enc1_features: torch.Tensor,
        enc2_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.proj_enc1(enc1_features)
        enc2 = self.proj_enc2(enc2_features)
        
        combined = torch.cat([enc1, enc2], dim=-1)
        combined = self.input_norm(combined)
        
        router_logits = self.router(combined)
        topk_weights, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)
        
        expert_outputs = torch.stack([
            expert(combined) for expert in self.experts
        ], dim=2)
        
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.output_dim)
        selected_outputs = torch.gather(expert_outputs, 2, topk_indices_expanded)
        
        routed_output = (selected_outputs * topk_weights.unsqueeze(-1)).sum(dim=2)
        
        if self.use_shared_expert:
            shared_output = self.shared_expert(combined)
            gate = torch.sigmoid(self.shared_gate)
            output = gate * shared_output + (1 - gate) * routed_output
        else:
            output = routed_output
        
        output = self.out_norm(output)
        
        return output, attention_mask


class MoE3LayerFusion(nn.Module):
    """
    Multi-layer MoE fusion module:
    extract 8/16/32 layer features from both encoders
    """
    def __init__(
        self,
        enc1_dim: int,
        enc2_dim: int,
        output_dim: int | None = None,
        layer_indices: list[int] | None = None,
        experts_per_layer: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        aggregation_mode: str = "weighted",  # "weighted", "concat", "add"
    ):
        super().__init__()
        self.enc1_dim = enc1_dim
        self.enc2_dim = enc2_dim
        self.experts_per_layer = experts_per_layer
        self.aggregation_mode = aggregation_mode
        
        if layer_indices is None:
            layer_indices = [8, 16, 32]
        self.layer_indices = layer_indices
        self.num_layers = len(layer_indices)
        
        if output_dim is None:
            output_dim = max(enc1_dim, enc2_dim)
        self.layer_output_dim = output_dim
        
        if hidden_dim is None:
            hidden_dim = output_dim * 4
        
        # layer fusion
        self.layer_fusions = nn.ModuleDict()
        for layer_idx in layer_indices:
            self.layer_fusions[str(layer_idx)] = LayerMoEFusion(
                enc1_dim=enc1_dim,
                enc2_dim=enc2_dim,
                output_dim=output_dim,
                num_experts=experts_per_layer,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        
        if aggregation_mode == "weighted":
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
            self.output_dim = output_dim
        elif aggregation_mode == "concat":
            self.concat_proj = nn.Sequential(
                nn.Linear(output_dim * self.num_layers, output_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.output_dim = output_dim
        else:  # add
            self.output_dim = output_dim
        
        self.out_norm = RMSNorm(self.output_dim)
        
    def forward(
        self,
        enc1_layer_features: dict[int, torch.Tensor],  # {layer_idx: [B, T, D]}
        enc2_layer_features: dict[int, torch.Tensor],  # {layer_idx: [B, T, D]}
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            enc1_layer_features: Flamingo3 {8: [B,T,D], 16: [B,T,D], 32: [B,T,D]}
            enc2_layer_features: Qwen2 {8: [B,T,D], 16: [B,T,D], 32: [B,T,D]}
            attention_mask: [B, T]
        
        Returns:
            output: [B, T, output_dim]
            attention_mask: [B, T]
        """
        layer_outputs = []
        
        for layer_idx in self.layer_indices:
            enc1_feat = enc1_layer_features[layer_idx]
            enc2_feat = enc2_layer_features[layer_idx]
            
            min_len = min(enc1_feat.shape[1], enc2_feat.shape[1])
            enc1_feat = enc1_feat[:, :min_len, :]
            enc2_feat = enc2_feat[:, :min_len, :]
            mask = attention_mask[:, :min_len]
            
            layer_out, _ = self.layer_fusions[str(layer_idx)](
                enc1_feat, enc2_feat, mask
            )
            layer_outputs.append(layer_out)
        
        # aggregation
        if self.aggregation_mode == "weighted":
            weights = F.softmax(self.layer_weights, dim=0)
            output = sum(w * out for w, out in zip(weights, layer_outputs))
        elif self.aggregation_mode == "concat":
            output = torch.cat(layer_outputs, dim=-1)
            output = self.concat_proj(output)
        else:  # add
            output = sum(layer_outputs) / len(layer_outputs)
        
        output = self.out_norm(output)
        
        # mask
        final_mask = attention_mask[:, :output.shape[1]]
        
        return output, final_mask


class LayerMoEFusion(nn.Module):
    """
    Single-layer MoE fusion module (for MoE3LayerFusion)
    """
    def __init__(
        self,
        enc1_dim: int,
        enc2_dim: int,
        output_dim: int,
        num_experts: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        
        if hidden_dim is None:
            hidden_dim = output_dim * 4
        
        # projector
        self.proj_enc1 = nn.Linear(enc1_dim, output_dim)
        self.proj_enc2 = nn.Linear(enc2_dim, output_dim)
        self.input_norm = RMSNorm(output_dim * 2)
        
        # router
        self.router = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim // 2),
            nn.SiLU(),
            nn.Linear(output_dim // 2, num_experts),
        )
        
        # SwiGLU Expert
        self.experts = nn.ModuleList([
            SwiGLUExpert(
                input_dim=output_dim * 2,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])
        
        self.residual_proj = nn.Linear(output_dim * 2, output_dim)
        self.out_norm = RMSNorm(output_dim)
        
    def forward(
        self,
        enc1_features: torch.Tensor,
        enc2_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.proj_enc1(enc1_features)
        enc2 = self.proj_enc2(enc2_features)
        
        combined = torch.cat([enc1, enc2], dim=-1)
        combined_normed = self.input_norm(combined)
        
        router_logits = self.router(combined_normed)
        
        # Top-1
        topk_weights, topk_indices = torch.topk(router_logits, 1, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)
        
        expert_outputs = torch.stack([
            expert(combined_normed) for expert in self.experts
        ], dim=2)
        
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.output_dim)
        selected_outputs = torch.gather(expert_outputs, 2, topk_indices_expanded)
        
        output = (selected_outputs * topk_weights.unsqueeze(-1)).sum(dim=2)
        
        residual = self.residual_proj(combined)
        output = output + residual
        
        output = self.out_norm(output)
        
        return output, attention_mask


# ========== Main Model ==========
class FlamingoQwen2MoEEncoder(nn.Module):
    """
    Audio Flamingo 3 + Qwen2-Audio MoE Fusion Encoder
    
    mode:
        - "moe": standard MoE fusion – uses MLP experts with SiLU
        - "moe_3layer": multi-layer MoE fusion – fuses 8/16/32 layer
        - "moe_swiglu": improved MoE fusion – uses SwiGLU Expert (SOTA)
    """
    
    SAMPLE_RATE = 16000
    MAX_AUDIO_SECONDS = 30.0
    MAX_AUDIO_LENGTH = int(MAX_AUDIO_SECONDS * SAMPLE_RATE)
    
    def __init__(
        self,
        fusion_mode: str = "moe_swiglu",
        checkpoint_path: str | None = "./checkpoints",
        fusion_output_dim: int | None = None,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_hidden_dim: int | None = None,
        fusion_dropout: float = 0.1,
        use_shared_expert: bool = True,
        # moe_3layer
        layer_indices: list[int] | None = None,
        experts_per_layer: int = 2,
        aggregation_mode: str = "weighted",
        # general
        freeze_encoders: bool = True,
        freeze_fusion: bool = False,
        use_adapter: bool = False,
        max_audio_seconds: float = 30.0,
        **kwargs,
    ):
        super().__init__()
        
        # fusion_mode
        valid_modes = ("moe", "moe_swiglu", "moe_3layer")
        if fusion_mode not in valid_modes:
            raise ValueError(f"fusion_mode must be one of {valid_modes}, got '{fusion_mode}'")
        
        self.use_adapter = True
        self.fusion_mode = fusion_mode
        self.freeze_encoders = freeze_encoders
        self.max_audio_length = int(max_audio_seconds * self.SAMPLE_RATE)
        
        if layer_indices is None:
            layer_indices = [8, 16, 32]
        self.layer_indices = layer_indices
        
        # Initialize encoder
        if fusion_mode == "moe_3layer":
            print(f"[Info] Using multi-layer encoders with layers: {layer_indices}")
            self.flamingo_encoder = AudioFlamingo3MultiLayerEncoders(layer_indices=layer_indices)
            self.qwen2_encoder = Qwen2AudioMultiLayerEncoders(layer_indices=layer_indices)
        else:
            # moe/moe_swiglu
            from models.uniae.audio_flamingo import AudioFlamingo3Encoder as flamingo
            from models.uniae.qwen2_audio import Qwen2AudioEncoder as qwen2
            self.flamingo_encoder = flamingo()
            self.qwen2_encoder = qwen2()
        
        self.enc1_dim = self.flamingo_encoder.output_dim
        self.enc2_dim = self.qwen2_encoder.output_dim
        
        print(f"[Info] Audio Flamingo 3 output_dim: {self.enc1_dim}")
        print(f"[Info] Qwen2-Audio output_dim: {self.enc2_dim}")
        
        # freeze encoders
        if freeze_encoders:
            for param in self.flamingo_encoder.parameters():
                param.requires_grad = False
            for param in self.qwen2_encoder.parameters():
                param.requires_grad = False
            print("[Info] Base encoders frozen")
        
        # output dim
        if fusion_output_dim is None:
            fusion_output_dim = max(self.enc1_dim, self.enc2_dim)
        
        # create fusion mode
        if fusion_mode == "moe":
            print(f"[Info] Using Standard MoE Fusion (experts={moe_num_experts}, top_k={moe_top_k})")
            self.fusion_module = MoEFusion(
                enc1_dim=self.enc1_dim,
                enc2_dim=self.enc2_dim,
                output_dim=fusion_output_dim,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                hidden_dim=moe_hidden_dim,
                dropout=fusion_dropout,
            )
            self.output_dim = fusion_output_dim
            
        elif fusion_mode == "moe_swiglu":
            print(f"[Info] Using SwiGLU MoE Fusion (experts={moe_num_experts}, top_k={moe_top_k}, shared={use_shared_expert})")
            self.fusion_module = MoESwiGLUFusion(
                enc1_dim=self.enc1_dim,
                enc2_dim=self.enc2_dim,
                output_dim=fusion_output_dim,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                hidden_dim=moe_hidden_dim,
                dropout=fusion_dropout,
                use_shared_expert=use_shared_expert,
            )
            self.output_dim = fusion_output_dim
            
        else:  # moe_3layer
            print(f"[Info] Using 3-Layer MoE Fusion (layers={layer_indices}, experts_per_layer={experts_per_layer})")
            self.fusion_module = MoE3LayerFusion(
                enc1_dim=self.enc1_dim,
                enc2_dim=self.enc2_dim,
                output_dim=fusion_output_dim,
                layer_indices=layer_indices,
                experts_per_layer=experts_per_layer,
                hidden_dim=moe_hidden_dim,
                dropout=fusion_dropout,
                aggregation_mode=aggregation_mode,
            )
            self.output_dim = self.fusion_module.output_dim
        
        # Compatible training framework
        self.audio_adapter = None
        self.audio_qformer = None
        
        cache_dir = "./cache"
        model_name = "Syclus/UniAE-MoE"

        snapshot_download(
            repo_id=model_name,
            local_dir=checkpoint_path,
            cache_dir=cache_dir,
            # local_files_only=False,
        )

        # load checkpoint
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                self._load_fusion_weights(checkpoint_path)
            else:
                print(f"[Warning] Checkpoint path does not exist: {checkpoint_path}")
        else:
            print("[Warning] No checkpoint_path provided, using randomly initialized fusion weights")
        
        if freeze_fusion and self.fusion_module is not None:
            for param in self.fusion_module.parameters():
                param.requires_grad = False
            print("[Info] Fusion module frozen")
    
    def _load_fusion_weights(self, checkpoint_path: Path):
        """ Load fusion module weights """
        print(f"[Info] Loading fusion weights from: {checkpoint_path}")
        
        safetensors_path = checkpoint_path / "model.safetensors"
        pytorch_path = checkpoint_path / "pytorch_model.bin"
        safetensors_index_path = checkpoint_path / "model.safetensors.index.json"
        
        state_dict = {}
        
        if safetensors_path.exists():
            state_dict = load_file(str(safetensors_path))
            print(f"[Info] Loaded single safetensors file")
        elif safetensors_index_path.exists():
            import json
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            shard_files = set(index['weight_map'].values())
            for shard_file in sorted(shard_files):
                shard_path = checkpoint_path / shard_file
                if shard_path.exists():
                    shard_state = load_file(str(shard_path))
                    state_dict.update(shard_state)
        elif pytorch_path.exists():
            state_dict = torch.load(pytorch_path, map_location="cpu")
        else:
            print(f"[Warning] No model weights found in {checkpoint_path}")
            return
        
        if len(state_dict) == 0:
            return
        
        fusion_state = {}
        fusion_keywords = ['proj_enc', 'input_norm', 'router', 'experts', 'shared_expert', 
                          'shared_gate', 'out_norm', 'w_gate', 'w_up', 'w_down', 
                          'layer_fusions', 'layer_weights', 'concat_proj', 'residual_proj']
        
        for k, v in state_dict.items():
            if k.startswith("audio_encoder.fusion_module."):
                key = k.replace("audio_encoder.fusion_module.", "")
                fusion_state[key] = v
        
        if len(fusion_state) == 0:
            for k, v in state_dict.items():
                if k.startswith("fusion_module."):
                    key = k.replace("fusion_module.", "")
                    fusion_state[key] = v
        
        if len(fusion_state) == 0:
            for k, v in state_dict.items():
                if any(keyword in k for keyword in fusion_keywords):
                    fusion_state[k] = v
        
        if len(fusion_state) > 0 and self.fusion_module is not None:
            missing, unexpected = self.fusion_module.load_state_dict(fusion_state, strict=False)
            print(f"[Info] Loaded fusion weights: {len(fusion_state)} params")
            if missing:
                print(f"[Warning] Missing keys: {missing[:5]} ...")
            if unexpected:
                print(f"[Warning] Unexpected keys: {unexpected[:5]} ...")
    
    def save_fusion_weights(self, save_path: str | Path, use_safetensors: bool = True):
        """ save """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        state_dict = {
            f"fusion_module.{k}": v 
            for k, v in self.fusion_module.state_dict().items()
        }
        
        if use_safetensors:
            save_file(state_dict, save_path / "model.safetensors")
        else:
            torch.save(state_dict, save_path / "pytorch_model.bin")
        
        config = {
            "fusion_mode": self.fusion_mode,
            "output_dim": self.output_dim,
            "enc1_dim": self.enc1_dim,
            "enc2_dim": self.enc2_dim,
            "layer_indices": self.layer_indices if self.fusion_mode == "moe_3layer" else None,
        }
        import json
        with open(save_path / "fusion_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"[Info] Saved fusion weights to: {save_path}")
    
    def forward(
        self,
        audio: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
        task_id: torch.Tensor | None = None,
        query_budget: str = "gen",
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # audio truncation
        if audio.shape[1] > self.max_audio_length:
            audio = audio[:, :self.max_audio_length]
            if audio_attention_mask is not None:
                audio_attention_mask = audio_attention_mask[:, :self.max_audio_length]
        
        if self.fusion_mode == "moe_3layer":
            with torch.no_grad() if self.freeze_encoders else torch.enable_grad():
                flamingo_out = self.flamingo_encoder(audio, audio_attention_mask, output_hidden_states=True)
                qwen2_out = self.qwen2_encoder(audio, audio_attention_mask, output_hidden_states=True)
            
            enc1_layer_features = flamingo_out["layer_hidden_states"]
            enc2_layer_features = qwen2_out["layer_hidden_states"]
            attention_mask = flamingo_out["attention_mask"]
            
            min_len = min(
                min(enc1_layer_features[idx].shape[1] for idx in self.layer_indices),
                min(enc2_layer_features[idx].shape[1] for idx in self.layer_indices),
            )
            
            for idx in self.layer_indices:
                enc1_layer_features[idx] = enc1_layer_features[idx][:, :min_len, :]
                enc2_layer_features[idx] = enc2_layer_features[idx][:, :min_len, :]
            attention_mask = attention_mask[:, :min_len]
            
            # fusion
            device = next(self.fusion_module.parameters()).device
            for idx in self.layer_indices:
                enc1_layer_features[idx] = enc1_layer_features[idx].to(device)
                enc2_layer_features[idx] = enc2_layer_features[idx].to(device)
            attention_mask = attention_mask.to(device)
            
            output, output_mask = self.fusion_module(
                enc1_layer_features,
                enc2_layer_features,
                attention_mask,
            )
        else:
            # moe / moe_swiglu
            with torch.no_grad() if self.freeze_encoders else torch.enable_grad():
                flamingo_out, flamingo_mask = self.flamingo_encoder(audio, audio_attention_mask)
                qwen2_out, _ = self.qwen2_encoder(audio, audio_attention_mask)
            
            min_len = min(flamingo_out.shape[1], qwen2_out.shape[1])
            flamingo_out = flamingo_out[:, :min_len, :]
            qwen2_out = qwen2_out[:, :min_len, :]
            attention_mask = flamingo_mask[:, :min_len]
            
            device = next(self.fusion_module.parameters()).device
            flamingo_out = flamingo_out.to(device)
            qwen2_out = qwen2_out.to(device)
            attention_mask = attention_mask.to(device)
            
            output, output_mask = self.fusion_module(
                flamingo_out,
                qwen2_out,
                attention_mask,
            )
        
        return output, output_mask


# ========== Utils ==========
def _num_params(module: nn.Module | None, trainable_only: bool = False) -> int:
    if module is None:
        return 0
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _fmt_mparams(n: int) -> str:
    return f"{n/1e6:.3f} M"


def print_model_report(model: nn.Module, title: str = ""):
    if title:
        print(f"\n================= {title} =================")
    
    total_p = _num_params(model, trainable_only=False)
    train_p = _num_params(model, trainable_only=True)
    ratio = (train_p / max(total_p, 1)) * 100.0
    print(f"[Total Params]      {_fmt_mparams(total_p)}")
    print(f"[Trainable Params]  {_fmt_mparams(train_p)}  ({ratio:.2f}%)")
    
    flamingo = getattr(model, "flamingo_encoder", None)
    qwen2 = getattr(model, "qwen2_encoder", None)
    fusion = getattr(model, "fusion_module", None)
    
    if flamingo is not None:
        print(f"[Flamingo Encoder]  {_fmt_mparams(_num_params(flamingo))}")
    if qwen2 is not None:
        print(f"[Qwen2 Encoder]     {_fmt_mparams(_num_params(qwen2))}")
    if fusion is not None:
        print(f"[Fusion Module]     {_fmt_mparams(_num_params(fusion))} (trainable {_fmt_mparams(_num_params(fusion, True))})")
    
    print(f"[Output dim]        {model.output_dim}")
    print(f"[Fusion mode]       {model.fusion_mode}")


def print_trainable_names(model: nn.Module, max_lines: int = 30):
    cnt = 0
    print("\n[Trainable Parameters]")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  [T] {n}  shape={tuple(p.shape)}")
            cnt += 1
            if cnt >= max_lines:
                print("  ... (truncated)")
                break


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    test_audio = torch.randn(2, 160000).to(device)
    attention_mask_lens = torch.tensor([160000, 80000]).to(device)
    audio_attention_mask = length_to_mask(attention_mask_lens, max_len=160000).to(device)
    
    print("=" * 60)
    print("input:", test_audio.shape)
    
    # test
    fusion_modes = ["moe", "moe_swiglu", "moe_3layer"]
    
    for fusion_mode in fusion_modes:
        print(f"\n{'='*60}")
        print(f"Testing fusion_mode = '{fusion_mode}'")
        print('='*60)
        
        kwargs = {
            "fusion_mode": fusion_mode,
            "fusion_output_dim": 1280,
            "fusion_dropout": 0.1,
            "max_audio_seconds": 30.0,
        }
        
        if fusion_mode in ["moe", "moe_swiglu"]:
            kwargs["moe_num_experts"] = 4
            kwargs["moe_top_k"] = 2
        
        if fusion_mode == "moe_swiglu":
            kwargs["use_shared_expert"] = True
        
        if fusion_mode == "moe_3layer":
            kwargs["layer_indices"] = [8, 16, 32]
            kwargs["experts_per_layer"] = 2
            kwargs["aggregation_mode"] = "weighted"
        
        encoder = FlamingoQwen2MoEEncoder(**kwargs).to(device)
        
        print_model_report(encoder, f"Fusion Mode: {fusion_mode}")
        
        encoder.train()
        with torch.no_grad():
            out, out_mask = encoder(test_audio, audio_attention_mask)
        print(f"[Forward] out={tuple(out.shape)} mask={tuple(out_mask.shape)}")
        
        print_trainable_names(encoder, max_lines=10)
        
        del encoder
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("=" * 60)
