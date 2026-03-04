"""
Audio Flamingo 3 Encoder
"""
import torch
from torch import nn
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
from transformers import AudioFlamingo3Processor
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import snapshot_download

def _clear_meta_device_context():
    """Clear the meta-device context"""
    original_device = torch.get_default_device()
    if original_device is not None and original_device.type == "meta":
        torch.set_default_device(None)


class AudioFlamingo3Encoder(nn.Module):
    def __init__(self, model_name="nvidia/audio-flamingo-3-hf"):
        super().__init__()
        
        # Clear meta-device context to avoid nested from_pretrained conflicts
        _clear_meta_device_context()
        
        print(f"Loading full model from {model_name} to extract encoder...")
        cache_dir="./cache"
        local_dir="./audio-flamingo-3-hf"

        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            cache_dir=cache_dir,
            # local_files_only=False,
        )

        full_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            local_dir, 
            trust_remote_code=True,
            torch_dtype=torch.float
        )
        
        # Audio Encoder
        self.encoder = full_model.audio_tower
        
        # hidden_size for output_dim
        self.output_dim = self.encoder.config.d_model
        
        del full_model.language_model
        del full_model.multi_modal_projector
        del full_model
        
        self.processor = AudioFlamingo3Processor.from_pretrained(local_dir, trust_remote_code=True)
        
        self.SAMPLE_RATE = 16000
        # Whisper 30s (1500 frame)
        self.MAX_LENGTH_SECONDS = 30.0
        self.CHUNK_LENGTH = 30 * 16000
        
    def forward(
        self,
        audio: torch.Tensor,
        audio_attention_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        return:
            padded_embeddings: [B, Max_Valid_Seq_Len, Dim]
            output_mask: [B, Max_Valid_Seq_Len]
        """
        device = audio.device
        batch_size = audio.size(0)

        audio = audio.to(self.encoder.dtype)
        
        # chunking
        all_chunks_list = []
        sample_to_chunk_indices = [[] for _ in range(batch_size)]
        
        audio_cpu = audio.detach().cpu()
        mask_cpu = audio_attention_mask.detach().cpu() if audio_attention_mask is not None else torch.ones_like(audio_cpu)

        global_chunk_idx = 0
        
        for i in range(batch_size):
            valid_length = mask_cpu[i].sum().int().item()
            valid_wav = audio_cpu[i, :valid_length].float().numpy()
            
            if len(valid_wav) == 0:
                # short/empty audio
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

        # feature extract
        # padding=max_length -> 30
        inputs = self.processor.feature_extractor(
            all_chunks_list, 
            sampling_rate=self.SAMPLE_RATE,
            padding="max_length", 
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        input_features = inputs.input_features.to(device).to(self.encoder.dtype)
        input_features_mask = inputs.attention_mask.to(device)

        # Encoder Forward
        encoder_outputs = self.encoder(
            input_features=input_features,
            input_features_mask=input_features_mask
        )
        # output: [Total_Chunks, 1500, Dim]
        last_hidden_state = encoder_outputs.last_hidden_state
        
        # input_features_mask.shape: [Total_Chunks, 3000]
        conv_output_lengths = (input_features_mask.sum(dim=-1) - 1) // 2 + 1
        post_lengths = (conv_output_lengths - 2) // 2 + 1
        max_len = last_hidden_state.shape[1]
        post_lengths = torch.clamp(post_lengths, min=0, max=max_len).long()
        # Re-assemble
        merged_embeddings_list = []
        
        for i in range(batch_size):
            chunk_indices = sample_to_chunk_indices[i]
            
            sample_parts = []
            for idx in chunk_indices:
                raw_embed = last_hidden_state[idx] # [1500, Dim]
                valid_len = post_lengths[idx]
                valid_embed = raw_embed[:valid_len]
                
                sample_parts.append(valid_embed)
            
            if len(sample_parts) > 0:
                merged_sample = torch.cat(sample_parts, dim=0)
            else:
                merged_sample = torch.zeros((1, self.output_dim), device=device)
                
            merged_embeddings_list.append(merged_sample)
        
        # Batch Padding -> [B, Max_Total_Len, Dim]
        padded_embeddings = pad_sequence(merged_embeddings_list, batch_first=True, padding_value=0.0)
        
        # Output Mask
        output_mask = torch.zeros(
            (batch_size, padded_embeddings.size(1)), 
            dtype=torch.long, 
            device=device
        )
        for i, sample_embed in enumerate(merged_embeddings_list):
            output_mask[i, :sample_embed.size(0)] = 1
        
        return padded_embeddings, output_mask

if __name__ == "__main__":
    torch.manual_seed(42)
    audio = torch.randn(1, 16000) 
    print(f"input: {audio.shape}")

    encoder = AudioFlamingo3Encoder()
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        audio = audio.cuda()

    def count_parameters(model):
        """
        Count model parameters.
        Returns: 
            total parameters, trainable parameters, non-trainable parameters.
        """
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        for name, param in model.named_parameters():
            param_num = param.numel()
            total_params += param_num
            
            if param.requires_grad:
                trainable_params += param_num
            else:
                non_trainable_params += param_num
        
        return total_params, trainable_params, non_trainable_params

    total, trainable, non_trainable = count_parameters(encoder)

    print(f"Parameters: {total / 1000000000:,}")
    print(f"Trainable parameters: {trainable / 100000000:,}")
    print(f"Frozen parameters: {non_trainable / 100000000:,}")

    output, _ = encoder(audio)
    
    print(f"shape: {output.shape}") 
    print(output)

    model_id = "nvidia/audio-flamingo-3-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

    # 1) encoder layer
    print("AF3 audio num_hidden_layers =", model.config.audio_config.num_hidden_layers)
    print("AF3 audio hidden_size      =", model.config.audio_config.hidden_size)
    print("AF3 text num_hidden_layers =", model.config.text_config.num_hidden_layers)
    print("AF3 text hidden_size       =", model.config.text_config.hidden_size)
