"""
Qwen2-Audio
"""
import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from transformers import Qwen2AudioProcessor, Qwen2AudioEncoderConfig, Qwen2AudioForConditionalGeneration


def _clear_meta_device_context():
    """Clear the meta-device context"""
    original_device = torch.get_default_device()
    if original_device is not None and original_device.type == "meta":
        torch.set_default_device(None)
        # print("clean meta device context")


class Qwen2AudioEncoder(nn.Module):
    def __init__(self, model_id="Qwen/Qwen2-Audio-7B"):
        super().__init__()
        
        # Clear meta-device context to avoid nested from_pretrained conflicts
        _clear_meta_device_context()

        cache_dir="./cache"
        local_dir="./Qwen2-Audio-7B"

        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            cache_dir=cache_dir,
            # local_files_only=False,
        )
        
        full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                local_dir, 
                # device_map="auto", 
                trust_remote_code=True
            )
        self.encoder = full_model.audio_tower
        self.processor = Qwen2AudioProcessor.from_pretrained(local_dir)

        self.output_dim = self.encoder.config.d_model
        
        # define
        self.SAMPLE_RATE = 16000
        self.CHUNK_LENGTH = 30 # 30s
        self.SAMPLES_PER_CHUNK = self.SAMPLE_RATE * self.CHUNK_LENGTH

    def forward(self, audio, attention_mask=None):
        """
        Forward function that supports arbitrary-length audio
        Args:
            audio: [B, T] raw audio tensor
            attention_mask: [B, T] audio_mask
        """
        input_audio_len = audio.shape[-1]

        device = audio.device
        batch_embeddings = []
        
        for i, waveform in enumerate(audio):
            waveform_np = waveform.detach().cpu().numpy()
            
            # chunking
            total_samples = waveform_np.shape[0]
            chunks = []
            
            if total_samples > self.SAMPLES_PER_CHUNK:
                for start in range(0, total_samples, self.SAMPLES_PER_CHUNK):
                    end = min(start + self.SAMPLES_PER_CHUNK, total_samples)
                    chunk = waveform_np[start:end]
                    # auto pad
                    chunks.append(chunk)
            else:
                chunks.append(waveform_np)
            
            # chunk pad
            inputs = self.processor.feature_extractor(
                chunks, 
                sampling_rate=self.SAMPLE_RATE, 
                padding="max_length",
                return_tensors="pt"
            )
            
            chunk_features = inputs.input_features.to(device) # [Num_Chunks, 80, 3000]
            
            # [Num_Chunks, 1500, D]
            chunk_outputs = self.encoder(chunk_features).last_hidden_state # [chunks, 750, D]
            
            output_len = int(total_samples / 16000 * 25)
            chunk_outputs = chunk_outputs[:, :output_len, :]  # [chunk, real length, D]
            
            # concat
            full_embedding = torch.cat([c for c in chunk_outputs], dim=0) 
            
            batch_embeddings.append(full_embedding)

        # batch pad
        max_len = max([e.size(0) for e in batch_embeddings])
        final_output = torch.zeros(len(batch_embeddings), max_len, self.output_dim, device=device)
        
        for i, emb in enumerate(batch_embeddings):
            final_output[i, :emb.size(0), :] = emb

        return final_output, None


if __name__ == "__main__":
    audio_input = torch.randn(1, 160000) 
    attention_mask = torch.randint(0, audio_input.shape[-1], (2,))
    print(attention_mask)

    qwen2 = Qwen2AudioEncoder()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qwen2.to(device)
    audio_input = audio_input.to(device)
        
    output, _ = qwen2(audio_input, attention_mask)
    
    print(f"Output shape: {output.shape}") # [b, s, h]
    
    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "/models/qwen2-audio",
        trust_remote_code=True
    )

    enc = full_model.audio_tower
    print("Qwen2Audio encoder_layers =", enc.config.encoder_layers)
    print("Qwen2Audio d_model        =", enc.config.d_model)
    # layer
    # AF3 audio num_hidden_layers = 32
    # AF3 audio hidden_size      = 1280
    # AF3 text num_hidden_layers = 28
    # AF3 text hidden_size       = 3584