"""
Model for two-stage Encoder Tuning Training

Training Strategy:
  Stage 1: Freeze the Encoder and LLM, train the internal Bridge and Projector
  Stage 2: Freeze the Encoder, train the internal Bridge, Projector, and LLM (LoRA)
"""

import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Any, Literal

from xares_llm.audio_encoder_checker import check_audio_encoder
from xares_llm.utils import attr_from_module, attr_from_py_path


TrainingStage = Literal["stage1", "stage2"]


class EncoderTuningModelConfig(PretrainedConfig):
    """
    Encoder Tuning Model Configuration
    
    Note: Bridge configurations (adapter_type, qformer...) should be passed to the Encoder,which will create the internal Bridge itself. These parameters are only for backward compatibility.
    """
    
    model_type = "encoder_tuning_model"
    
    def __init__(
        self,
        audio_encoder_name: str | None = None,
        audio_encoder_params: Dict[str, Any] = {},
        decoder_type: str = "/private/models/SmolLM2-135M",
        # Encoder training control
        train_encoder_stage1: bool = False, 
        train_encoder_stage2: bool = False,
        # LoRA
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: str = "all-linear",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.decoder_type = decoder_type
        self.audio_encoder_name = audio_encoder_name
        self.audio_encoder_params = audio_encoder_params
        # Encoder training control
        self.train_encoder_stage1 = train_encoder_stage1
        self.train_encoder_stage2 = train_encoder_stage2
        # LoRA config
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules


class EncoderTuningModel(PreTrainedModel, nn.Module):
    """
    Fine-tuning the model with an Encoder that supports two-stage training
    Audio -> Encoder -> Projector(MLP) -> LLM
    """
    
    config_class = EncoderTuningModelConfig
    _tied_weights_keys = []
    
    def __init__(self, config: EncoderTuningModelConfig, training_stage: TrainingStage = "stage1") -> None:
        super().__init__(config)
        self.config = config
        self.training_stage = training_stage
        
        original_default_device = torch.get_default_device()
        if original_default_device is not None and original_default_device.type == "meta":
            torch.set_default_device(None)
            logger.info("[Meta Device] 退出 meta device 上下文以加载模型（不恢复）")
        
        # load Audio Encoder
        if Path(self.config.audio_encoder_name).is_file():
            audio_encoder = attr_from_py_path(self.config.audio_encoder_name, endswith="Encoder")(
                **self.config.audio_encoder_params
            )
        else:
            audio_encoder = attr_from_module(self.config.audio_encoder_name)(**self.config.audio_encoder_params)
        
        # check encoder
        try:
            audio_encoder_parameters = list(audio_encoder.parameters())
            if len(audio_encoder_parameters) > 0:
                device_type = audio_encoder_parameters[0].device.type
                if device_type != "meta":
                    check_audio_encoder(audio_encoder)
        except Exception as e:
            logger.exception(e)
            return
            
        self.audio_encoder = audio_encoder
        encoder_output_dim = self.audio_encoder.output_dim
        
        encoder_has_adapter = (
            hasattr(audio_encoder, 'audio_adapter') and audio_encoder.audio_adapter is not None
        )
        encoder_has_qformer = (
            hasattr(audio_encoder, 'audio_qformer') and audio_encoder.audio_qformer is not None
        )
        encoder_has_fusion = (
            hasattr(audio_encoder, 'fusion_module') and audio_encoder.fusion_module is not None
        )
        encoder_use_adapter = getattr(audio_encoder, 'use_adapter', False)

        self.use_encoder_builtin_bridge = encoder_use_adapter and (encoder_has_adapter or encoder_has_qformer or encoder_has_fusion)

        # bridge type
        self.bridge_type = "none"
        
        if self.use_encoder_builtin_bridge:
            if encoder_has_qformer:
                logger.info(f"[Bridge] Encoder has builtin QFormer (output_dim={encoder_output_dim})")
                self.bridge_type = "qformer"
                self.cls_task_ids = getattr(audio_encoder, 'cls_task_ids', set())
            elif encoder_has_adapter:
                logger.info(f"[Bridge] Encoder has builtin Adapter (output_dim={encoder_output_dim})")
                self.bridge_type = "adapter"
            elif encoder_has_fusion:
                logger.info(f"[Bridge] Encoder has builtin FusionModule (output_dim={encoder_output_dim})")
                self.bridge_type = "fusion"
        else:
            logger.info(f"[Bridge] Encoder has no builtin Bridge, use Encoder output directly (output_dim={encoder_output_dim})")
        
        projector_input_dim = encoder_output_dim
        
        # audioflamingo:
        is_in_meta_device = torch.get_default_device() is not None and torch.get_default_device().type == "meta"
        
        # load LLM Decoder
        if is_in_meta_device:
            logger.info(f"[Meta Device Mode] Creating decoder structure without loading weights...")
            from transformers import AutoConfig
            decoder_config = AutoConfig.from_pretrained(config.decoder_type)
            decoder = AutoModelForCausalLM.from_config(decoder_config)
        else:
            decoder = AutoModelForCausalLM.from_pretrained(config.decoder_type)
        
        # set pad_token_id
        if decoder.config.pad_token_id is None:
            logger.info('Setting pad_token_id to eos_token_id for decoder')
            decoder.config.pad_token_id = decoder.config.eos_token_id
        if decoder.generation_config.pad_token_id is None:
            logger.info('Setting pad_token_id to eos_token_id for decoder.generation_config')
            decoder.generation_config.pad_token_id = decoder.config.eos_token_id
        
        # create Projector (MLP)
        self.audio_projector = nn.Linear(projector_input_dim, decoder.config.hidden_size)
        
        # configure LoRA for training stage
        if training_stage == "stage2":
            peft_config = LoraConfig(
                target_modules=config.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
            )
            self.decoder = get_peft_model(decoder, peft_config)
            logger.info("Stage 2: LLM with LoRA")
            self.decoder.print_trainable_parameters()
        else:
            self.decoder = decoder
            logger.info("Stage 1: LLM frozen (no LoRA)")
        
        # set parameters training status
        self._setup_training_params(training_stage)
        
    def _setup_training_params(self, training_stage: TrainingStage):
        """
        set parameters training status
        
        Stage 1: only train Bridge(Adapter/QFormer) and Projector
        Stage 2: train Bridge, Projector and LLM LoRA parameters
        """
        # Encoder
        if training_stage == "stage1":
            train_encoder = self.config.train_encoder_stage1
        else:
            train_encoder = self.config.train_encoder_stage2
    
        if train_encoder:
            # unfreeze
            self.audio_encoder.train()
            for param in self.audio_encoder.parameters():
                param.requires_grad = True
            encoder_status = "train"
        else:
            # freeze
            self.audio_encoder.eval()
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            encoder_status = "freeze"
            
            # if use builtin Bridge, unfreeze builtin adapter/qformer
            if self.use_encoder_builtin_bridge:
                if hasattr(self.audio_encoder, 'audio_adapter') and self.audio_encoder.audio_adapter is not None:
                    self.audio_encoder.audio_adapter.train()
                    for param in self.audio_encoder.audio_adapter.parameters():
                        param.requires_grad = True
                    logger.info("[Bridge] unfreeze Adapter parameters")
                if hasattr(self.audio_encoder, 'audio_qformer') and self.audio_encoder.audio_qformer is not None:
                    self.audio_encoder.audio_qformer.train()
                    for param in self.audio_encoder.audio_qformer.parameters():
                        param.requires_grad = True
                    logger.info("[Bridge] unfreeze QFormer parameters")
                if hasattr(self.audio_encoder, 'fusion_module') and self.audio_encoder.fusion_module is not None:
                    self.audio_encoder.fusion_module.train()
                    for param in self.audio_encoder.fusion_module.parameters():
                        param.requires_grad = True
                    logger.info("[Bridge] unfreeze FusionModule parameters")
        
        # Projector train
        for param in self.audio_projector.parameters():
            param.requires_grad = True
        
        # log
        bridge_names = {
            "adapter": "Adapter",
            "qformer": "QFormer",
            "fusion": "FusionModule",
            "none": "no Bridge"
        }
        bridge_name = bridge_names.get(self.bridge_type, "no Bridge")
        bridge_status = "train" if self.use_encoder_builtin_bridge else "freeze"
        
        if training_stage == "stage1":
            # Stage 1: frozen LLM
            for param in self.decoder.parameters():
                param.requires_grad = False
            logger.info(f"Stage 1: Encoder{encoder_status}, {bridge_name}{bridge_status}")
        else:
            # Stage 2: LLM LoRA train
            logger.info(f"Stage 2: Encoder{encoder_status}, {bridge_name}{bridge_status}")
        
        self._print_trainable_params()
        
    def _print_trainable_params(self):
        """print trainable parameters"""
        trainable_params = 0
        all_params = 0
        
        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        logger.info(
            f"trainable parameters: {trainable_params:,} / {all_params:,} "
            f"({100 * trainable_params / all_params:.2f}%)"
        )
        
    def switch_to_stage2(self):
        """
        switch to stage 2
        reload decoder and add LoRA
        """
        if self.training_stage == "stage2":
            logger.warning("Already in stage 2")
            return
            
        # reload decoder and add LoRA
        decoder = AutoModelForCausalLM.from_pretrained(self.config.decoder_type)
        
        if decoder.config.pad_token_id is None:
            decoder.config.pad_token_id = decoder.config.eos_token_id
        if decoder.generation_config.pad_token_id is None:
            decoder.generation_config.pad_token_id = decoder.config.eos_token_id
            
        peft_config = LoraConfig(
            target_modules=self.config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
        )
        self.decoder = get_peft_model(decoder, peft_config)
        self.training_stage = "stage2"
        
        self._setup_training_params("stage2")
        
    def merge_and_unload(self):
        """merge LoRA weights and unload"""
        if hasattr(self.decoder, 'merge_and_unload'):
            self.decoder = self.decoder.merge_and_unload()
            
    @property
    def all_tied_weights_keys(self):
        """
        return tied weights keys (for HuggingFace transformers from_pretrained)
        
        return empty dictionary, because:
        1. our model has no tied weights at top level
        2. tied weights inside decoder are managed by decoder itself
        3. if return decoder's tied weights, HuggingFace will try to access self.lm_head,
           but the actual path is self.decoder.lm_head, will cause AttributeError
        """
        return {}
    
    def mark_tied_weights_as_initialized(self):
        """
        override this method to avoid tied weights access error
        
        our model structure is self.decoder.lm_head, not self.lm_head,
        so should not handle tied weights at top level.
        
        instead, let decoder handle its own tied weights.
        """
        # if decoder has mark_tied_weights_as_initialized method Invocation
        if hasattr(self, 'decoder') and self.decoder is not None:
            if hasattr(self.decoder, 'mark_tied_weights_as_initialized'):
                try:
                    self.decoder.mark_tied_weights_as_initialized()
                except Exception as e:
                    logger.warning(f"Failed to mark decoder tied weights: {e}")
        
        pass
    
    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration as e:
            logger.error("Rerun the script with 'accelerate launch -m xares_llm.run_encoder_tuning'")
            raise e
    
    def _call_encoder(self, audio, audio_attention_mask, task_id=None):
        """wrap encoder call, handle different forward signatures"""
        import inspect
        if hasattr(self.audio_encoder, 'forward'):
            sig = inspect.signature(self.audio_encoder.forward)
            if 'task_id' in sig.parameters and 'query_budget' in sig.parameters:
                return self.audio_encoder(
                    audio, audio_attention_mask, task_id=task_id, query_budget="auto"
                )
        return self.audio_encoder(audio, audio_attention_mask)
            
    def _prepare_multimodal_inputs(self, audio, audio_attention_mask, input_ids, attention_mask, labels=None, task_id=None):
        audio = audio.to(self.device)
        audio_attention_mask = audio_attention_mask.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if task_id is not None:
            task_id = task_id.to(self.device)
            
        final_audio_attention_mask = None
        
        encoder_requires_grad = any(p.requires_grad for p in self.audio_encoder.parameters())
        
        # Encoder forward
        if encoder_requires_grad:
            audio_feature, final_audio_attention_mask = self._call_encoder(
                audio, audio_attention_mask, task_id
            )
        else:
            # frozen
            with torch.no_grad():
                audio_feature, final_audio_attention_mask = self._call_encoder(
                    audio, audio_attention_mask, task_id
                )
        
        audio_feature = audio_feature.to(self.device)
        
        audio_feature = self.audio_projector(audio_feature)
        
        if final_audio_attention_mask is None:
            final_audio_attention_mask = torch.ones(*audio_feature.shape[:2], device=attention_mask.device)
        else:
            final_audio_attention_mask = final_audio_attention_mask.to(attention_mask.device)
            
        # text embedding
        input_embeds = self.decoder.get_input_embeddings()(input_ids)
        
        # ensure audio_feature and input_embeds have the same data type
        audio_feature = audio_feature.to(dtype=input_embeds.dtype)
        
        # concat: [AUDIO, TEXT]
        input_embeds = torch.cat((audio_feature, input_embeds), dim=1)
        
        # audio labels = -100（no loss）
        zero_audio_targets = torch.full(
            audio_feature.shape[:2], device=audio_feature.device, dtype=torch.long, fill_value=-100
        )
        
        if labels is not None:
            labels = torch.cat((zero_audio_targets, labels), dim=1)
            
        attention_mask = torch.cat(
            (final_audio_attention_mask, attention_mask),
            dim=1,
        )
        
        return input_embeds, attention_mask, labels
        
    def forward(self, audio, audio_attention_mask, input_ids, attention_mask, labels, task_id=None, **kwargs):
        """
        Next Token Prediction loss
        only calculate loss on answer part (prompt part is masked as -100 in data processing)
        """
        input_embeds, attention_mask, labels = self._prepare_multimodal_inputs(
            audio, audio_attention_mask, input_ids, attention_mask, labels, task_id=task_id
        )
        return self.decoder(input_ids=None, inputs_embeds=input_embeds, labels=labels, attention_mask=attention_mask)
    
    @torch.no_grad()
    def generate(self, audio, audio_attention_mask, input_ids, attention_mask, task_id=None, **gen_kwargs):
        """generate text"""
        input_embeds, attention_mask, _ = self._prepare_multimodal_inputs(
            audio, audio_attention_mask, input_ids, attention_mask, labels=None, task_id=task_id
        )
        return self.decoder.generate(
            input_ids=None, inputs_embeds=input_embeds, attention_mask=attention_mask, **gen_kwargs
        )
    
    def save_pretrained(self, save_directory, **kwargs):
        """save model"""
        super().save_pretrained(save_directory, **kwargs)
        # save training_stage information
        import json
        stage_info = {"training_stage": self.training_stage}
        with open(Path(save_directory) / "training_stage.json", "w") as f:
            json.dump(stage_info, f)
            
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """load model"""
        # read training_stage information
        import json
        stage_file = Path(pretrained_model_name_or_path) / "training_stage.json"
        if stage_file.exists():
            with open(stage_file, "r") as f:
                stage_info = json.load(f)
                kwargs["training_stage"] = stage_info.get("training_stage", "stage1")
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

