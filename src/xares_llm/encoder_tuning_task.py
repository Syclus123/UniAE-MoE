"""
Two stage Instruction Tuning
Multi-task Instruction Tuning, convert 12 downstream tasks to instruction-response text sequence generation tasks
"""

from __future__ import annotations

import torch
from pathlib import Path
from transformers import AutoTokenizer, TrainingArguments
import pandas as pd
import yaml
from dataclasses import dataclass, field, asdict
from loguru import logger
from typing import Any, Dict, List, Literal
import pprint
import importlib

from xares_llm.utils import seed_everything, setup_global_logger
from xares_llm.audiowebdataset import AudioTextDataType, AudioTextTokenWebdataset
from xares_llm.trainer import XaresLLMTrainerEvaluator
from xares_llm.modeling_encoder_tuning import EncoderTuningModel, EncoderTuningModelConfig
from xares_llm.metrics import get_metric, RegisteredMetricsLiteral, TokenDecoder


# training config mapping
AVAILABLE_ENCODER_TUNING_TRAINING_CONFIGS = {
    "all": next(importlib.resources.files("xares_llm.tasks.all.train").iterdir()),
    "task1": next(importlib.resources.files("xares_llm.tasks.task1.train").iterdir()),
    "task2": next(importlib.resources.files("xares_llm.tasks.task2.train").iterdir()),
    # Instruction Tuning config（no prob，full）
    "instruct_task2": importlib.resources.files("xares_llm.tasks.instruct_tuning") / "train_task2_config.yaml",
    "instruct_all": importlib.resources.files("xares_llm.tasks.instruct_tuning") / "all_train_config.yaml",
} | {
    str(Path(_).stem).replace("_config", ""): _
    for _ in importlib.resources.files("xares_llm.tasks.single.train").iterdir()
}

AVAILABLE_ENCODER_TUNING_EVALUATION_CONFIGS = {
    "all": next(importlib.resources.files("xares_llm.tasks.all.eval").iterdir()),
    "task1": next(importlib.resources.files("xares_llm.tasks.task1.eval").iterdir()),
    "task2": next(importlib.resources.files("xares_llm.tasks.task2.eval").iterdir()),
} | {
    str(Path(_).stem).replace("_test_config", ""): _
    for _ in importlib.resources.files("xares_llm.tasks.single.eval").iterdir()
}


@dataclass
class EncoderTuningTrainConfig:
    """Encoder Tuning Training Configuration"""
    
    audio_encoder_module_path: str
    audio_encoder_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())
    output_dir: str = "experiments_encoder_tuning/"
    config_name: str = "default"
    
    # training stage configuration
    training_stage: Literal["stage1", "stage2"] = "stage1"
    stage1_checkpoint: str | None = None  
    stage1_output_suffix: str = "_stage1"  # Output directory suffix for Stage 1
    stage2_output_suffix: str = "_stage2"  # Output directory suffix for Stage 2
    
    # general configuration
    torch_num_threads: int = 1
    seed: int = 42
    train_data: List[AudioTextDataType] | None = None
    
    # Decoder model
    decoder_model_name: str = "/private/models/SmolLM2-135M"
    
    # Adapter configuration
    adapter_type: str = "adapter"  # "adapter" or "qformer"
    adapter_hidden_dim: int | None = None  # None:encoder_output_dim * 4
    adapter_dropout: float = 0.1
    
    # QFormer configuration
    qformer_d_model: int = 1024
    qformer_num_global_queries: int = 32
    qformer_num_local_queries: int = 96
    qformer_layers: int = 6
    qformer_heads: int = 16
    qformer_pool_stride: int = 4
    qformer_dropout: float = 0.1
    qformer_drop_path: float = 0.05
    num_tasks: int = 12
    task_cond: bool = True
    cls_task_ids: list[int] | None = None
    
    # Encoder training control
    train_encoder_stage1: bool = False
    train_encoder_stage2: bool = False
    
    # LoRA（Stage 2）
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: str = "all-linear"
    
    # dataloder
    crop_audio_length: float = 30
    save_total_limit: int | None = 1
    save_steps: float = 200
    warmup_steps: int = 200
    max_steps: int = 10000
    per_device_train_batch_size: int = 4
    
    # optimizer configuration
    optimizer: str = "adamw_torch"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_dir: str = "tensorboard_logs"
    logging_steps: int = 100
    num_training_workers: int = 4
    sort_by_length: int = 128
    
    # log
    report_to: str = "tensorboard"  # "tensorboard", "wandb", "none"
    log_level: str = "info"
    save_safetensors: bool = True
    
    def __post_init__(self):
        if isinstance(self.train_data, dict):
            self.train_data = [AudioTextDataType(name=k, **val) for k, val in self.train_data.items()]
        torch.set_num_threads(self.torch_num_threads)
        setup_global_logger()
        seed_everything(self.seed)
        
    def __repr__(self):
        return pprint.pformat(asdict(self))
    
    @classmethod
    def from_file(
        cls,
        config_file: str,
        encoder_path: str,
        model_kwargs: Dict[str, Any] | None = None,
        overwrite_kwargs: Dict[str, Any] | None = None,
    ) -> EncoderTuningTrainConfig:
        with open(config_file) as con_read:
            yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
        yaml_config["config_name"] = Path(config_file).stem
        yaml_config["audio_encoder_module_path"] = encoder_path
        
        # encoder kwargs
        yaml_to_encoder_mapping = {
            "adapter_type": "adapter_type",
            "adapter_hidden_dim": "adapter_hidden_dim",
            "adapter_dropout": "adapter_dropout",
            "qformer_d_model": "qformer_d_model",
            "qformer_num_global_queries": "qformer_num_global_queries",
            "qformer_num_local_queries": "qformer_num_local_queries",
            "qformer_layers": "qformer_layers",
            "qformer_heads": "qformer_heads",
            "qformer_pool_stride": "qformer_pool_stride",
            "qformer_dropout": "qformer_dropout",
            "qformer_drop_path": "qformer_drop_path",
            "num_tasks": "num_tasks",
            "task_cond": "task_cond",
            "cls_task_ids": "cls_task_ids",
        }
        
        if overwrite_kwargs is None:
            overwrite_kwargs = dict()
        yaml_config.update(overwrite_kwargs)
        
        encoder_kwargs_from_yaml = {}
        for yaml_key, encoder_key in yaml_to_encoder_mapping.items():
            if yaml_key in yaml_config and yaml_config[yaml_key] is not None:
                encoder_kwargs_from_yaml[encoder_key] = yaml_config[yaml_key]
        
        if model_kwargs is None:
            model_kwargs = dict()
        final_encoder_kwargs = {**encoder_kwargs_from_yaml, **model_kwargs}
        yaml_config["audio_encoder_kwargs"] = final_encoder_kwargs
        return cls(**yaml_config)
    
    @classmethod
    def from_file_or_key(
        cls,
        config_identifier: str,
        encoder_path: str,
        model_kwargs: Dict[str, Any] | None = None,
        overwrite_kwargs: Dict[str, Any] | None = None,
    ) -> EncoderTuningTrainConfig:
        if config_identifier in AVAILABLE_ENCODER_TUNING_TRAINING_CONFIGS:
            return cls.from_file(
                AVAILABLE_ENCODER_TUNING_TRAINING_CONFIGS[config_identifier],
                encoder_path=encoder_path,
                model_kwargs=model_kwargs,
                overwrite_kwargs=overwrite_kwargs,
            )
        path_obj = Path(config_identifier)
        if path_obj.is_file():
            return cls.from_file(
                config_identifier,
                encoder_path=encoder_path,
                model_kwargs=model_kwargs,
                overwrite_kwargs=overwrite_kwargs,
            )
        raise ValueError(f"Unknown config identifier {config_identifier}")


@dataclass
class EncoderTuningEvaluationConfig:
    """Encoder Tuning Evaluation Configuration"""
    
    data: AudioTextDataType
    metric: RegisteredMetricsLiteral
    metric_args: Dict[str, Any] = field(default_factory=lambda: dict())
    batch_size: int = 32
    num_workers: int = 0
    weight: float = 1
    
    @classmethod
    def configs_from_file(cls, yaml_config_file: str) -> List[EncoderTuningEvaluationConfig]:
        with open(yaml_config_file) as con_read:
            yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
        evaluation_configs = []
        for k, values in yaml_config.items():
            data_kwargs = values.pop("data")
            metric = values.pop("metric")
            evaluation_configs.append(cls(data=AudioTextDataType(name=k, **data_kwargs), metric=metric, **values))
        return evaluation_configs
    
    def __repr__(self):
        return pprint.pformat(asdict(self))
    
    @classmethod
    def configs_from_file_or_key(cls, config_identifier: str) -> List[EncoderTuningEvaluationConfig]:
        if config_identifier in AVAILABLE_ENCODER_TUNING_EVALUATION_CONFIGS:
            return cls.configs_from_file(AVAILABLE_ENCODER_TUNING_EVALUATION_CONFIGS[config_identifier])
        path_obj = Path(config_identifier)
        if path_obj.is_file():
            return cls.configs_from_file(config_identifier)
        raise ValueError(f"Unknown config identifier {config_identifier}")


class EncoderTuningTask:
    """
    Two-stage Encoder Tuning Training Task
    
    Stage 1: Frozen Encoder and LLM, Finetune Adapter/MLP (Task 2)
    Stage 2: Frozen Encoder, Finetune Adapter/MLP/LLM（Task 1+Task 2）
    """
    
    def __init__(self, train_config: EncoderTuningTrainConfig):
        self.train_config = train_config
        
        if Path(self.train_config.audio_encoder_module_path).is_file():
            model_name = str(Path(self.train_config.audio_encoder_module_path).stem)
        else:
            model_name = self.train_config.audio_encoder_module_path.split(".")[-1]
        
        # Use configured output suffix
        if train_config.training_stage == "stage1":
            stage_suffix = train_config.stage1_output_suffix
        else:
            stage_suffix = train_config.stage2_output_suffix
        
        self.output_dir = Path(train_config.output_dir) / train_config.config_name / (model_name + stage_suffix)
        
        logger.add(
            self.output_dir / "log.txt",
            enqueue=True,
            level="INFO",
            format="[{level} {time:YYYY-MM-DD HH:mm:ss}] {message}",
        )
        logger.info(f"output: {self.output_dir}")
        logger.info(f"train: {train_config.training_stage}")
        logger.info(f"load tokenizer: {train_config.decoder_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(train_config.decoder_model_name)
        
        # set pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"pad_token EOS token: {self.tokenizer.pad_token}")
        
        # training arguments
        tensorboard_log_dir = Path(self.output_dir) / self.train_config.logging_dir
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        
        # training_args = TrainingArguments(
        #     output_dir=str(self.output_dir),
        #     learning_rate=self.train_config.learning_rate,
        #     per_device_train_batch_size=self.train_config.
        #     per_device_train_batch_size,
        #     save_total_limit=self.train_config.save_total_limit,
        #     save_steps=self.train_config.save_steps,
        #     warmup_steps=self.train_config.warmup_steps,
        #     max_grad_norm=self.train_config.max_grad_norm,
        #     max_steps=self.train_config.max_steps,
        #     optim=self.train_config.optimizer,
        #     weight_decay=self.train_config.weight_decay,
        #     seed=self.train_config.seed,
        #     logging_steps=self.train_config.logging_steps,
        #     logging_dir=str(tensorboard_log_dir),
        #     # TensorBoard
        #     report_to=self.train_config.report_to,
        #     log_level=self.train_config.log_level,
        #     save_safetensors=self.train_config.save_safetensors,
        #     logging_first_step=True,
        #     logging_nan_inf_filter=True,
        # )
        
        training_args_dict = {
            "output_dir": str(self.output_dir),
            "learning_rate": self.train_config.learning_rate,
            "per_device_train_batch_size": self.train_config.per_device_train_batch_size,
            "save_total_limit": self.train_config.save_total_limit,
            "save_steps": self.train_config.save_steps,
            "warmup_steps": self.train_config.warmup_steps,
            "max_grad_norm": self.train_config.max_grad_norm,
            "max_steps": self.train_config.max_steps,
            "optim": self.train_config.optimizer,
            "weight_decay": self.train_config.weight_decay,
            "seed": self.train_config.seed,
            "logging_steps": self.train_config.logging_steps,
            "logging_dir": str(tensorboard_log_dir),
            "report_to": self.train_config.report_to,
            "log_level": self.train_config.log_level,
            "logging_first_step": True,
            "logging_nan_inf_filter": True,
        }
        
        # save_safetensors
        try:
            import inspect
            if 'save_safetensors' in inspect.signature(TrainingArguments.__init__).parameters:
                training_args_dict["save_safetensors"] = self.train_config.save_safetensors
        except Exception:
            pass
        
        training_args = TrainingArguments(**training_args_dict)
        
        logger.info(f"TensorBoard: {tensorboard_log_dir}")
        
        logger.info("=" * 60)
        logger.info("Encoder:")
        logger.info(f"  Encoder module: {self.train_config.audio_encoder_module_path}")
        logger.info(f"  Bridge type: {self.train_config.adapter_type}")
        logger.info(f"  Adapter: adapter_type='{self.train_config.adapter_type}'")
        
        if self.train_config.audio_encoder_kwargs:
            logger.info("audio_encoder_kwargs):")
            for key, value in self.train_config.audio_encoder_kwargs.items():
                logger.info(f"    - {key}: {value}")
        else:
            logger.warning(" audio_encoder_kwargs 为空! Encoder 将使用默认参数")
        
        # adapter_type
        encoder_adapter_type = self.train_config.audio_encoder_kwargs.get("adapter_type") if self.train_config.audio_encoder_kwargs else None
        if encoder_adapter_type:
            logger.info(f"adapter_type='{encoder_adapter_type}'")
        else:
            logger.warning(f"The adapter_type was not passed to the Encoder!")
        logger.info("=" * 60)
        
        def model_init_function():
            return EncoderTuningModel(
                config=EncoderTuningModelConfig(
                    decoder_type=self.train_config.decoder_model_name,
                    audio_encoder_name=self.train_config.audio_encoder_module_path,
                    audio_encoder_params=self.train_config.audio_encoder_kwargs,
                    # Adapter
                    # adapter_type=self.train_config.adapter_type,
                    # adapter_hidden_dim=self.train_config.adapter_hidden_dim,
                    # adapter_dropout=self.train_config.adapter_dropout,
                    # # QFormer
                    # qformer_d_model=self.train_config.qformer_d_model,
                    # qformer_num_global_queries=self.train_config.qformer_num_global_queries,
                    # qformer_num_local_queries=self.train_config.qformer_num_local_queries,
                    # qformer_layers=self.train_config.qformer_layers,
                    # qformer_heads=self.train_config.qformer_heads,
                    # qformer_pool_stride=self.train_config.qformer_pool_stride,
                    # qformer_dropout=self.train_config.qformer_dropout,
                    # qformer_drop_path=self.train_config.qformer_drop_path,
                    # num_tasks=self.train_config.num_tasks,
                    # task_cond=self.train_config.task_cond,
                    # cls_task_ids=self.train_config.cls_task_ids,
                    # Encoder training
                    train_encoder_stage1=self.train_config.train_encoder_stage1,
                    train_encoder_stage2=self.train_config.train_encoder_stage2,
                    # LoRA
                    lora_r=self.train_config.lora_r,
                    lora_alpha=self.train_config.lora_alpha,
                    lora_dropout=self.train_config.lora_dropout,
                    lora_target_modules=self.train_config.lora_target_modules,
                ),
                training_stage=self.train_config.training_stage,
            )
        
        self.model = None
        
        # check checkpoint
        checkpoint_dirs = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if checkpoint_dirs:
            logger.info(f"load checkpoint, from{checkpoint_dirs[0]}")
            self.model = EncoderTuningModel.from_pretrained(
                checkpoint_dirs[0],
                training_stage=self.train_config.training_stage
            )
        elif self.train_config.stage1_checkpoint and self.train_config.training_stage == "stage2":
            # load Stage 1 checkpoint for Stage 2
            logger.info(f"Stage 2: from Stage 1 checkpoint: {self.train_config.stage1_checkpoint}")
            self.model = self._load_stage1_for_stage2(self.train_config.stage1_checkpoint)
            
        self.trainer = XaresLLMTrainerEvaluator(model=None, model_init=model_init_function, args=training_args)
        
    def _load_stage1_for_stage2(self, stage1_checkpoint: str) -> EncoderTuningModel:
        """
        Load Stage 1 checkpoint for Stage 2 training
        """
        logger.info(f"Load Stage 1: {stage1_checkpoint}")
        
        from safetensors.torch import load_file
        import json
        
        checkpoint_path = Path(stage1_checkpoint)
        
        config_path = checkpoint_path / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        config = EncoderTuningModelConfig(**config_dict)
        
        logger.info("Create Stage 2 Model...")
        stage2_model = EncoderTuningModel(
            config=config,
            training_stage="stage2"
        )
        
        logger.info("Load Stage 1 ...")
        safetensors_path = checkpoint_path / "model.safetensors"
        pytorch_path = checkpoint_path / "pytorch_model.bin"
        safetensors_index_path = checkpoint_path / "model.safetensors.index.json"
        
        state_dict = {}
        if safetensors_path.exists():
            state_dict = load_file(str(safetensors_path))
            # logger.info("model.safetensors")
        elif safetensors_index_path.exists():
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            shard_files = set(index['weight_map'].values())
            for shard_file in sorted(shard_files):
                shard_path = checkpoint_path / shard_file
                if shard_path.exists():
                    shard_state = load_file(str(shard_path))
                    state_dict.update(shard_state)
            logger.info(f"{len(shard_files)} safetensors load")
        elif pytorch_path.exists():
            state_dict = torch.load(pytorch_path, map_location="cpu")
            logger.info("from pytorch_model.bin load")
        else:
            logger.warning(f"no checkpoints: {checkpoint_path}")
        
        # Projector
        projector_state = {}
        for k, v in state_dict.items():
            if k.startswith("audio_projector."):
                projector_state[k.replace("audio_projector.", "")] = v
        
        if projector_state:
            stage2_model.audio_projector.load_state_dict(projector_state)
            logger.info(f"Load Projector checkpoints({len(projector_state)} parameters)")
        
        # Load Bridge checkpoints（Adapter/QFormer/FusionModule）
        if stage2_model.use_encoder_builtin_bridge:
            adapter_state = {}
            for k, v in state_dict.items():
                if "audio_encoder.audio_adapter." in k:
                    adapter_state[k.replace("audio_encoder.audio_adapter.", "")] = v
            
            if adapter_state and hasattr(stage2_model.audio_encoder, 'audio_adapter') and stage2_model.audio_encoder.audio_adapter is not None:
                stage2_model.audio_encoder.audio_adapter.load_state_dict(adapter_state)
                logger.info(f"Adapter({len(adapter_state)} parameters)")
            
            # QFormer
            qformer_state = {}
            for k, v in state_dict.items():
                if "audio_encoder.audio_qformer." in k:
                    qformer_state[k.replace("audio_encoder.audio_qformer.", "")] = v
            
            if qformer_state and hasattr(stage2_model.audio_encoder, 'audio_qformer') and stage2_model.audio_encoder.audio_qformer is not None:
                stage2_model.audio_encoder.audio_qformer.load_state_dict(qformer_state)
                logger.info(f"QFormer({len(qformer_state)} parameters)")
            
            # FusionModule
            fusion_state = {}
            for k, v in state_dict.items():
                if "audio_encoder.fusion_module." in k:
                    fusion_state[k.replace("audio_encoder.fusion_module.", "")] = v
            
            if fusion_state and hasattr(stage2_model.audio_encoder, 'fusion_module') and stage2_model.audio_encoder.fusion_module is not None:
                stage2_model.audio_encoder.fusion_module.load_state_dict(fusion_state)
                logger.info(f"FusionModule({len(fusion_state)} parameters)")
        else:
            logger.info("Encoder w/o Bridge")
        
        del state_dict
        torch.cuda.empty_cache()
        
        return stage2_model
        
    def train(self) -> EncoderTuningModel:
        checkpoint_dirs = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if self.model is not None and checkpoint_dirs:
            self.trainer.model = self.model
            logger.info(f"checkpoint {checkpoint_dirs[0].name}, skip the training")
            return self.model
        
        if self.model is not None:
            self.trainer.model = self.model
            logger.info("From Stage 1 load checkpoints, training")
        
        train_data_object = AudioTextTokenWebdataset(
            data_urls=self.train_config.train_data,
            tokenizer=self.tokenizer,
            training=True,
            batch_size=self.train_config.per_device_train_batch_size,
            resample=True,
            sort_by_length=self.train_config.sort_by_length,
            num_workers=self.train_config.num_training_workers,
            crop_audio_length=self.train_config.crop_audio_length,
        )
        
        self.trainer.train_data_object = train_data_object
        self.trainer.train()
        logger.info(f"Training completed: {self.output_dir}")
        return self.trainer.model
    
    def evaluate(
        self,
        eval_config: EncoderTuningEvaluationConfig,
        trained_model: EncoderTuningModel | None = None,
        chpt_path: str | Path | None = None,
    ) -> tuple[Dict[RegisteredMetricsLiteral, float], pd.DataFrame]:
        """eval"""
        if trained_model is not None:
            model = trained_model
        elif chpt_path is not None:
            logger.info(f"load {chpt_path}")
            model = EncoderTuningModel.from_pretrained(chpt_path)
        else:
            model = self.trainer.model
            
        self.trainer.model = model
        
        metrics_compute_function = get_metric(eval_config.metric, tokenizer=self.tokenizer, **eval_config.metric_args)
        
        data_object_eval = AudioTextTokenWebdataset(
            data_urls=eval_config.data,
            tokenizer=self.tokenizer,
            training=False,
            batch_size=eval_config.batch_size,
            sort_by_length=256,
            num_workers=eval_config.num_workers,
        )
        
        self.trainer.compute_metrics = metrics_compute_function
        
        logger.info("evaluate...")
        result = self.trainer.predict(test_dataset=data_object_eval)
        
        decoder = TokenDecoder(self.tokenizer)
        predicted_text, targets = decoder.decode_predictions(result)
        
        prediction_df = pd.DataFrame({"predict": predicted_text, "labels": targets})
        return result.metrics[f"test_{eval_config.metric}"], prediction_df
    
    def run(self, eval_configs: List[EncoderTuningEvaluationConfig]) -> List[Dict[str, Any]]:
        """trian/eval"""
        if not isinstance(eval_configs, list):
            eval_configs = [eval_configs]
            
        result = []
        model = self.train()
        
        for eval_config in eval_configs:
            dataset_name = eval_config.data.name
            score_file_cache = self.output_dir / f'score_{dataset_name}.yaml'
            
            if score_file_cache.exists() and score_file_cache.stat().st_size > 0:
                with open(score_file_cache, 'r') as rp:
                    score = yaml.load(rp, Loader=yaml.SafeLoader)['score']
                logger.debug(f"Find the cached result [{score_file_cache}] {dataset_name}: skip eval, Score: {score:.2f}")
            else:
                score, output_df = self.evaluate(trained_model=model, eval_config=eval_config)
                logger.info(f"{dataset_name}: [{eval_config.metric}]: {score:.2f}")
                output_df.to_csv(self.output_dir / f"predictions_{dataset_name}.csv", index=False)
                
                with open(score_file_cache, 'w') as wp:
                    yaml.dump({'score': score}, wp, default_flow_style=False)
                    
            result.append({"Task": dataset_name, "score": score, "weight": eval_config.weight})
            logger.debug(f"model save {self.output_dir / f'predictions_{dataset_name}.csv'}")
            
        return result


class TwoStageEncoderTuningTask:
    """
    A complete two-stage Encoder fine-tuning training process
    """
    
    def __init__(
        self,
        encoder_path: str,
        model_kwargs: Dict[str, Any] | None = None,
        output_dir: str = "experiments_encoder_tuning/",
        decoder_model_name: str = "/private/models/SmolLM2-135M",
        # data config
        stage1_train_config: str = "task2",  # task2
        stage2_train_config: str = "all",    # all
        # Stage configuration
        stage1_checkpoint: str | None = None,  # Stage 1 checkpoint for Stage 2
        stage1_output_suffix: str = "_stage1",
        stage2_output_suffix: str = "_stage2",
        # Stage 1
        stage1_max_steps: int = 50000,
        stage1_learning_rate: float = 1e-4,
        # Stage 2
        stage2_max_steps: int = 100000,
        stage2_learning_rate: float = 1e-5,
        # General
        per_device_train_batch_size: int = 4,
        save_steps: float = 500,
        warmup_steps: int = 200,
        seed: int = 42,
        # Adapter
        adapter_type: str = "adapter",
        adapter_hidden_dim: int | None = None,
        adapter_dropout: float = 0.1,
        # QFormer
        qformer_d_model: int = 1024,
        qformer_num_global_queries: int = 32,
        qformer_num_local_queries: int = 96,
        qformer_layers: int = 6,
        qformer_heads: int = 16,
        qformer_pool_stride: int = 4,
        qformer_dropout: float = 0.1,
        qformer_drop_path: float = 0.05,
        num_tasks: int = 12,
        task_cond: bool = True,
        cls_task_ids: list[int] | None = None,
        # Encoder training control
        train_encoder_stage1: bool = False,
        train_encoder_stage2: bool = False,
        # LoRA
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        # log
        report_to: str = "tensorboard",
        logging_steps: int = 100,
    ):
        self.encoder_path = encoder_path
        self.model_kwargs = model_kwargs or {}
        self.output_dir = output_dir
        self.decoder_model_name = decoder_model_name
        
        self.stage1_train_config = stage1_train_config
        self.stage2_train_config = stage2_train_config
    
        # Stage configuration
        self.stage1_checkpoint = stage1_checkpoint
        self.stage1_output_suffix = stage1_output_suffix
        self.stage2_output_suffix = stage2_output_suffix
        
        self.stage1_max_steps = stage1_max_steps
        self.stage1_learning_rate = stage1_learning_rate
        self.stage2_max_steps = stage2_max_steps
        self.stage2_learning_rate = stage2_learning_rate
        
        self.per_device_train_batch_size = per_device_train_batch_size
        self.save_steps = save_steps
        self.warmup_steps = warmup_steps
        self.seed = seed
        
        # Adapter config
        self.adapter_type = adapter_type
        self.adapter_hidden_dim = adapter_hidden_dim
        self.adapter_dropout = adapter_dropout
        
        # QFormer config
        self.qformer_d_model = qformer_d_model
        self.qformer_num_global_queries = qformer_num_global_queries
        self.qformer_num_local_queries = qformer_num_local_queries
        self.qformer_layers = qformer_layers
        self.qformer_heads = qformer_heads
        self.qformer_pool_stride = qformer_pool_stride
        self.qformer_dropout = qformer_dropout
        self.qformer_drop_path = qformer_drop_path
        self.num_tasks = num_tasks
        self.task_cond = task_cond
        self.cls_task_ids = cls_task_ids
        
        # Encoder training control
        self.train_encoder_stage1 = train_encoder_stage1
        self.train_encoder_stage2 = train_encoder_stage2
        
        # LoRA config
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        self.report_to = report_to
        self.logging_steps = logging_steps
        
    def run_stage1(self, eval_configs: List[EncoderTuningEvaluationConfig] | None = None) -> Path:
        """
        execute Stage 1 training: Adapter+MLP on Task 2 data
        
        return Stage 1 checkpoint path
        """
        logger.info("=" * 50)
        logger.info("Stage 1: ")
        logger.info("=" * 50)
        
        stage1_config = EncoderTuningTrainConfig.from_file_or_key(
            self.stage1_train_config,
            encoder_path=self.encoder_path,
            model_kwargs=self.model_kwargs,
            overwrite_kwargs={
                "output_dir": self.output_dir,
                "training_stage": "stage1",
                "stage1_output_suffix": self.stage1_output_suffix,
                "stage2_output_suffix": self.stage2_output_suffix,
                "decoder_model_name": self.decoder_model_name,
                "max_steps": self.stage1_max_steps,
                "learning_rate": self.stage1_learning_rate,
                "per_device_train_batch_size": self.per_device_train_batch_size,
                "save_steps": self.save_steps,
                "warmup_steps": self.warmup_steps,
                "seed": self.seed,
                # Adapter
                "adapter_type": self.adapter_type,
                "adapter_hidden_dim": self.adapter_hidden_dim,
                "adapter_dropout": self.adapter_dropout,
                # QFormer
                "qformer_d_model": self.qformer_d_model,
                "qformer_num_global_queries": self.qformer_num_global_queries,
                "qformer_num_local_queries": self.qformer_num_local_queries,
                "qformer_layers": self.qformer_layers,
                "qformer_heads": self.qformer_heads,
                "qformer_pool_stride": self.qformer_pool_stride,
                "qformer_dropout": self.qformer_dropout,
                "qformer_drop_path": self.qformer_drop_path,
                "num_tasks": self.num_tasks,
                "task_cond": self.task_cond,
                "cls_task_ids": self.cls_task_ids,
                # Encoder training
                "train_encoder_stage1": self.train_encoder_stage1,
                "train_encoder_stage2": self.train_encoder_stage2,
                # Logging
                "report_to": self.report_to,
                "logging_steps": self.logging_steps,
            }
        )
        
        stage1_task = EncoderTuningTask(stage1_config)
        
        if eval_configs:
            stage1_task.run(eval_configs)
        else:
            stage1_task.train()
            
        # return latest checkpoint path
        checkpoint_dirs = sorted(
            stage1_task.output_dir.glob("checkpoint-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not checkpoint_dirs:
            raise RuntimeError("Stage 1 complete ,but not find checkpoint")
            
        return checkpoint_dirs[0]
    
    def run_stage2(
        self,
        stage1_checkpoint: str | Path,
        eval_configs: List[EncoderTuningEvaluationConfig] | None = None
    ) -> Path:
        """
        execute Stage 2 training: Adapter+MLP+LLM(LoRA) on all tasks
        
        return Stage 2 checkpoint path
        """
        logger.info("=" * 50)
        logger.info("Stage 2:")
        logger.info("=" * 50)
        
        stage2_config = EncoderTuningTrainConfig.from_file_or_key(
            self.stage2_train_config,  # Data: Task 1 + Task 2
            encoder_path=self.encoder_path,
            model_kwargs=self.model_kwargs,
            overwrite_kwargs={
                "output_dir": self.output_dir,
                "training_stage": "stage2",
                "stage1_checkpoint": str(stage1_checkpoint),
                "stage1_output_suffix": self.stage1_output_suffix,
                "stage2_output_suffix": self.stage2_output_suffix,
                "decoder_model_name": self.decoder_model_name,
                "max_steps": self.stage2_max_steps,
                "learning_rate": self.stage2_learning_rate,
                "per_device_train_batch_size": self.per_device_train_batch_size,
                "save_steps": self.save_steps,
                "warmup_steps": self.warmup_steps,
                "seed": self.seed,
                # Adapter
                "adapter_type": self.adapter_type,
                "adapter_hidden_dim": self.adapter_hidden_dim,
                "adapter_dropout": self.adapter_dropout,
                # QFormer
                "qformer_d_model": self.qformer_d_model,
                "qformer_num_global_queries": self.qformer_num_global_queries,
                "qformer_num_local_queries": self.qformer_num_local_queries,
                "qformer_layers": self.qformer_layers,
                "qformer_heads": self.qformer_heads,
                "qformer_pool_stride": self.qformer_pool_stride,
                "qformer_dropout": self.qformer_dropout,
                "qformer_drop_path": self.qformer_drop_path,
                "num_tasks": self.num_tasks,
                "task_cond": self.task_cond,
                "cls_task_ids": self.cls_task_ids,
                # Encoder training
                "train_encoder_stage1": self.train_encoder_stage1,
                "train_encoder_stage2": self.train_encoder_stage2,
                # LoRA
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                # Logging
                "report_to": self.report_to,
                "logging_steps": self.logging_steps,
            }
        )
        
        stage2_task = EncoderTuningTask(stage2_config)
        
        if eval_configs:
            stage2_task.run(eval_configs)
        else:
            stage2_task.train()
            
        checkpoint_dirs = sorted(
            stage2_task.output_dir.glob("checkpoint-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not checkpoint_dirs:
            raise RuntimeError("Stage 2 complete ,but not find checkpoint")
        
            
        return checkpoint_dirs[0]
    
    def run_full_pipeline(
        self,
        eval_configs: List[EncoderTuningEvaluationConfig] | None = None
    ) -> Dict[str, Any]:
        """
        Carry out the complete two-stage training process
        """
        logger.info("Start the two-stage Encoder fine-tuning training")
        
        # Check if Stage 1 checkpoint is provided
        if self.stage1_checkpoint and Path(self.stage1_checkpoint).exists():
            logger.info(f"Use stage 1 checkpoint: {self.stage1_checkpoint}")
            stage1_checkpoint = Path(self.stage1_checkpoint)
        else:
            # Run Stage 1 training
            stage1_checkpoint = self.run_stage1(eval_configs=None)
            logger.info(f"Stage 1 complete, checkpoint: {stage1_checkpoint}")
        
        # Run Stage 2 training
        stage2_checkpoint = self.run_stage2(stage1_checkpoint, eval_configs=eval_configs)
        logger.info(f"Stage 2 complete, checkpoint: {stage2_checkpoint}")
        
        return {
            "stage1_checkpoint": str(stage1_checkpoint),
            "stage2_checkpoint": str(stage2_checkpoint),
        }

