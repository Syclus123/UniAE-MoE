"""
Multi-task Instruction Tuning Entry Script
"""

import argparse
import json
import os
from typing import Dict, Any, List

import yaml
import pandas as pd
from loguru import logger
from pathlib import Path

from xares_llm.encoder_tuning_task import (
    EncoderTuningTask,
    EncoderTuningTrainConfig,
    EncoderTuningEvaluationConfig,
    TwoStageEncoderTuningTask,
    AVAILABLE_ENCODER_TUNING_TRAINING_CONFIGS,
    AVAILABLE_ENCODER_TUNING_EVALUATION_CONFIGS,
)


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """load yaml"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logger.info(f"load yaml: {config_path}")
    return config or {}


def run_single_stage(args):
    """single stage"""
    overwrite_kwargs = args.args.copy() if args.args else {}
    
    if args.config:
        config_kwargs = load_config_from_yaml(args.config)
        overwrite_kwargs.update(config_kwargs)
    
    # set training stage
    overwrite_kwargs["training_stage"] = args.stage
    
    # command stage1_checkpoint > YAML
    if args.stage == "stage2":
        if args.stage1_checkpoint:
            overwrite_kwargs["stage1_checkpoint"] = args.stage1_checkpoint
        elif "stage1_checkpoint" not in overwrite_kwargs or not overwrite_kwargs.get("stage1_checkpoint"):
            logger.warning("Stage 2 requires Adapter & Projector weights (use --stage1_checkpoint or set in YAML config)")
    
    train_config = EncoderTuningTrainConfig.from_file_or_key(
        args.train_config, 
        encoder_path=args.encoder_path, 
        model_kwargs=args.model_args, 
        overwrite_kwargs=overwrite_kwargs
    )
    eval_configs = EncoderTuningEvaluationConfig.configs_from_file_or_key(args.eval_configs)
    
    if args.benchmark:
        logger.info("deterministic mode")
        import torch
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"training config:\n{train_config}\nevaluation config: {eval_configs}")
    runner = EncoderTuningTask(train_config)
    scores: List[Dict[str, Any]] = runner.run(eval_configs)
    
    return scores, runner.output_dir


def run_full_pipeline(args):
    """full two stage"""
    if args.benchmark:
        logger.info("deterministic mode")
        import torch
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    config_kwargs = {}
    if args.config:
        config_kwargs = load_config_from_yaml(args.config)
    
    overwrite_kwargs = args.args or {}
    config_kwargs.update(overwrite_kwargs)
    
    stage1_max_steps = config_kwargs.pop("stage1_max_steps", 50000)
    stage1_learning_rate = config_kwargs.pop("stage1_learning_rate", 1e-4)
    stage2_max_steps = config_kwargs.pop("stage2_max_steps", 100000)
    stage2_learning_rate = config_kwargs.pop("stage2_learning_rate", 1e-5)
    
    runner = TwoStageEncoderTuningTask(
        encoder_path=args.encoder_path,
        model_kwargs=args.model_args,
        stage1_train_config=config_kwargs.get("stage1_train_config", "task2"),
        stage2_train_config=config_kwargs.get("stage2_train_config", "all"),
        output_dir=config_kwargs.get("output_dir", "experiments_encoder_tuning/"),
        decoder_model_name=config_kwargs.get("decoder_model_name", "/private/models/SmolLM2-135M"),
        # Stage configuration
        stage1_checkpoint=config_kwargs.get("stage1_checkpoint", None),
        stage1_output_suffix=config_kwargs.get("stage1_output_suffix", "_stage1"),
        stage2_output_suffix=config_kwargs.get("stage2_output_suffix", "_stage2"),
        # Training steps and learning rates
        stage1_max_steps=stage1_max_steps,
        stage1_learning_rate=stage1_learning_rate,
        stage2_max_steps=stage2_max_steps,
        stage2_learning_rate=stage2_learning_rate,
        # General training config
        per_device_train_batch_size=config_kwargs.get("per_device_train_batch_size", 4),
        save_steps=config_kwargs.get("save_steps", 500),
        warmup_steps=config_kwargs.get("warmup_steps", 200),
        seed=config_kwargs.get("seed", 42),
        # Adapter config
        adapter_type=config_kwargs.get("adapter_type", "adapter"),
        adapter_hidden_dim=config_kwargs.get("adapter_hidden_dim", None),
        adapter_dropout=config_kwargs.get("adapter_dropout", 0.1),
        # QFormer config
        qformer_d_model=config_kwargs.get("qformer_d_model", 1024),
        qformer_num_global_queries=config_kwargs.get("qformer_num_global_queries", 32),
        qformer_num_local_queries=config_kwargs.get("qformer_num_local_queries", 96),
        qformer_layers=config_kwargs.get("qformer_layers", 6),
        qformer_heads=config_kwargs.get("qformer_heads", 16),
        qformer_pool_stride=config_kwargs.get("qformer_pool_stride", 4),
        qformer_dropout=config_kwargs.get("qformer_dropout", 0.1),
        qformer_drop_path=config_kwargs.get("qformer_drop_path", 0.05),
        num_tasks=config_kwargs.get("num_tasks", 12),
        task_cond=config_kwargs.get("task_cond", True),
        cls_task_ids=config_kwargs.get("cls_task_ids", None),
        # Encoder training control
        train_encoder_stage1=config_kwargs.get("train_encoder_stage1", False),
        train_encoder_stage2=config_kwargs.get("train_encoder_stage2", False),
        # LoRA config
        lora_r=config_kwargs.get("lora_r", 8),
        lora_alpha=config_kwargs.get("lora_alpha", 32),
        lora_dropout=config_kwargs.get("lora_dropout", 0.1),
        # Logging config
        report_to=config_kwargs.get("report_to", "tensorboard"),
        logging_steps=config_kwargs.get("logging_steps", 100),
    )
    
    eval_configs = EncoderTuningEvaluationConfig.configs_from_file_or_key(args.eval_configs)
    
    result = runner.run_full_pipeline(eval_configs=eval_configs)
    
    logger.info(f"Two-stage training complete:")
    logger.info(f"  Stage 1 checkpoint: {result['stage1_checkpoint']}")
    logger.info(f"  Stage 2 checkpoint: {result['stage2_checkpoint']}")
    return result


def main(args):
    if args.stage == "full":
        result = run_full_pipeline(args)
        return
    
    scores, output_dir = run_single_stage(args)
    
    logger.info("all task evaluation complete")
    
    df = pd.DataFrame(scores)
    df.sort_values(by="Task", inplace=True)
    
    new_row = pd.DataFrame([{
        "Task": "Overall",
        "score": (df["score"] * df["weight"]).sum() / df["weight"].sum(),
        "weight": df["weight"].sum(),
    }])
    df = pd.concat((df, new_row), ignore_index=True)
    logger.info(f"\nresults:\n{df.to_string(index=False, float_format='%.3f')}")
    df.to_csv(Path(output_dir) / "scores.tsv", sep="\t", index=False, float_format='%.3f')
    logger.info(f"\nfile saved: {Path(output_dir) / 'scores.tsv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task Instruction Tuning")
    
    parser.add_argument(
        "encoder_path",
        type=str,
        help="",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="config/Instruction_tuning_config.yaml",
        default=None,
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["stage1", "stage2", "full"],
        default="full",
        help="",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        help=f"datasets: {list(AVAILABLE_ENCODER_TUNING_TRAINING_CONFIGS.keys())}",
        nargs="?",
        default="all",
    )
    parser.add_argument(
        "--eval_configs",
        type=str,
        nargs="?",
        help=f"eval_datasets:{list(AVAILABLE_ENCODER_TUNING_EVALUATION_CONFIGS.keys())}",
        default="all",
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        help="",
        default=None,
    )
    parser.add_argument(
        "--model_args",
        type=lambda arg: json.loads(arg),
        help="additional args passed to the encoder model. JSON: --model_args '{\"my_param1\":2, \"my_param2\":30}'",
        default={},
    )
    parser.add_argument(
        "--args",
        type=lambda arg: json.loads(arg),
        help="JSON: --args '{\"per_device_train_batch_size\":16}'",
        default={},
    )
    parser.add_argument(
        "--benchmark",
        action='store_true',
        help="using deterministic mode for training/evaluation. slows down training, but is reproducible",
        default=True,
    )
    
    args = parser.parse_args()
    main(args)