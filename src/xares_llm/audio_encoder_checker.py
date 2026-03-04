from loguru import logger
from typing import Sequence
import torch
import inspect


class EncoderValidationError(Exception):
    """Custom exception raised for invalid audio encoder structure or output."""

    pass


def check_audio_encoder(encoder):
    device = list(encoder.parameters())[0].device
    if not isinstance(encoder, torch.nn.Module):
        raise EncoderValidationError(f"Expected torch.nn.Module for encoder, got {type(encoder)}")

    if not hasattr(encoder, "output_dim"):
        raise EncoderValidationError("Encoder must have a 'output_dim' attribute")

    if not check_supports_two_args(encoder.forward):
        raise EncoderValidationError("Encoder forward needs to accept two arguments: (audio, attention_mask)")

    sample_audio = torch.randn(3, 50000, device=device)
    try:
        result = encoder(sample_audio, None)
        is_sequence = isinstance(result, Sequence) and not isinstance(result, (str, bytes))
        if not is_sequence or len(result) != 2:
            # If the format is wrong, log a specific error
            raise EncoderValidationError(
                f"Expected encoder to return a 2-element tuple or list,but got type {type(result).__name__}.\nCheck your encoder to return (embeddings, attention_mask)"
            )
        encoded_audio, attention_mask = result
    except Exception as e:
        raise EncoderValidationError(f"Failed to encode the sample audio: {e}")

    if not isinstance(encoded_audio, torch.Tensor):
        raise EncoderValidationError(f"Expected tensor for encoded_audio, got {type(encoded_audio)}")

    if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
        raise EncoderValidationError(f"Expected tensor for attention_mask, got {type(encoded_audio)}")

    if encoded_audio.ndim != 3:  # [B, T, D]
        raise EncoderValidationError(
            f"Expected 3D tensor [B, T, D] for encoded_audio, got {encoded_audio.dim()}D tensor"
        )

    if attention_mask is not None and attention_mask.ndim != 2:  # [B, T]
        raise EncoderValidationError(
            f"Expected 2D tensor [B, T] for attention_mask, got {attention_mask.dim()}D tensor"
        )

    if encoded_audio.size(0) != sample_audio.size(0):
        raise EncoderValidationError(
            f"Expected batch size={sample_audio.size(0)} for encoded_audio, got {encoded_audio.size(0)}"
        )

    if encoded_audio.size(2) != encoder.output_dim:
        raise EncoderValidationError(
            f"Expected output_dim={encoder.output_dim} for encoded_audio, got {encoded_audio.size(2)}"
        )


def check_supports_two_args(func):
    """Checks if a callable requires or accepts exactly two positional arguments."""
    try:
        sig = inspect.signature(func)
    except ValueError:
        logger.exception(f"Cannot inspect signature for {func.__name__}")
        return False

    num_positional_args = 0
    has_var_args = False  # Checks for *args

    for param in sig.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            # Check for arguments that must be supplied positionally
            if param.default is inspect.Parameter.empty:
                num_positional_args += 1
            else:
                num_positional_args += 1
        elif param.kind == param.VAR_POSITIONAL:
            # If *args is present, it can accept any number of extra positional arguments
            has_var_args = True
    if num_positional_args >= 2 or (has_var_args and num_positional_args <= 2):
        return True
    return False
