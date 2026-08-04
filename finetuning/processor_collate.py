"""Build processor batch inputs with a single mel pass."""

from typing import Any, Dict, List, Tuple

import numpy as np
from qwen_asr.core.transformers_backend.processing_qwen3_asr import (
    Qwen3ASRProcessorKwargs,
    _get_feat_extract_output_lengths,
)


def build_processor_batch_inputs(
    processor: Any,
    full_texts: List[str],
    prefix_texts: List[str],
    audios: List[np.ndarray],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract mel once, then tokenize full and prefix texts separately."""
    output_kwargs = processor._merge_kwargs(
        Qwen3ASRProcessorKwargs,
        tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    audio_kwargs = dict(output_kwargs["audio_kwargs"])
    audio_kwargs["padding"] = True
    audio_kwargs["truncation"] = False
    audio_kwargs["return_tensors"] = "pt"

    text_kwargs = dict(output_kwargs["text_kwargs"])
    text_kwargs["padding"] = True
    text_kwargs["return_tensors"] = "pt"

    mel = processor.feature_extractor(audios, **audio_kwargs)
    feat_attn = mel.pop("attention_mask")
    input_features = mel.pop("input_features")
    audio_lengths = _get_feat_extract_output_lengths(feat_attn.sum(-1))

    full_expanded = processor.replace_multimodal_special_tokens(
        full_texts, iter(audio_lengths)
    )
    prefix_expanded = processor.replace_multimodal_special_tokens(
        prefix_texts, iter(audio_lengths)
    )

    full_tok = processor.tokenizer(full_expanded, **text_kwargs)
    prefix_tok = processor.tokenizer(prefix_expanded, **text_kwargs)

    full_inputs = dict(full_tok)
    full_inputs["input_features"] = input_features
    full_inputs["feature_attention_mask"] = feat_attn
    return full_inputs, prefix_tok
