"""Qwen3-ASR transformers 流式推理。

上游 qwen_asr 包 0.0.6 的 streaming API 只支持 vLLM backend（qwen3_asr.py:625
`if self.backend != "vllm": raise ValueError`）。在昇腾 NPU 上 vLLM 走不通
（vllm-ascend 未支持 Qwen3-ASR 架构 / torch 版本死锁），这里提供等价的
transformers 实现：算法与 vLLM 版本一致，只是 decode 换成 HF `model.generate`。

算法要点（与 qwen_asr.Qwen3ASRModel.streaming_transcribe 完全一致）：
  1. 累积 buffer；每凑齐 chunk_size_samples 就消费一个 chunk
  2. audio_accum 追加该 chunk，用「全长 audio + prefix-prompt」再生成一次
  3. prefix 策略：
       - chunk_id < unfixed_chunk_num → prefix = ""（前 N 个 chunk 不回滚）
       - 否则 → tokenize 上一次累积文本，砍掉末尾 K token 作 prefix（避免边界抖动）
  4. 整句 prompt = prompt_raw + prefix；generate 后 new_tokens = sequences[:, L:]
  5. parse_asr_output 剥掉 <language:...><asr_text> 后更新 state.text / .language

自 v2 同步（qwen3_transformers_streaming_v2.py）：
  - commit_segment() 分段落定：在语音停顿点把当前文本定稿、清空 audio_accum，
    解决累积音频带来的 O(N) 延迟增长（原依赖的 qwen3_fast_decode.TOKENIZER_LOCK
    换成本地 threading.Lock；_fast_decoder/_batch_scheduler 钩子未同步）

性能档位（环境变量）：
  QWEN3_PROFILE=1            → INFO 级别打印 processor/generate/decode 三段耗时
  QWEN3_FORCE_SDPA=0         → 跳过 sdpa attn_implementation 注入（默认开）

命令行用法：
  python examples/qwen3_transformers_streaming.py --audio test.wav \
      [--model Qwen/Qwen3-ASR-0.6B] [--language Cantonese] [--chunk_size_sec 2.0] \
      [--auto_commit 1]
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Any, List, Optional

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from qwen_asr.inference.qwen3_asr import ASRStreamingState
from qwen_asr.inference.utils import (
    SAMPLE_RATE,
    normalize_language_name,
    parse_asr_output,
    validate_language,
)


_QWEN3_PROFILE = os.environ.get("QWEN3_PROFILE", "0") == "1"
_QWEN3_FORCE_SDPA = os.environ.get("QWEN3_FORCE_SDPA", "1") == "1"
_SDPA_TRIED: set = set()  # id(inner) → 已尝试过 sdpa 注入的 model
# tokenizer 是跨会话共享的 Rust 对象，并发使用会抛 "Already borrowed"；
# v2 从 qwen3_fast_decode 导入，此处换成本地锁（该模块不在本仓库）
_TOKENIZER_LOCK = threading.Lock()
# 招 1：RMS < 阈值视为静音 chunk，跳过 _decode_chunk（仍累积 audio_accum）。
# 0.003 ≈ -50dB，正常讲话 RMS 0.02-0.1，背景噪声 0.005-0.01。
# 设 0.003 是保守值——边缘喃喃可能误判，但宁错放别误跳。可用 env 调。
_QWEN3_SILENCE_RMS = float(os.environ.get("QWEN3_SILENCE_RMS", "0.003"))


def _maybe_enable_sdpa(inner: Any) -> None:
    """一次性尝试把 LLM/encoder 切到 SDPA attention。失败静默回退。"""
    if not _QWEN3_FORCE_SDPA:
        return
    key = id(inner)
    if key in _SDPA_TRIED:
        return
    _SDPA_TRIED.add(key)
    try:
        cfg = getattr(inner, "config", None)
        if cfg is not None and getattr(cfg, "_attn_implementation", "") != "sdpa":
            cfg._attn_implementation = "sdpa"
            logger.info("[qwen3] inner attn_implementation set to sdpa")
    except Exception as exc:  # noqa: BLE001
        logger.debug("[qwen3] sdpa enable skipped: {}", exc)


def init_streaming_state(
    model: Any,
    context: str = "",
    language: Optional[str] = None,
    unfixed_chunk_num: int = 2,
    unfixed_token_num: int = 5,
    chunk_size_sec: float = 2.0,
) -> ASRStreamingState:
    """创建 transformers backend 下的流式 state（签名与 vLLM 版一致）。

    language 支持：
      - 单语言 "English" → 官方 force_language 路径
      - 多语言 "English,Cantonese" → 魔改 prompt：拼接多段 language X<asr_text>
        让模型在输出时被两种语言 prefix 同时 prime（hack，非官方用法）
    """
    if chunk_size_sec is None or float(chunk_size_sec) <= 0:
        raise ValueError(f"chunk_size_sec must be > 0, got: {chunk_size_sec}")

    force_language = None
    multi_langs: List[str] = []
    if language is not None and str(language).strip():
        raw = str(language).strip()
        if "," in raw:
            multi_langs = [normalize_language_name(x) for x in raw.split(",") if x.strip()]
            for ln in multi_langs:
                validate_language(ln)
            force_language = ",".join(multi_langs)  # 标记多语言模式
        else:
            ln = normalize_language_name(raw)
            validate_language(ln)
            force_language = ln

    chunk_size_samples = max(1, int(round(float(chunk_size_sec) * SAMPLE_RATE)))
    if multi_langs:
        # 手工拼 prompt：base + (language X<asr_text>) × N
        msgs = model._build_messages(context=context, audio_payload="")
        base = model.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        suffix = "".join(f"language {ln}<asr_text>" for ln in multi_langs)
        prompt_raw = base + suffix
    else:
        prompt_raw = model._build_text_prompt(context=context, force_language=force_language)

    return ASRStreamingState(
        unfixed_chunk_num=int(unfixed_chunk_num),
        unfixed_token_num=int(unfixed_token_num),
        chunk_size_sec=float(chunk_size_sec),
        chunk_size_samples=int(chunk_size_samples),
        chunk_id=0,
        buffer=np.zeros((0,), dtype=np.float32),
        audio_accum=np.zeros((0,), dtype=np.float32),
        prompt_raw=prompt_raw,
        context=context or "",
        force_language=force_language,
        language="",
        text="",
        _raw_decoded="",
    )


def _committed(state: Any) -> str:
    """分段落定后的历史文本；未落定过则为空串。"""
    return getattr(state, "_committed_text", "")


def _seg_chunk_id(state: Any) -> int:
    """当前分段内的 chunk 序号，用于 unfixed_chunk_num 判定（落定后重新计数）。"""
    return state.chunk_id - getattr(state, "_seg_start_chunk", 0)


def commit_segment(state: Any) -> bool:
    """把当前已转写内容落定，清空累积音频，从下一个 chunk 起重新开始。

    为什么需要：audio_accum 只增不减，每个 chunk 都要重新处理全长音频，
    预处理耗时随会话时长线性增长，长会话延迟不可接受。

    必须由调用方在 VAD 判定的语音结束点调用，不能自己找切点：
      - 此刻 _raw_decoded 恰好覆盖 audio_accum 全部内容，两者边界对齐，
        全提交 + 全丢弃 => 不丢字也不重复
      - 语音结束点上没有词被切断
    在别处调用会丢字或重复。
    """
    if state is None or not getattr(state, "_raw_decoded", ""):
        return False
    if state.audio_accum.shape[0] == 0:
        return False
    _, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
    accum_sec = state.audio_accum.shape[0] / float(SAMPLE_RATE)
    state._committed_text = _committed(state) + txt
    state._raw_decoded = ""
    state.audio_accum = np.zeros((0,), dtype=np.float32)
    state._seg_start_chunk = state.chunk_id
    state.text = state._committed_text
    if _QWEN3_PROFILE:
        logger.info(
            "[qwen3.profile] COMMIT_SEGMENT  落定 {:.1f}s 音频，累计文本 {} 字",
            accum_sec, len(state._committed_text),
        )
    return True


def _as_mono_float32(pcm: np.ndarray) -> np.ndarray:
    x = np.asarray(pcm)
    if x.ndim != 1:
        x = x.reshape(-1)
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0)
    return x.astype(np.float32, copy=False)


def _rollback_prefix(tokenizer, raw_decoded: str, rollback_k: int) -> str:
    with _TOKENIZER_LOCK:
        cur_ids = tokenizer.encode(raw_decoded)
    k = int(rollback_k)
    while True:
        end_idx = max(0, len(cur_ids) - k)
        with _TOKENIZER_LOCK:
            prefix = tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
        if "\ufffd" not in prefix:
            return prefix
        if end_idx == 0:
            return ""
        k += 1


@torch.inference_mode()
def _decode_chunk(model: Any, audio_accum: np.ndarray, prompt: str,
                  max_new_tokens: Optional[int] = None) -> str:
    """对 (prompt, audio_accum) 跑一次 generate，只返回新生成 token 的 decoded 文本。

    max_new_tokens override（A 方案）：CTC 估的字数 × factor，None 时回退 model.max_new_tokens。

    性能档位（阶段 1 免费档）：
      - inference_mode 替代 no_grad：少一层 autograd 状态追踪
      - processor padding=False：batch=1 不需要 pad，省一次拷贝
      - sdpa attn_implementation：触发 PyTorch 融合注意力
    QWEN3_PROFILE=1 时打印 processor / generate / decode 三段耗时。

    跨会话批处理（招 8）：若 model 上挂了 _batch_scheduler，则改为 submit 给中央 worker；
    各会话的 _decode_chunk 调用并发提交，scheduler 自动批量 generate，效率取决于
    当下队列深度（自然形成 batch=2-4）。
    """
    sched = getattr(model, "_batch_scheduler", None)
    if sched is not None:
        mnt = int(max_new_tokens) if max_new_tokens else int(model.max_new_tokens)
        t0 = time.perf_counter() if _QWEN3_PROFILE else 0.0
        out = sched.submit(audio_accum, prompt, mnt)
        if _QWEN3_PROFILE:
            elapsed = time.perf_counter() - t0
            logger.info(
                "[qwen3.profile] sched_submit  elapsed={:.1f}ms  "
                "audio_samples={}  prompt_chars={}  mnt={}",
                elapsed * 1000.0, int(audio_accum.shape[0]), len(prompt), mnt,
            )
        return out

    inner = model.model  # HF AutoModel
    processor = model.processor
    _maybe_enable_sdpa(inner)

    t0 = time.perf_counter()
    with _TOKENIZER_LOCK:
        inputs = processor(
            text=[prompt],
            audio=[audio_accum],
            return_tensors="pt",
            padding=False,
        )
    inputs = inputs.to(inner.device).to(inner.dtype)
    t1 = time.perf_counter()

    mnt = int(max_new_tokens) if max_new_tokens else int(model.max_new_tokens)
    # no_repeat_ngram_size：防止 Qwen3 ASR 在静音/噪音段陷入"a little seat, a little seat..."循环
    # 0=关闭，环境变量 QWEN3_NO_REPEAT_NGRAM 调（默认 3，影响很小但能切断小循环）
    nr = int(os.environ.get("QWEN3_NO_REPEAT_NGRAM", "3") or "0")
    gen_kwargs = {"max_new_tokens": mnt}
    if nr > 0:
        gen_kwargs["no_repeat_ngram_size"] = nr
    text_ids = inner.generate(**inputs, **gen_kwargs)
    t2 = time.perf_counter()

    new_ids = text_ids.sequences[:, inputs["input_ids"].shape[1]:]
    with _TOKENIZER_LOCK:
        decoded = processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    t3 = time.perf_counter()

    if _QWEN3_PROFILE:
        logger.info(
            "[qwen3.profile] proc={:.1f}ms  gen={:.1f}ms  dec={:.1f}ms  "
            "audio_samples={}  prompt_chars={}  mnt={}",
            (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0,
            int(audio_accum.shape[0]), len(prompt), mnt,
        )
    return decoded[0] if decoded else ""


def streaming_transcribe(
    model: Any,
    pcm16k: np.ndarray,
    state: ASRStreamingState,
    dynamic_max_new_tokens: Optional[int] = None,
    skip_decode: bool = False,
    rag_context: Optional[str] = None,
) -> ASRStreamingState:
    """transformers backend 的流式增量 decode。state 原地更新。

    新增 CTC-driven 参数：
      dynamic_max_new_tokens : 单次 generate 的 max_new_tokens 上限（None 用 model.max_new_tokens）
      skip_decode            : True → 跳过 _decode_chunk，仅吸收 chunk + 推进 id（静音节省）
      rag_context            : 非空 → 即时重建 prompt_raw（mid-stream 热词注入）
    """
    if state is None:
        raise ValueError("state must not be None. Call init_streaming_state() first.")
    if pcm16k is None:
        raise ValueError("pcm16k must not be None.")

    x = _as_mono_float32(pcm16k)
    if x.shape[0] > 0:
        state.buffer = np.concatenate([state.buffer, x], axis=0)

    # mid-stream 热词更新：覆写 prompt_raw（context 透过 _build_text_prompt 重建）
    if rag_context:
        try:
            new_prompt = model._build_text_prompt(
                context=rag_context, force_language=state.force_language,
            )
            state.prompt_raw = new_prompt
            state.context = rag_context
        except Exception as exc:
            logger.debug("[qwen3] rag_context update skipped: {}", exc)

    tokenizer = model.processor.tokenizer

    while state.buffer.shape[0] >= state.chunk_size_samples:
        chunk = state.buffer[: state.chunk_size_samples]
        state.buffer = state.buffer[state.chunk_size_samples:]

        state.audio_accum = (chunk if state.audio_accum.shape[0] == 0
                             else np.concatenate([state.audio_accum, chunk], axis=0))

        # 招 1：RMS 静音检测——本 chunk 全是静音/低噪 → 跳 _decode_chunk
        # （audio_accum 已累积，下一个非静音 chunk 会一起带进去）
        skip_silence = False
        if _QWEN3_SILENCE_RMS > 0:
            chunk_rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2) + 1e-12))
            if chunk_rms < _QWEN3_SILENCE_RMS:
                skip_silence = True
                if _QWEN3_PROFILE:
                    logger.info(
                        "[qwen3.profile] chunk_id={}  rms={:.5f}  SKIPPED_silence_rms",
                        state.chunk_id, chunk_rms,
                    )

        # C 方案 / 招 1：静音 chunk 跳过 forward，但仍消化 buffer + 累积 audio_accum + 推进 chunk_id
        if skip_decode or skip_silence:
            if _QWEN3_PROFILE and skip_decode:
                logger.info(
                    "[qwen3.profile] chunk_id={}  audio_accum={:.2f}s  SKIPPED_decode_flag",
                    state.chunk_id, state.audio_accum.shape[0] / float(SAMPLE_RATE),
                )
            state.chunk_id += 1
            continue

        prefix = ("" if _seg_chunk_id(state) < state.unfixed_chunk_num
                  else _rollback_prefix(tokenizer, state._raw_decoded, state.unfixed_token_num))

        prompt = state.prompt_raw + prefix
        t_chunk = time.perf_counter() if _QWEN3_PROFILE else 0.0
        gen_text = _decode_chunk(model, state.audio_accum, prompt,
                                 max_new_tokens=dynamic_max_new_tokens)
        state._raw_decoded = (prefix + gen_text)

        if _QWEN3_PROFILE:
            audio_dur = state.audio_accum.shape[0] / float(SAMPLE_RATE)
            elapsed = time.perf_counter() - t_chunk
            logger.info(
                "[qwen3.profile] chunk_id={}  audio_accum={:.2f}s  total={:.1f}ms  "
                "RTF_inst={:.3f}",
                state.chunk_id, audio_dur, elapsed * 1000.0,
                elapsed / max(audio_dur, 1e-6),
            )

        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language = lang
        state.text = _committed(state) + txt
        state.chunk_id += 1

    return state


def finish_streaming_transcribe(model: Any, state: ASRStreamingState) -> ASRStreamingState:
    """flush 剩余 buffer（不足一个 chunk 的尾部），做最后一次 decode。"""
    if state is None:
        raise ValueError("state must not be None.")
    if state.buffer is None or state.buffer.shape[0] == 0:
        if _committed(state) and not state.text:
            state.text = _committed(state)
        return state

    tail = state.buffer
    state.buffer = np.zeros((0,), dtype=np.float32)
    state.audio_accum = (tail if state.audio_accum.shape[0] == 0
                         else np.concatenate([state.audio_accum, tail], axis=0))

    tokenizer = model.processor.tokenizer
    if _seg_chunk_id(state) < state.unfixed_chunk_num:
        prefix = ""
    else:
        with _TOKENIZER_LOCK:
            cur_ids = tokenizer.encode(state._raw_decoded)
        end_idx = max(1, len(cur_ids) - int(state.unfixed_token_num))
        prefix = tokenizer.decode(cur_ids[:end_idx])

    prompt = state.prompt_raw + prefix
    gen_text = _decode_chunk(model, state.audio_accum, prompt)
    state._raw_decoded = (prefix + gen_text)

    lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
    state.language = lang
    state.text = _committed(state) + txt
    state.chunk_id += 1
    return state


def _load_wav_16k(path: str) -> np.ndarray:
    """读音频为 16k 单声道 float32；非 16k 用线性插值重采样（与 vLLM 流式示例同款，够 demo 用）。"""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:  # 多声道 → 平均成单声道
        wav = wav.mean(axis=1)
    if sr == SAMPLE_RATE:
        return wav
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * SAMPLE_RATE))
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3-ASR transformers 流式转写（qwen_asr 包 streaming API 的 transformers 等价实现）",
    )
    p.add_argument("--audio", nargs="+", required=True,
                   help="输入音频路径（soundfile 支持的格式），可传多个逐个转写")
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B",
                   help="模型路径或 HF repo id（默认 Qwen/Qwen3-ASR-0.6B）")
    p.add_argument("--language", default=None,
                   help="强制语言，如 English / Cantonese；逗号分隔多语言走 hack prompt；默认自动识别")
    p.add_argument("--context", default="",
                   help="热词/上下文 prompt（可含领域术语）")
    p.add_argument("--chunk_size_sec", type=float, default=2.0,
                   help="流式 chunk 长度（秒）")
    p.add_argument("--unfixed_chunk_num", type=int, default=2,
                   help="前 N 个 chunk 不做 prefix 回滚")
    p.add_argument("--unfixed_token_num", type=int, default=5,
                   help="prefix 回滚时砍掉末尾的 token 数")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="推理设备（默认 cuda，无 GPU 回退 cpu）")
    p.add_argument("--auto_commit", type=int, default=0,
                   help="1=整 chunk 静音且段长≥min_commit_sec 时自动 commit_segment（简易 VAD；"
                        "正式使用应由外部 VAD 在语音结束点调用 commit_segment）")
    p.add_argument("--pause_rms", type=float, default=0.003,
                   help="auto_commit 的静音 chunk RMS 阈值（默认 0.003 ≈ -50dB）")
    p.add_argument("--min_commit_sec", type=float, default=6.0,
                   help="auto_commit 触发前当前段最少累积音频秒数")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from qwen_asr import Qwen3ASRModel

    dtype = torch.bfloat16 if "cuda" in args.device else torch.float32
    model = Qwen3ASRModel.from_pretrained(args.model, dtype=dtype, device_map=args.device)
    logger.info("model loaded: {} on {} ({})", args.model, args.device, dtype)

    for path in args.audio:
        wav16k = _load_wav_16k(path)
        dur = wav16k.shape[0] / float(SAMPLE_RATE)
        print(f"\n===== {path} ({dur:.1f}s, chunk={args.chunk_size_sec}s) =====")
        state = init_streaming_state(
            model,
            context=args.context,
            language=args.language,
            unfixed_chunk_num=args.unfixed_chunk_num,
            unfixed_token_num=args.unfixed_token_num,
            chunk_size_sec=args.chunk_size_sec,
        )
        step = state.chunk_size_samples
        for pos in range(0, wav16k.shape[0], step):
            seg = wav16k[pos:pos + step]
            state = streaming_transcribe(model, seg, state)
            note = ""
            if args.auto_commit and seg.shape[0] == step:
                rms = float(np.sqrt(np.mean(seg.astype(np.float32) ** 2) + 1e-12))
                seg_sec = state.audio_accum.shape[0] / float(SAMPLE_RATE)
                if rms < args.pause_rms and seg_sec >= args.min_commit_sec and state._raw_decoded:
                    if commit_segment(state):
                        note = f"  [commit @ {pos / SAMPLE_RATE:.0f}s, 定稿 {len(state.text)} 字]"
            print(f"[chunk {state.chunk_id:02d}] language={state.language!r} text={state.text!r}{note}")
        state = finish_streaming_transcribe(model, state)
        print(f"[final] language={state.language!r}")
        print(f"[final] text={state.text!r}")


if __name__ == "__main__":
    main()
