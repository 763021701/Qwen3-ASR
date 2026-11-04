"""Contextual biasing for Qwen3-ASR with one shared AuT pass.

AuT -> AuT-CTC (coarse transcript + log-probs) -> ctc_rag_hw hotword retrieval -> hotwords as the system prompt ->
Qwen3-ASR decodes from the same audio embeddings. Per batch the LLM's own AuT runs once: its ln_post output feeds
the CTC head and its final output is scattered into the LLM's input embeddings, so generate() never re-encodes the
audio. Transformers backend only: the vLLM model class takes mel features, not audio embeddings.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import torch

from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
from qwen_asr.inference.utils import (
    SAMPLE_RATE,
    merge_languages,
    normalize_audios,
    normalize_language_name,
    parse_asr_output,
    split_audio_into_chunks,
    validate_language,
)

from .data import duration_batches
from .frontend import _qwen3_asr_modeling, aut_fingerprint, compute_features, extract_aut_features
from .inference import AuTCTCHead
from .tokenizer import _TAG_RE


class CTCRagTokenizer:
    """ctc_rag_hw's ``decode(ids)`` contract over the AuT-CTC vocabulary, as a precomputed piece table.

    ctc_rag_hw decodes every id of the vocabulary on each call and joins greedy pieces without separators, so a
    word-initial ``▁`` becomes a space (its radar strips it again); blank, <unk> and tags decode to ''.
    """

    def __init__(self, tokenizer):
        self.pieces = ["" if i < 2 or _TAG_RE.fullmatch(p) else p.replace("▁", " ")
                       for i, p in enumerate(map(tokenizer.id_to_piece, range(tokenizer.vocab_size)))]

    def decode(self, ids) -> str:
        return "".join(self.pieces[int(i)] for i in ids)


@dataclass
class BiasedTranscription:
    language: str
    text: str
    ctc_text: str  # coarse AuT-CTC transcript of the same audio
    hotwords: list[str]  # retrieved over all segments, first occurrence first
    contexts: list[str]  # the system prompt of each segment


class CTCRagQwen3ASR:
    """Qwen3-ASR (transformers backend) biased by hotwords that ctc_rag_hw retrieves from AuT-CTC log-probs.

    Long audio is cut into segments of at most ``max_segment_sec`` at low-energy points (like Qwen3ASRModel at
    1200 s): retrieval cost grows with the query length, and each segment gets its own hotwords. Segments are
    batched by padded duration (``max_batch_seconds``) and count (``max_batch_size``, which bounds the LLM prefill).
    ``timings`` holds the seconds spent per stage in the last ``transcribe`` call.
    """

    def __init__(self, asr: Qwen3ASRModel, ctc: AuTCTCHead, retriever=None, hotwords=None, max_hotwords: int = 30,
                 ctc_topk: int = 30, max_segment_sec: float = 30.0, max_batch_seconds: float = 240.0,
                 max_batch_size: int = 8):
        if asr.backend != "transformers":
            raise ValueError("the shared AuT pass needs the transformers backend")
        _qwen3_asr_modeling()  # the repo's qwen_asr: an installed wheel's AuT ignores its attention windows
        if aut_fingerprint(asr.model.name_or_path) != ctc.aut_fingerprint:
            raise ValueError(f"{asr.model.name_or_path} does not have the AuT the CTC head was trained on "
                             f"({ctc.aut_model_dir})")
        self.asr, self.ctc, self.retriever, self.hotwords = asr, ctc, retriever, hotwords
        self.aut, self.device = asr.model.thinker.audio_tower, asr.device
        self.max_hotwords, self.ctc_topk = max_hotwords, ctc_topk
        self.max_segment_sec, self.max_batch_seconds, self.max_batch_size = max_segment_sec, max_batch_seconds, max_batch_size
        self.timings = {}

    @classmethod
    def from_pretrained(cls, model_dir, exp_dir, ckpt: str = "best", hotwords=None, retriever_device: str = "cpu",
                        device: str = "cuda:0", **kwargs) -> "CTCRagQwen3ASR":
        """hotwords: default library (path, text or list). retriever_device: "cpu" (numpy) or "gpu" (Triton)."""
        from ctc_rag_hw import CTCRagRetriever

        asr = Qwen3ASRModel.from_pretrained(model_dir, dtype=torch.bfloat16, device_map=device)
        ctc = AuTCTCHead(exp_dir, ckpt, device)
        retriever = CTCRagRetriever(blank_id=ctc.tokenizer.BLANK, ctc_tokenizer=CTCRagTokenizer(ctc.tokenizer),
                                    frame_shift_ms=ctc.frame_shift_ms, device=retriever_device)
        if hotwords is not None:
            retriever.load_hotwords(hotwords)  # builds the phoneme index once; per-call lookups hit its cache
        return cls(asr, ctc, retriever, hotwords, **kwargs)

    @torch.no_grad()
    def transcribe(self, audio, hotwords=None, context: str | None = None,
                   language: str | None = None) -> list[BiasedTranscription]:
        """audio: what Qwen3ASRModel.transcribe takes (path / URL / base64 / (wav, sr), or a list of them).
        hotwords: library for this call (path, text or list); default: the one given at construction.
        context: one fixed system prompt for every segment instead of retrieval ("" = no biasing).
        language: force the output language, as in Qwen3ASRModel.transcribe.
        """
        if language:
            language = normalize_language_name(language)
            validate_language(language)
        hotwords = self.hotwords if hotwords is None else hotwords
        self.timings = defaultdict(float)
        start = time.perf_counter()
        wavs = normalize_audios(audio)
        segs = [(i, seg) for i, wav in enumerate(wavs)
                for seg, _ in split_audio_into_chunks(wav, SAMPLE_RATE, self.max_segment_sec)]
        batches = duration_batches(range(len(segs)), [len(s) / SAMPLE_RATE for _, s in segs], self.max_batch_seconds)
        outs = [None] * len(segs)
        for batch in (b[k : k + self.max_batch_size] for b in batches for k in range(0, len(b), self.max_batch_size)):
            for k, out in zip(batch, self._run([segs[k][1] for k in batch], hotwords, context, language)):
                outs[k] = out

        per_audio = defaultdict(list)
        for (i, _), out in zip(segs, outs):
            per_audio[i].append(out)
        results = []
        for i in range(len(wavs)):
            parts = per_audio[i]
            results.append(BiasedTranscription(
                language=merge_languages([p["language"] for p in parts]),
                text="".join(p["text"] for p in parts),
                ctc_text="".join(p["ctc_text"] for p in parts),
                hotwords=list(dict.fromkeys(h for p in parts for h in p["hotwords"])),
                contexts=[p["context"] for p in parts],
            ))
        self.timings["total"] = time.perf_counter() - start
        return results

    def _run(self, wavs, hotwords, context, language) -> list[dict]:
        """One batch of segments through every stage."""
        t = time.perf_counter()
        feats, mel_lens = compute_features(self.asr.processor.feature_extractor, wavs)
        t = self._lap("features", t)
        layers, lens, embeds = extract_aut_features(self.aut, feats, mel_lens, self.ctc.layer_ids, return_output=True)
        t = self._lap("aut", t)
        log_probs, lp_lens = self.ctc.log_probs(layers, lens)
        ctc_texts = self.ctc.greedy_text(log_probs, lp_lens)
        t = self._lap("ctc", t)
        retrieve = context is None and bool(hotwords)
        found = [self.retriever.retrieve_from_log_probs(lp[:n], self.max_hotwords, self.ctc_topk, hotwords=hotwords)
                 .retrieved_hotwords[: self.max_hotwords] if retrieve else []
                 for lp, n in zip(log_probs, lp_lens.tolist())]
        contexts = [" ".join(h) if context is None else context for h in found]  # space-separated, as in the examples
        t = self._lap("retrieval", t)
        raw = self._generate(embeds, contexts, language)
        self._lap("llm", t)
        outs = []
        for r, ctc_text, h, c in zip(raw, ctc_texts, found, contexts):
            lang, text = parse_asr_output(r, user_language=language)
            outs.append({"language": lang, "text": text, "ctc_text": ctc_text, "hotwords": h, "context": c})
        return outs

    def _generate(self, embeds, contexts, language) -> list[str]:
        """Batch generate whose audio placeholders take the AuT embeddings directly (no input_features)."""
        processor, model = self.asr.processor, self.asr.model
        prompts = [self.asr._build_text_prompt(context=c, force_language=language) for c in contexts]
        prompts = processor.replace_multimodal_special_tokens(prompts, iter([e.size(0) for e in embeds]))
        inputs = processor.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.device)
        thinker = model.thinker
        inputs_embeds = thinker.get_input_embeddings()(inputs["input_ids"])
        inputs_embeds[inputs["input_ids"] == thinker.config.audio_token_id] = torch.cat(embeds).to(inputs_embeds.dtype)
        out = model.generate(input_ids=inputs["input_ids"], inputs_embeds=inputs_embeds,
                             attention_mask=inputs["attention_mask"], max_new_tokens=self.asr.max_new_tokens)
        return processor.batch_decode(out.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)

    def _lap(self, stage: str, since: float) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)  # charge GPU work to the stage that launched it
        now = time.perf_counter()
        self.timings[stage] += now - since
        return now
