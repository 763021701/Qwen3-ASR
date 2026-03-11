# coding=utf-8
"""
CTC-based hotword retrieval using Fun-ASR-Nano's encoder + CTC decoder.

Provides CTCHotwordRetriever that:
  1. Loads Fun-ASR-Nano's audio encoder + CTC components (optionally discards the LLM to save VRAM).
  2. Runs CTC greedy decode on audio to produce rough transcription.
  3. Uses PhonemeCorrector (FastRAG + AccuRAG) to retrieve relevant hotwords.
  4. Formats retrieved hotwords as a context string for Qwen3-ASR injection.
"""

import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class HotwordRetrievalResult:
    """Result of CTC-based hotword retrieval."""

    ctc_text: str
    retrieved_hotwords: List[str]
    context_string: str
    hotword_scores: Dict[str, float] = field(default_factory=dict)
    correction_result: Optional[Any] = None
    details: Optional[Dict] = None


class CTCHotwordRetriever:
    """
    Lightweight CTC hotword retriever backed by Fun-ASR-Nano's encoder.

    Workflow:
        audio -> fbank -> SenseVoice encoder -> CTC decoder -> ctc_text
        ctc_text + hotword_list -> PhonemeCorrector -> retrieved_hotwords
        retrieved_hotwords -> format_context() -> context string for Qwen3-ASR
    """

    def __init__(
        self,
        nano_model: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        nano_remote_code: Optional[str] = None,
        device: str = "cuda:0",
        ctc_only: bool = True,
        threshold: float = 0.7,
        similar_threshold: float = 0.5,
        min_hotword_chars: int = 2,
        min_hotword_phonemes: int = 3,
        min_char_coverage: float = 0.6,
    ):
        """
        Args:
            nano_model: Fun-ASR-Nano model name or local path.
            nano_remote_code: Path to Fun-ASR model.py. If None, uses default.
            device: Torch device for CTC inference.
            ctc_only: If True, delete LLM and audio_adaptor after loading to
                      save ~1.2 GB VRAM.
            threshold: PhonemeCorrector match threshold.
            similar_threshold: PhonemeCorrector similar-match threshold.
            min_hotword_chars: Minimum character count for a hotword.
            min_hotword_phonemes: Minimum phoneme count (shorter ones get stricter thresholds).
            min_char_coverage: Minimum ratio of matched chars to hotword length.
        """
        self.device = device
        self.threshold = threshold
        self.similar_threshold = similar_threshold
        self.min_hotword_chars = min_hotword_chars
        self.min_hotword_phonemes = min_hotword_phonemes
        self.min_char_coverage = min_char_coverage

        self._corrector = None
        self._hotword_cache_key = None

        self._load_ctc_model(nano_model, nano_remote_code, ctc_only)

    def _load_ctc_model(self, nano_model: str, nano_remote_code: Optional[str], ctc_only: bool):
        """Load Fun-ASR-Nano and extract CTC components."""
        from funasr import AutoModel as FunASRAutoModel

        load_kwargs = dict(
            model=nano_model,
            device=self.device,
            disable_update=True,
        )
        if nano_remote_code is not None:
            abs_remote_code = os.path.abspath(nano_remote_code)
            load_kwargs["remote_code"] = abs_remote_code
            load_kwargs["trust_remote_code"] = True
            # Workaround: funasr's import_module_from_path uses
            #   importlib.import_module("model")
            # which is fragile — it can resolve to a WRONG model.py if another
            # package (e.g. nagisa) has injected its own directory into sys.path.
            # Fix: clear the stale sys.modules cache AND prepend the correct
            # directory so it is found first during the path search.
            module_name = os.path.splitext(os.path.basename(abs_remote_code))[0]
            sys.modules.pop(module_name, None)
            remote_dir = os.path.dirname(abs_remote_code)
            if remote_dir not in sys.path:
                sys.path.insert(0, remote_dir)

        logger.info(f"Loading Fun-ASR-Nano CTC model: {nano_model}")
        wrapper = FunASRAutoModel(**load_kwargs)

        inner = wrapper.model  # FunASRNano nn.Module
        if inner.ctc_decoder is None:
            raise RuntimeError(
                "The loaded Fun-ASR-Nano model does not have a CTC decoder. "
                "Ensure the model checkpoint includes CTC components."
            )

        self.audio_encoder = inner.audio_encoder
        self.ctc_decoder = inner.ctc_decoder
        self.ctc = inner.ctc
        self.ctc_tokenizer = inner.ctc_tokenizer
        self.blank_id = inner.blank_id

        self.frontend = wrapper.kwargs.get("frontend", None)
        if self.frontend is None:
            raise RuntimeError("Failed to extract frontend from Fun-ASR-Nano model.")

        if ctc_only:
            if hasattr(inner, "llm") and inner.llm is not None:
                del inner.llm
                inner.llm = None
            if hasattr(inner, "audio_adaptor") and inner.audio_adaptor is not None:
                del inner.audio_adaptor
                inner.audio_adaptor = None
            del wrapper
            torch.cuda.empty_cache()
            logger.info("CTC-only mode: deleted LLM and audio_adaptor to save VRAM.")
        else:
            self._wrapper = wrapper

        self.audio_encoder.eval()
        self.ctc_decoder.eval()

    def _ensure_hotword_module(self):
        """Ensure the hotword module from Fun-ASR project is importable."""
        try:
            from hotword import PhonemeCorrector
            return PhonemeCorrector
        except ImportError:
            fun_asr_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            )
            if fun_asr_root not in sys.path:
                sys.path.insert(0, fun_asr_root)
                logger.info(f"Added Fun-ASR root to sys.path: {fun_asr_root}")
            from hotword import PhonemeCorrector
            return PhonemeCorrector

    def load_hotwords(self, hotwords: Union[str, List[str]]) -> int:
        """
        Load hotwords from a file path or a list of strings.

        Args:
            hotwords: File path (one hotword per line) or list of hotword strings.

        Returns:
            Number of hotwords loaded.
        """
        PhonemeCorrector = self._ensure_hotword_module()

        if isinstance(hotwords, str) and hotwords:
            with open(hotwords, "r", encoding="utf-8") as f:
                hw_text = f.read()
            cache_key = f"file:{os.path.abspath(hotwords)}"
        elif isinstance(hotwords, (list, tuple)):
            hw_text = "\n".join(hotwords)
            cache_key = f"list:{len(hotwords)}:{hash(tuple(hotwords))}"
        else:
            raise ValueError(f"hotwords must be a file path string or list, got {type(hotwords)}")

        if self._hotword_cache_key == cache_key and self._corrector is not None:
            return len(self._corrector.hotwords)

        self._corrector = PhonemeCorrector(
            threshold=self.threshold,
            similar_threshold=self.similar_threshold,
            min_hotword_chars=self.min_hotword_chars,
            min_hotword_phonemes=self.min_hotword_phonemes,
            min_char_coverage=self.min_char_coverage,
        )
        n = self._corrector.update_hotwords(hw_text)
        self._hotword_cache_key = cache_key
        logger.info(f"Loaded {n} hotwords into PhonemeCorrector")
        return n

    def _load_audio(self, audio: Union[str, np.ndarray, Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load audio and extract fbank features.

        Args:
            audio: File path, raw numpy waveform (16kHz mono float32),
                   or (np.ndarray, sample_rate) tuple.

        Returns:
            (speech, speech_lengths) tensors ready for encoder.
        """
        from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

        tmp_file = None
        try:
            if isinstance(audio, str):
                audio_path = audio
            elif isinstance(audio, tuple) and len(audio) == 2:
                wav, sr = audio
                wav = np.asarray(wav, dtype=np.float32)
                if sr != 16000:
                    try:
                        import librosa
                        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                    except ImportError:
                        raise RuntimeError(
                            f"Audio sample rate is {sr}, but librosa is not installed for resampling. "
                            "Install it with: pip install librosa"
                        )
                tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                import soundfile as sf
                sf.write(tmp_file.name, wav, 16000)
                audio_path = tmp_file.name
            elif isinstance(audio, np.ndarray):
                tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                import soundfile as sf
                sf.write(tmp_file.name, audio.astype(np.float32), 16000)
                audio_path = tmp_file.name
            else:
                raise ValueError(f"Unsupported audio type: {type(audio)}")

            data_src = load_audio_text_image_video(audio_path, fs=self.frontend.fs)
            speech, speech_lengths = extract_fbank(
                data_src, data_type="sound", frontend=self.frontend, is_final=True
            )
        finally:
            if tmp_file is not None:
                try:
                    os.unlink(tmp_file.name)
                except OSError:
                    pass

        return speech.to(self.device), speech_lengths.to(self.device)

    @torch.no_grad()
    def ctc_decode(self, audio: Union[str, np.ndarray, Tuple]) -> str:
        """
        Run CTC greedy decode on audio.

        Args:
            audio: File path, numpy waveform, or (ndarray, sr) tuple.

        Returns:
            CTC decoded text (rough transcription).
        """
        speech, speech_lengths = self._load_audio(audio)

        encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)
        decoder_out, _ = self.ctc_decoder(encoder_out, encoder_out_lens)
        ctc_logits = self.ctc.log_softmax(decoder_out)

        x = ctc_logits[0, : encoder_out_lens[0].item(), :]
        yseq = x.argmax(dim=-1)
        yseq = torch.unique_consecutive(yseq, dim=-1)
        ctc_text = self.ctc_tokenizer.decode(yseq[yseq != self.blank_id].tolist())

        return ctc_text

    def retrieve(
        self,
        audio: Union[str, np.ndarray, Tuple],
        top_k: int = 50,
        max_hotwords: int = 30,
    ) -> HotwordRetrievalResult:
        """
        Run full CTC -> PhonemeCorrector pipeline to retrieve hotwords.

        Args:
            audio: Audio input (file path / numpy array / (ndarray, sr) tuple).
            top_k: Number of candidates for PhonemeCorrector.
            max_hotwords: Maximum number of hotwords to return.

        Returns:
            HotwordRetrievalResult with ctc_text, retrieved hotwords, and context string.
        """
        if self._corrector is None or not self._corrector.hotwords:
            raise RuntimeError("No hotwords loaded. Call load_hotwords() first.")

        ctc_text = self.ctc_decode(audio)

        if not ctc_text:
            return HotwordRetrievalResult(
                ctc_text="",
                retrieved_hotwords=[],
                context_string="",
            )

        res = self._corrector.correct(ctc_text, k=top_k)

        hw_best_score: Dict[str, float] = {}
        for _, hw, score in res.matchs:
            hw_best_score[hw] = max(hw_best_score.get(hw, 0.0), score)
        for _, hw, score in res.similars:
            hw_best_score[hw] = max(hw_best_score.get(hw, 0.0), score)

        ranked = sorted(hw_best_score.items(), key=lambda x: x[1], reverse=True)
        retrieved = [hw for hw, _ in ranked[:max_hotwords]]
        scores = {hw: sc for hw, sc in ranked[:max_hotwords]}

        context_str = self.format_context(retrieved)

        return HotwordRetrievalResult(
            ctc_text=ctc_text,
            retrieved_hotwords=retrieved,
            context_string=context_str,
            hotword_scores=scores,
            correction_result=res,
            details=getattr(res, "details", None),
        )

    @staticmethod
    def format_context(hotwords: List[str]) -> str:
        """Format hotwords as space-separated string for Qwen3-ASR context."""
        return " ".join(hotwords)
