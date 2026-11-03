"""AuT-CTC: frozen Qwen3-ASR audio encoder (AuT) + trainable Conformer-lite CTC decoder (zh/en).

Design: references/aut_ctc_design/DESIGN.md (v1.1). Run every entry point from the repo root
(``python -m aut_ctc.<module>``) so that the repo's ``qwen_asr`` (with the AuT window-mask fix) is used.
"""
