# Transcript Normalization

## Task

Normalize transcripts for English, Mandarin Chinese, and Cantonese ASR training.

## Global Rules

- Apply the same normalization to train, validation, and test references.
- Use Unicode normalization.
- Remove invisible/control characters.
- Remove non-speech annotations, speaker labels, timestamps, and markup.
- Remove punctuation unless the target task explicitly requires punctuation prediction.
- Normalize whitespace.
- Discard samples whose transcript becomes empty.
- Use one consistent number normalization policy across all datasets.
- Preserve characters that are linguistically meaningful for the target language.

## English

- Lowercase all letters.
- Remove punctuation.
- Normalize apostrophes consistently: either keep apostrophe in contractions or remove/expand it.
- Normalize numbers consistently: either digits or spoken words.
- Keep only target vocabulary characters: `a-z`, space, and optionally apostrophe.

## Mandarin

- Remove Chinese and English punctuation.
- Remove spaces between Chinese characters.
- Choose one script policy: simplified, traditional, or original.
- Normalize full-width Latin letters and digits.
- Lowercase embedded English words.
- Normalize numbers consistently.
- Keep Chinese characters and allowed code-switching characters.

## Cantonese

- Remove Chinese and English punctuation.
- Remove spaces between Chinese characters.
- Lowercase embedded English words.
- Normalize numbers consistently.
