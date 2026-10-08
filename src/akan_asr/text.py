"""Text normalisation for Akan (Twi) transcripts.

Every reference transcript and every model prediction passes through the same
`normalize` function before scoring. If normalisation differs between models,
WER differences reflect formatting, not recognition quality.

Design choices (state them in your write-up):
- Unicode NFC first, so visually identical strings compare equal.
- Look-alike characters typed in place of the Akan letters are mapped to the
  real ones: Greek epsilon -> ɛ, open-o look-alikes -> ɔ.
- Tone and accent marks (combining diacritics) are removed by default, matching
  the 2025 UG benchmark, whose references use standard orthography without tone.
  ɛ and ɔ are base letters, not diacritics, so they survive.
- Lowercase, strip punctuation, keep word-internal apostrophes (n'ani).
- Optional: map the informal keyboard substitutes "3" -> ɛ and ")" -> ɔ that
  appear in typed Twi. Off by default because "3" can be a real digit.
"""

from __future__ import annotations

import re
import unicodedata

# Characters people type in place of the Akan letters ɛ (U+025B) and ɔ (U+0254).
_LOOKALIKES = {
    "ε": "ɛ",  # Greek small epsilon ε -> ɛ
    "Ε": "ɛ",  # Greek capital epsilon Ε -> ɛ
    "Ɛ": "ɛ",  # Latin capital open E Ɛ -> ɛ
    "Ɔ": "ɔ",  # Latin capital open O Ɔ -> ɔ
    "ͻ": "ɔ",  # Greek small reversed lunate sigma ͻ -> ɔ
    "Ͻ": "ɔ",  # Greek capital reversed lunate sigma Ͻ -> ɔ
    "ↄ": "ɔ",  # Latin small reversed C ↄ -> ɔ
    "Ↄ": "ɔ",  # Roman numeral reversed C Ↄ -> ɔ
}

_APOSTROPHES = {
    "’": "'",  # right single quotation mark
    "‘": "'",  # left single quotation mark
    "ʼ": "'",  # modifier letter apostrophe
    "`": "'",
    "´": "'",
}

_LOOKALIKE_TABLE = str.maketrans({**_LOOKALIKES, **_APOSTROPHES})

# Informal ASCII substitutes, applied only inside words that contain letters.
_ASCII_SUB_RE = re.compile(r"(?<=[^\W\d_])3|3(?=[^\W\d_])")
_ASCII_PAREN_RE = re.compile(r"(?<=[^\W\d_])\)|\)(?=[^\W\d_])")

_WS_RE = re.compile(r"\s+")


def _strip_combining(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    kept = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", kept)


def _strip_punctuation(text: str) -> str:
    out: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch == "'":
            # Keep apostrophes only between two letters (n'ani, w'ani).
            prev_is_letter = i > 0 and text[i - 1].isalpha()
            next_is_letter = i + 1 < n and text[i + 1].isalpha()
            out.append("'" if prev_is_letter and next_is_letter else " ")
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def normalize(
    text: str | None,
    *,
    strip_diacritics: bool = True,
    map_ascii_substitutes: bool = False,
) -> str:
    """Normalise an Akan transcript for training targets and WER scoring.

    >>> normalize("Ɛte sɛn?  Me ho yɛ!")
    'ɛte sɛn me ho yɛ'
    >>> normalize("Ͻno na ɔbae.")
    'ɔno na ɔbae'
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.translate(_LOOKALIKE_TABLE)
    if map_ascii_substitutes:
        t = _ASCII_SUB_RE.sub("ɛ", t)
        t = _ASCII_PAREN_RE.sub("ɔ", t)
    t = t.lower()
    if strip_diacritics:
        t = _strip_combining(t)
    t = _strip_punctuation(t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def akan_charset(texts) -> set[str]:
    """Return every character used across normalised texts.

    Run this on each dataset after normalising: unexpected characters
    (digits, stray symbols, Latin letters Twi rarely uses) are cleaning targets.
    """
    chars: set[str] = set()
    for t in texts:
        chars.update(normalize(t))
    chars.discard(" ")
    return chars
