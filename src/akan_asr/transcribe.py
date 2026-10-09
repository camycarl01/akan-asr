"""Batch transcription with Whisper (zero-shot or fine-tuned).

Whisper has no Akan language token. By default we let it auto-detect
(language=None), which is what the 2025 UG benchmark effectively did. You can
force a token with --language to test whether e.g. "yoruba" or "swahili"
decodes Twi better; record the choice next to the result.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

log = logging.getLogger(__name__)


# Measured on Ashesi and WAXAL: real transcripts use at most ~14 Whisper tokens
# per second of normalised Twi (median 2-6). Allowing a little under double that
# keeps every real transcript while stopping a looping decode early; one looping
# clip otherwise holds its whole batch to the full 225-token limit.
TOKENS_PER_SECOND = 12
TOKEN_MARGIN = 24


def token_budget(seconds: float, cap: int) -> int:
    """Max new tokens for a batch whose longest clip lasts `seconds`.

    >>> token_budget(4.0, 225)
    72
    >>> token_budget(30.0, 225)
    225
    """
    return min(cap, int(seconds * TOKENS_PER_SECOND) + TOKEN_MARGIN)


def _batches(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        language: str | None = None,
        device: str | None = None,
        num_beams: int = 1,
        max_new_tokens: int = 225,
        adapter: str | None = None,
    ):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
        if adapter:  # a LoRA adapter folder from scripts/train_lora.py
            from peft import PeftModel

            from .finetune import read_train_config
            trained = read_train_config(adapter) or {}
            if language is None and trained.get("language"):
                # Decode with the same stand-in token the adapter was trained with.
                language = trained["language"]
                log.info("using language token %r from the adapter's train_config", language)
            elif language and trained.get("language") not in (None, language):
                log.warning("adapter was trained with language %r but decoding with %r",
                            trained["language"], language)
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.dtype = dtype
        self.language = language
        self.gen_kwargs = {"task": "transcribe", "num_beams": num_beams,
                           "max_new_tokens": max_new_tokens}
        if language:
            self.gen_kwargs["language"] = language
        if self.device == "cpu":
            log.warning("running on CPU: fine for a smoke test, far too slow for a full test set")

    def transcribe_paths(self, paths: list[str], batch_size: int = 16) -> list[str]:
        import soundfile as sf

        out: list[str] = []
        for i, batch in enumerate(_batches(paths, batch_size)):
            audio = [sf.read(p, dtype="float32")[0] for p in batch]
            feats = self.processor(audio, sampling_rate=16_000, return_tensors="pt")
            inputs = feats.input_features.to(self.device, self.dtype)
            with self.torch.inference_mode():
                longest = max(len(a) for a in audio) / 16_000
            kwargs = {**self.gen_kwargs,
                      "max_new_tokens": token_budget(longest, self.gen_kwargs["max_new_tokens"])}
            ids = self.model.generate(inputs, **kwargs)
            out.extend(self.processor.batch_decode(ids, skip_special_tokens=True))
            if (i + 1) % 20 == 0:
                log.info("transcribed %d / %d", len(out), len(paths))
        return out
