"""Google WAXAL, Akan ASR subset, from Hugging Face.

Dataset: https://huggingface.co/datasets/google/WaxalNLP  (config "aka_asr")
Columns: id, speaker_id, transcription, language, gender, audio.
Splits: train / validation / test (transcribed) and unlabeled (no text).
Licence: CC BY / CC BY-SA 4.0.

Caution: the University of Ghana collected WAXAL's Akan data with the same
picture-description method as UGSpeechData. Run scripts/check_overlap.py before
treating them as two separate domains, or your "cross-domain" result may just
be the same recordings counted twice.

Note: WAXAL has Fante only as text-to-speech data, not ASR.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..audio import TARGET_SR
from ..manifest import make_manifest, normalize_gender

log = logging.getLogger(__name__)

REPO = "google/WaxalNLP"
CONFIG = "aka_asr"
LABELLED_SPLITS = ("train", "validation", "test")


def load(out_dir: str | Path, config: str = CONFIG, max_per_split: int | None = None
         ) -> pd.DataFrame:
    """Download the labelled splits and write each clip as 16 kHz mono WAV.

    Keeps WAXAL's own split in an `official_split` column; scripts/make_splits.py
    can either respect it or re-split by speaker.
    """
    import soundfile as sf
    from datasets import Audio, load_dataset

    out_dir = Path(out_dir)
    rows = []
    for split in LABELLED_SPLITS:
        ds = load_dataset(REPO, config, split=split)
        ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
        if max_per_split:
            ds = ds.select(range(min(max_per_split, len(ds))))
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for ex in ds:
            text = (ex.get("transcription") or "").strip()
            if not text:
                continue
            wav = split_dir / f"{ex['id']}.wav"
            arr = ex["audio"]["array"]
            if not wav.exists():
                sf.write(str(wav), arr, TARGET_SR, subtype="PCM_16")
            rows.append({
                "utt_id": f"waxal:{ex['id']}",
                "audio_path": str(wav),
                "text": text,
                "speaker_id": f"waxal:{ex['speaker_id']}",
                "dataset": "waxal",
                "dialect": "akan",
                "gender": normalize_gender(ex.get("gender")),
                "duration_s": len(arr) / TARGET_SR,
                "official_split": split,
            })
        log.info("waxal/%s: %d utterances", split, len(ds))
    df = make_manifest(rows)
    df["official_split"] = [r["official_split"] for r in rows]
    return df
