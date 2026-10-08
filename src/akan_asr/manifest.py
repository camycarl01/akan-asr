"""The common manifest format every dataset is converted into.

One row per utterance. Everything downstream (splits, training, evaluation)
reads only this format, so adding a dataset means writing one loader.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "utt_id",       # globally unique: "<dataset>:<original id>"
    "audio_path",   # path to the audio file (16 kHz mono WAV after conversion)
    "text",         # raw transcript, exactly as the dataset provides it
    "speaker_id",   # globally unique: "<dataset>:<original speaker id>"
    "dataset",      # ugspeech | waxal | fin_incl | ...
    "dialect",      # asante | akuapem | fante | akan (unspecified)
    "gender",       # female | male | unknown
    "duration_s",   # float seconds, filled in by audio conversion
]

REQUIRED = ["utt_id", "audio_path", "text", "speaker_id", "dataset"]

_GENDER_MAP = {
    "f": "female", "female": "female", "fm": "female", "woman": "female",
    "m": "male", "male": "male", "ml": "male", "man": "male",
}


def normalize_gender(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    return _GENDER_MAP.get(str(value).strip().lower(), "unknown")


def make_manifest(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[COLUMNS]
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise with a clear message if the manifest would break later stages."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    for col in REQUIRED:
        n_null = int(df[col].isna().sum())
        if n_null:
            raise ValueError(f"Manifest column '{col}' has {n_null} empty values")
    dupes = df["utt_id"][df["utt_id"].duplicated()]
    if len(dupes):
        raise ValueError(
            f"{len(dupes)} duplicate utt_id values, e.g. {dupes.iloc[0]!r}"
        )


def save(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"utt_id": str, "speaker_id": str, "text": str})
    validate(df)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Hours, utterances and speakers per dataset (and split, if present)."""
    keys = ["dataset"] + (["split"] if "split" in df.columns else [])
    g = df.groupby(keys, dropna=False)
    out = pd.DataFrame({
        "utterances": g.size(),
        "speakers": g["speaker_id"].nunique(),
        "hours": g["duration_s"].sum(min_count=1) / 3600,
    })
    return out.round({"hours": 2})
