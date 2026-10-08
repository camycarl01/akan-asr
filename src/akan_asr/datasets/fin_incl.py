"""Ashesi Financial Inclusion Speech Dataset (Asante, Akuapem, Fante; also Ga).

Source: https://github.com/Ashesi-Org/Financial-Inclusion-Speech-Dataset
Downloads: https://adr.ashesi.edu.gh/datasets/10 (Asante), /12 (Akuapem), /13 (Fante)
Licence: CC BY 4.0.

Facts from the README that shape this loader:
- Each folder has a data.csv with audio path, transcription, English translation.
- Ignore the first two directories in the stored path, so we match on filename.
- Filenames encode the speaker: e.g. GaFm21-ATuJLn5X-Tmp083-zykm34.ogg
  -> language/gender/age prefix "GaFm21", speaker id "ATuJLn5X".
- The 10p/90p archives are NOT a clean test split: the same speakers and the
  same sentences appear in both. We merge them and make our own split.
- Every speaker read the same ~130 sentences, so even a speaker-disjoint test
  set shares sentences with training. Report that (see splits.text_leakage).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from ..manifest import make_manifest, normalize_gender

log = logging.getLogger(__name__)

DIALECTS = {"asante", "akuapem", "fante"}
_FILENAME_RE = re.compile(
    r"^(?P<prefix>[A-Za-z]+?)(?P<age>\d{1,3})-(?P<speaker>[A-Za-z0-9]+)-"
)
_GENDER_SUFFIXES = (("Fm", "female"), ("Ml", "male"), ("F", "female"), ("M", "male"))
AUDIO_EXTS = {".ogg", ".wav", ".mp3", ".flac", ".m4a"}


def parse_filename(name: str) -> dict:
    """Extract speaker id, gender and age from a dataset filename."""
    m = _FILENAME_RE.match(Path(name).name)
    if not m:
        return {"speaker": None, "gender": "unknown", "age": None}
    prefix = m.group("prefix")
    gender = "unknown"
    for suffix, label in _GENDER_SUFFIXES:
        if prefix.endswith(suffix) and len(prefix) > len(suffix):
            gender = label
            break
    return {"speaker": m.group("speaker"), "gender": gender, "age": int(m.group("age"))}


def _pick_column(columns, *needles: str) -> str:
    lowered = {c: c.lower() for c in columns}
    for needle in needles:
        for col, low in lowered.items():
            if needle in low:
                return col
    raise KeyError(
        f"No column matching {needles} in data.csv. Columns are: {list(columns)}. "
        "Pass the right name with --path-col / --text-col."
    )


def load(
    root: str | Path,
    dialect: str,
    path_col: str | None = None,
    text_col: str | None = None,
) -> pd.DataFrame:
    """Build a manifest from one extracted dialect folder (10p and 90p together)."""
    dialect = dialect.lower()
    if dialect not in DIALECTS:
        raise ValueError(f"dialect must be one of {sorted(DIALECTS)}, got {dialect!r}")
    root = Path(root)

    csvs = sorted(root.rglob("data.csv"))
    if not csvs:
        raise FileNotFoundError(f"No data.csv found under {root}")
    audio_index = {
        p.name: p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS
    }
    if not audio_index:
        raise FileNotFoundError(f"No audio files found under {root}")

    rows, missing = [], 0
    for csv in csvs:
        meta = pd.read_csv(csv)
        pcol = path_col or _pick_column(meta.columns, "path", "file", "audio")
        tcol = text_col or _pick_column(meta.columns, "transcri", "text", "sentence")
        for raw_path, text in zip(meta[pcol], meta[tcol]):
            fname = Path(str(raw_path)).name
            audio = audio_index.get(fname)
            if audio is None or not isinstance(text, str) or not text.strip():
                missing += 1
                continue
            info = parse_filename(fname)
            if info["speaker"] is None:
                missing += 1
                continue
            rows.append({
                "utt_id": f"fin_incl:{dialect}:{Path(fname).stem}",
                "audio_path": str(audio),
                "text": text.strip(),
                "speaker_id": f"fin_incl:{info['speaker']}",
                "dataset": "fin_incl",
                "dialect": dialect,
                "gender": normalize_gender(info["gender"]),
            })
    if missing:
        log.warning("fin_incl/%s: skipped %d rows (no audio, no text or unparsable name)",
                    dialect, missing)
    df = make_manifest(rows)
    return df.drop_duplicates("utt_id").reset_index(drop=True)
