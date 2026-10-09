"""Ashesi Financial Inclusion Speech Dataset (Asante, Akuapem, Fante; also Ga).

Source: https://github.com/Ashesi-Org/Financial-Inclusion-Speech-Dataset
Downloads: https://adr.ashesi.edu.gh/datasets/10 (Asante), /12 (Akuapem), /13 (Fante)
Licence: CC BY 4.0.

Facts from the README that shape this loader:
- Each folder has a data.csv with audio path, transcription, English translation.
- Ignore the first two directories in the stored path, so we match on filename.
- Filenames encode the speaker: e.g. AsantiTwiFm23-MMuHe3cd-Tmp033-90pZBJ.ogg
  -> language "AsantiTwi", gender "Fm" (female; "Ma" = male), age 23,
     speaker id "MMuHe3cd", prompt "Tmp033".
- Extracted archives are named fisd-<dialect>-10p / fisd-<dialect>-90p, each
  with a data.csv and an audios/ folder. Put both inside one dialect folder.
- The 10p/90p archives are NOT a clean test split: the same speakers and the
  same sentences appear in both. We merge them and make our own split.
- Every speaker read the same ~130 sentences, so even a speaker-disjoint test
  set shares sentences with training. Report that (see splits.text_leakage).
- A few Twi prompts carry reading notes, not speech (see clean_prompt).
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
    # Speaker ids are usually 8 alphanumerics but can contain spaces
    # (seen: "AsantiTwiFm20-A SLRKMb-Tmp010-..."), so accept anything but "-".
    r"^(?P<prefix>[A-Za-z]+?)(?P<age>\d{1,3})-(?P<speaker>[^-]+?)-Tmp\d+"
)
# Observed in the real filenames: "Fm" = female, "Ma" = male
# (AsantiTwiFm23-..., AkuapemTwiMa24-..., GaMa22-...).
_GENDER_SUFFIXES = (("Fm", "female"), ("Ma", "male"), ("Ml", "male"),
                    ("F", "female"), ("M", "male"))
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


# "(spoken)", "(spoken - informal)", "(spoken. Written: Mepawokyew)": a note on
# the text before it, which is what the speaker said.
_SPOKEN_NOTE_RE = re.compile(r"\(\s*spoken\b[^)]*\)", re.IGNORECASE)


def clean_prompt(text: str) -> str | None:
    """Remove reading notes from a prompt; None if what was said is unknowable.

    Prompts offering alternatives ("X (informal) / Y (formal)", "A/ B") are
    dropped: we can't tell which one the speaker read. Seen in 8 of ~130
    prompts per Twi dialect (2-4% of clips), none in Fante.

    >>> clean_prompt("Mepaa’kyɛw  (spoken. Written: Mepawokyew)")
    'Mepaa’kyɛw'
    >>> clean_prompt("Nnipa yɛ bad (informal) / Nnipa nnyɛ (formal)") is None
    True
    >>> clean_prompt("Mepɛ sɛ, metua ka")
    'Mepɛ sɛ, metua ka'
    """
    text = _SPOKEN_NOTE_RE.sub(" ", text)
    if re.search(r"[/()]", text):
        return None
    text = " ".join(text.split())
    return text or None


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


_CANDIDATE_SEPS = ("\t", "|", ";", ",")


def read_metadata(csv: str | Path) -> pd.DataFrame:
    """Read a data.csv whose separator is not documented.

    The Ashesi files are not plain comma-separated: transcripts contain commas.
    Try each separator and keep the one that gives a consistent table with at
    least 2 columns (path + transcription). Handles a UTF-8 BOM too.
    """
    errors = []
    for sep in _CANDIDATE_SEPS:
        try:
            df = pd.read_csv(csv, sep=sep, encoding="utf-8-sig", dtype=str,
                             keep_default_na=False)
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            errors.append(f"{sep!r}: {exc.__class__.__name__}")
            continue
        if df.shape[1] >= 2:
            log.info("%s: separator %r, columns %s", csv, sep, list(df.columns))
            return df
        errors.append(f"{sep!r}: only {df.shape[1]} column")
    raise ValueError(
        f"Could not parse {csv} with any of {_CANDIDATE_SEPS}. Attempts: {errors}. "
        "Run `head -3` on the file and share the output."
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

    rows, missing, ambiguous = [], 0, 0
    for csv in csvs:
        meta = read_metadata(csv)
        pcol = path_col or _pick_column(meta.columns, "path", "file", "audio")
        tcol = text_col or _pick_column(meta.columns, "transcri", "text", "sentence")
        for raw_path, text in zip(meta[pcol], meta[tcol]):
            fname = Path(str(raw_path)).name
            audio = audio_index.get(fname)
            if audio is None or not isinstance(text, str) or not text.strip():
                missing += 1
                continue
            text = clean_prompt(text)
            if text is None:
                ambiguous += 1
                continue
            info = parse_filename(fname)
            if info["speaker"] is None:
                missing += 1
                continue
            rows.append({
                "utt_id": f"fin_incl:{dialect}:{Path(fname).stem}",
                "audio_path": str(audio),
                "text": text,
                "speaker_id": f"fin_incl:{info['speaker']}",
                "dataset": "fin_incl",
                "dialect": dialect,
                "gender": normalize_gender(info["gender"]),
            })
    if missing:
        log.warning("fin_incl/%s: skipped %d rows (no audio, no text or unparsable name)",
                    dialect, missing)
    if ambiguous:
        log.warning("fin_incl/%s: dropped %d rows whose prompt offers alternatives",
                    dialect, ambiguous)
    df = make_manifest(rows)
    return df.drop_duplicates("utt_id").reset_index(drop=True)
