"""UGSpeechData (University of Ghana HCI Lab), Akan subset.

Data:    https://doi.org/10.57760/sciencedb.22298  (Science Data Bank)
Repo:    https://github.com/HCI-LAB-UGSPEECHDATA/speech_data_ghana_ug
Licence: CC BY-NC-ND 4.0 -> do not publish a model trained on it until the lab
         confirms that is allowed (that is what your email asked).

AUDIO_ID.csv documents these columns: IMAGE_URL, IMAGE_SRC_URL, AUDIO_URL,
ORG_NAME, PROJECT_NAME, SPEAKER_ID, LOCALE, GENDER, AGE, DEVICE, ENVIRONMENT,
YEAR. Akan rows have LOCALE == "ak_gh".

The README does not document where the ~100 transcribed hours' text lives, so
this loader takes the transcript file and its column names as arguments. Open
the download, find the transcript file, then run:

    python scripts/prepare_ugspeech.py --root <folder> --transcripts <file> \
        --transcript-audio-col <col> --transcript-text-col <col>

If the columns are wrong, the error lists the real ones.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..manifest import make_manifest, normalize_gender

log = logging.getLogger(__name__)

AKAN_LOCALE = "ak_gh"
AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _require(df: pd.DataFrame, cols: list[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{what} has no column(s) {missing}. Columns: {list(df.columns)}")


def load(
    root: str | Path,
    transcripts: str | Path,
    transcript_audio_col: str,
    transcript_text_col: str,
    audio_id_csv: str | Path | None = None,
) -> pd.DataFrame:
    root = Path(root)
    audio_index = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS}
    if not audio_index:
        raise FileNotFoundError(f"No audio files under {root}")

    tr = _read_table(Path(transcripts))
    _require(tr, [transcript_audio_col, transcript_text_col], "Transcript file")
    tr = tr[[transcript_audio_col, transcript_text_col]].dropna()
    tr["stem"] = tr[transcript_audio_col].map(lambda s: Path(str(s)).stem)

    meta_path = Path(audio_id_csv) if audio_id_csv else next(root.rglob("AUDIO_ID.csv"), None)
    meta = None
    if meta_path is not None and meta_path.exists():
        meta = _read_table(meta_path)
        _require(meta, ["AUDIO_URL", "SPEAKER_ID"], "AUDIO_ID.csv")
        if "LOCALE" in meta.columns:
            meta = meta[meta["LOCALE"].astype(str).str.lower() == AKAN_LOCALE]
        meta = meta.assign(stem=meta["AUDIO_URL"].map(lambda s: Path(str(s)).stem))
        meta = meta.drop_duplicates("stem").set_index("stem")
    else:
        log.warning("AUDIO_ID.csv not found: speaker ids unavailable, "
                    "speaker-disjoint splits will be impossible")

    rows, no_audio, no_speaker = [], 0, 0
    for stem, text in zip(tr["stem"], tr[transcript_text_col]):
        audio = audio_index.get(stem)
        if audio is None:
            no_audio += 1
            continue
        speaker, gender = None, None
        if meta is not None and stem in meta.index:
            speaker = meta.at[stem, "SPEAKER_ID"]
            gender = meta.at[stem, "GENDER"] if "GENDER" in meta.columns else None
        if speaker is None or pd.isna(speaker):
            no_speaker += 1
            continue
        rows.append({
            "utt_id": f"ugspeech:{stem}",
            "audio_path": str(audio),
            "text": str(text).strip(),
            "speaker_id": f"ugspeech:{speaker}",
            "dataset": "ugspeech",
            "dialect": "akan",
            "gender": normalize_gender(gender),
        })
    if no_audio or no_speaker:
        log.warning("ugspeech: dropped %d rows with no audio file, %d with no speaker id",
                    no_audio, no_speaker)
    if not rows:
        raise ValueError("No usable rows. Check the audio column matches the audio filenames.")
    return make_manifest(rows).drop_duplicates("utt_id").reset_index(drop=True)
