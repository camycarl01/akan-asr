"""Google WAXAL, Akan ASR subset, from Hugging Face.

Dataset: https://huggingface.co/datasets/google/WaxalNLP  (config "aka_asr")
Files: data/ASR/aka/aka-{train,validation,test}-*.parquet, about 4 GB labelled.
Columns: id, speaker_id, transcription, language, gender, audio (MP3 bytes,
44.1 kHz stereo). Gender is often blank.
Licence: CC BY / CC BY-SA 4.0.

Caution: the University of Ghana collected WAXAL's Akan data with the same
picture-description method as UGSpeechData. Check the duplicate report from
scripts/make_splits.py before treating them as two separate domains, or your
"cross-domain" result may just be the same recordings counted twice.

The "aka" ASR data is not only Twi: some transcripts are written in Fante
(e.g. "dz" spellings: "woridzi", "Adɔkɔdɔkɔdzi"). There is no dialect label,
so every clip gets dialect "akan".

We read the parquet files directly rather than through `datasets.load_dataset`:
resolving this repo (hundreds of files across languages) is very slow, and
audio decoding there depends on the installed `datasets` version.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ..audio import TARGET_SR
from ..manifest import make_manifest, normalize_gender

log = logging.getLogger(__name__)

REPO = "google/WaxalNLP"
CONFIG = "aka_asr"
LABELLED_SPLITS = ("train", "validation", "test")


def _rows(lang: str, split: str, limit: int | None) -> Iterator[dict]:
    """Yield raw rows of one split, one parquet row group at a time.

    With a limit, read remotely so only the first row group is fetched;
    otherwise download each file once into the Hugging Face cache.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, hf_hub_download

    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{REPO}/data/ASR/{lang}/{lang}-{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet files for {lang!r} in {REPO}")
    n = 0
    for remote in files:
        if limit:
            source = fs.open(remote, block_size=4 * 2**20)
        else:
            source = hf_hub_download(REPO, remote.split(f"{REPO}/", 1)[1], repo_type="dataset")
        pf = pq.ParquetFile(source)
        for group in range(pf.num_row_groups):
            for row in pf.read_row_group(group).to_pylist():
                yield row
                n += 1
                if limit and n >= limit:
                    return


def _write_wav(audio_bytes: bytes, dst: Path) -> float:
    """Decode MP3 bytes to 16 kHz mono WAV (skipped if it exists); return seconds."""
    import librosa
    import soundfile as sf

    if dst.exists() and dst.stat().st_size > 44:
        return float(sf.info(str(dst)).duration)
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, mono=True)
    if audio.size == 0:
        raise ValueError("decoded audio is empty")
    tmp = dst.with_name(dst.name + ".part")
    sf.write(str(tmp), audio, TARGET_SR, subtype="PCM_16", format="WAV")
    tmp.replace(dst)
    return len(audio) / TARGET_SR


def load(out_dir: str | Path, config: str = CONFIG, max_per_split: int | None = None
         ) -> pd.DataFrame:
    """Download the labelled splits and write each clip as 16 kHz mono WAV.

    Keeps WAXAL's own split in an `official_split` column; scripts/make_splits.py
    re-splits by speaker.
    """
    lang = config.split("_")[0]
    out_dir = Path(out_dir)
    rows, failed = [], 0
    for split in LABELLED_SPLITS:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        n_split = 0
        for ex in _rows(lang, split, max_per_split):
            text = (ex.get("transcription") or "").strip()
            audio = ex.get("audio") or {}
            if not text or not audio.get("bytes"):
                continue
            wav = split_dir / f"{ex['id']}.wav"
            try:
                duration = _write_wav(audio["bytes"], wav)
            except Exception as exc:  # one bad clip shouldn't stop a 4 GB run
                log.warning("waxal/%s: could not decode %s: %s", split, ex["id"], exc)
                failed += 1
                continue
            rows.append({
                "utt_id": f"waxal:{ex['id']}",
                "audio_path": str(wav),
                "text": text,
                "speaker_id": f"waxal:{ex['speaker_id']}",
                "dataset": "waxal",
                "dialect": "akan",
                "gender": normalize_gender(ex.get("gender")),
                "duration_s": duration,
                "official_split": split,
            })
            n_split += 1
            if n_split % 2000 == 0:
                log.info("waxal/%s: %d clips written", split, n_split)
        log.info("waxal/%s: %d utterances", split, n_split)
    if failed:
        log.warning("waxal: %d clips failed to decode and were skipped", failed)
    df = make_manifest(rows)
    df["official_split"] = [r["official_split"] for r in rows]
    return df
