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


def _row_groups(lang: str, split: str, limit: int | None) -> Iterator[list[dict]]:
    """Yield the raw rows of one split, one parquet row group (~100 rows) at a time.

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
            rows = pf.read_row_group(group).to_pylist()
            if limit:
                rows = rows[:limit - n]
            yield rows
            n += len(rows)
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


def _decode_job(job: tuple[bytes, Path]) -> tuple[float | None, str | None]:
    """Run in a worker process: (duration, None) or (None, error message)."""
    audio_bytes, dst = job
    try:
        return _write_wav(audio_bytes, dst), None
    except Exception as exc:  # one bad clip shouldn't stop a 4 GB run
        return None, f"{type(exc).__name__}: {exc}"


def load(out_dir: str | Path, config: str = CONFIG, max_per_split: int | None = None,
         workers: int = 4) -> pd.DataFrame:
    """Download the labelled splits and write each clip as 16 kHz mono WAV.

    Keeps WAXAL's own split in an `official_split` column; scripts/make_splits.py
    re-splits by speaker.
    """
    from concurrent.futures import ProcessPoolExecutor

    lang = config.split("_")[0]
    out_dir = Path(out_dir)
    rows, failed = [], 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for split in LABELLED_SPLITS:
            split_dir = out_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            n_split = 0
            for group in _row_groups(lang, split, max_per_split):
                group = [ex for ex in group
                         if (ex.get("transcription") or "").strip()
                         and (ex.get("audio") or {}).get("bytes")]
                jobs = [(ex["audio"]["bytes"], split_dir / f"{ex['id']}.wav") for ex in group]
                for ex, (duration, err) in zip(group, pool.map(_decode_job, jobs)):
                    if err:
                        log.warning("waxal/%s: could not decode %s: %s", split, ex["id"], err)
                        failed += 1
                        continue
                    rows.append({
                        "utt_id": f"waxal:{ex['id']}",
                        "audio_path": str(split_dir / f"{ex['id']}.wav"),
                        "text": ex["transcription"].strip(),
                        "speaker_id": f"waxal:{ex['speaker_id']}",
                        "dataset": "waxal",
                        "dialect": "akan",
                        "gender": normalize_gender(ex.get("gender")),
                        "duration_s": duration,
                        "official_split": split,
                    })
                    n_split += 1
                if n_split and n_split % 2000 < len(group):
                    log.info("waxal/%s: %d clips written", split, n_split)
            log.info("waxal/%s: %d utterances", split, n_split)
    if failed:
        log.warning("waxal: %d clips failed to decode and were skipped", failed)
    df = make_manifest(rows)
    df["official_split"] = [r["official_split"] for r in rows]
    return df
