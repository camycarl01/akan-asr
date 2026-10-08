"""Convert any input audio (mp3, ogg, wav at any rate) to 16 kHz mono 16-bit WAV.

Whisper and Wav2Vec2 both expect 16 kHz mono. Converting once up front, rather
than resampling inside every training step, saves a lot of GPU time on Kaggle.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

TARGET_SR = 16_000
log = logging.getLogger(__name__)


def convert_file(src: str | Path, dst: str | Path, sr: int = TARGET_SR) -> float:
    """Convert one file and return its duration in seconds.

    Skips work if `dst` already exists, so interrupted runs can be resumed.
    """
    import librosa
    import soundfile as sf

    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 44:
        return float(sf.info(str(dst)).duration)
    audio, _ = librosa.load(str(src), sr=sr, mono=True)
    if audio.size == 0:
        raise ValueError("decoded audio is empty")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    sf.write(str(tmp), audio, sr, subtype="PCM_16", format="WAV")
    tmp.replace(dst)  # atomic: a crash never leaves a half-written .wav behind
    return len(audio) / sr


def _target_path(out_dir: Path, utt_id: str) -> Path:
    safe = utt_id.replace(":", "__").replace("/", "_")
    return out_dir / f"{safe}.wav"


def _worker(args):
    src, dst = args
    try:
        return str(dst), convert_file(src, dst), None
    except Exception as exc:  # noqa: BLE001 - report every failure, keep going
        return str(dst), None, f"{type(exc).__name__}: {exc}"


def convert_manifest(
    df: pd.DataFrame, out_dir: str | Path, workers: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert every row's audio. Returns (converted_manifest, failures).

    Failed files are dropped from the returned manifest and listed in
    `failures` so you can inspect them instead of silently losing data.
    """
    out_dir = Path(out_dir)
    jobs = [(src, _target_path(out_dir, uid))
            for src, uid in zip(df["audio_path"], df["utt_id"])]
    results: dict[str, tuple[float | None, str | None]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            dst, dur, err = fut.result()
            results[dst] = (dur, err)
            if i % 2000 == 0:
                log.info("converted %d / %d", i, len(jobs))

    out = df.copy()
    out["audio_path"] = [str(dst) for _, dst in jobs]
    out["duration_s"] = [results[str(dst)][0] for _, dst in jobs]
    errors = [results[str(dst)][1] for _, dst in jobs]
    failed_mask = pd.Series([e is not None for e in errors], index=out.index)
    failures = df.loc[failed_mask, ["utt_id", "audio_path"]].copy()
    failures["error"] = [e for e in errors if e is not None]
    return out.loc[~failed_mask].reset_index(drop=True), failures
