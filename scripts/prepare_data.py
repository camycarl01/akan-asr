"""Turn each raw dataset into a 16 kHz manifest.

Examples (run from the repo root):

  # Ashesi Financial Inclusion, one call per dialect folder you extracted
  python scripts/prepare_data.py fin_incl --root raw/fin_incl/asante  --dialect asante
  python scripts/prepare_data.py fin_incl --root raw/fin_incl/akuapem --dialect akuapem
  python scripts/prepare_data.py fin_incl --root raw/fin_incl/fante   --dialect fante

  # WAXAL Akan (downloads from Hugging Face; already 16 kHz)
  python scripts/prepare_data.py waxal

  # UGSpeechData, once you've found the transcript file
  python scripts/prepare_data.py ugspeech --root raw/ugspeech/Akan \
      --transcripts raw/ugspeech/Akan/<file>.csv \
      --transcript-audio-col <col> --transcript-text-col <col>

Outputs go to data/manifests/<name>.csv and audio to data/audio/<name>/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import sys
from pathlib import Path

# Let scripts run without `pip install -e .` (e.g. a fresh terminal or Kaggle).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akan_asr import manifest
from akan_asr.audio import convert_manifest
from akan_asr.text import akan_charset

log = logging.getLogger("prepare")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["fin_incl", "waxal", "ugspeech"])
    ap.add_argument("--root", type=Path, help="extracted dataset folder")
    ap.add_argument("--dialect", help="fin_incl only: asante | akuapem | fante")
    ap.add_argument("--path-col"); ap.add_argument("--text-col")
    ap.add_argument("--transcripts", type=Path)
    ap.add_argument("--transcript-audio-col"); ap.add_argument("--transcript-text-col")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N rows, for a quick test")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.dataset == "fin_incl":
        if not (args.root and args.dialect):
            ap.error("fin_incl needs --root and --dialect")
        from akan_asr.datasets import fin_incl
        df = fin_incl.load(args.root, args.dialect, args.path_col, args.text_col)
        name = f"fin_incl_{args.dialect}"
    elif args.dataset == "waxal":
        from akan_asr.datasets import waxal
        name = "waxal"
        df = waxal.load(args.out / "audio" / name, max_per_split=args.limit,
                        workers=args.workers)
    else:
        need = [args.root, args.transcripts, args.transcript_audio_col, args.transcript_text_col]
        if not all(need):
            ap.error("ugspeech needs --root, --transcripts, --transcript-audio-col, "
                     "--transcript-text-col")
        from akan_asr.datasets import ugspeech
        df = ugspeech.load(args.root, args.transcripts,
                           args.transcript_audio_col, args.transcript_text_col)
        name = "ugspeech"

    if args.limit and args.dataset != "waxal":  # WAXAL already limits per split
        df = df.head(args.limit)

    if args.dataset != "waxal":  # WAXAL is written at 16 kHz during download
        df, failures = convert_manifest(df, args.out / "audio" / name, args.workers)
        if len(failures):
            # Not in manifests/: make_splits.py globs that folder for manifests.
            fail_path = args.out / "failures" / f"{name}.csv"
            fail_path.parent.mkdir(parents=True, exist_ok=True)
            failures.to_csv(fail_path, index=False)
            log.warning("%d files failed to convert; listed in %s", len(failures), fail_path)

    path = manifest.save(df, args.out / "manifests" / f"{name}.csv")
    print(f"\nSaved {path}")
    print(manifest.summarize(df).to_string())
    print("\nCharacters used after normalising (look for anything odd):")
    print("".join(sorted(akan_charset(df["text"]))))


if __name__ == "__main__":
    main()
