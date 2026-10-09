"""Combine manifests, check cross-dataset duplicates, and make speaker-disjoint splits.

  python scripts/make_splits.py data/manifests/fin_incl_*.csv data/manifests/waxal.csv \
      [data/manifests/ugspeech.csv]

Writes data/splits/all.csv (every row with a `split` column) plus one
<dataset>_<split>.csv per dataset and split, and prints:
  - hours / speakers per dataset and split
  - how many test sentences were already seen in training (text leakage)
  - WAXAL vs UGSpeechData duplicate recordings, if both are present
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys
from pathlib import Path

# Let scripts run without `pip install -e .` (e.g. a fresh terminal or Kaggle).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akan_asr import manifest
from akan_asr.splits import cross_dataset_overlap, speaker_split, text_leakage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--fractions", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--drop-waxal-duplicates", action="store_true",
                    help="remove WAXAL clips that duplicate UGSpeechData clips")
    args = ap.parse_args()

    df = pd.concat([manifest.load(p) for p in args.manifests], ignore_index=True)
    manifest.validate(df)

    datasets = set(df["dataset"])
    if {"waxal", "ugspeech"} <= datasets:
        dup = cross_dataset_overlap(df[df.dataset == "waxal"], df[df.dataset == "ugspeech"])
        n_wax = (df.dataset == "waxal").sum()
        print(f"\nWAXAL clips whose transcript exactly matches a UGSpeechData clip: "
              f"{dup['utt_id'].nunique()} of {n_wax}")
        if len(dup):
            args.out.mkdir(parents=True, exist_ok=True)
            dup.to_csv(args.out / "waxal_ugspeech_duplicates.csv", index=False)
            if args.drop_waxal_duplicates:
                df = df[~df["utt_id"].isin(dup["utt_id"])].reset_index(drop=True)
                print("Dropped them from WAXAL.")
            else:
                print("Kept them. Re-run with --drop-waxal-duplicates before "
                      "treating WAXAL and UGSpeechData as separate domains.")

    split = speaker_split(df, tuple(args.fractions), seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.out / "all.csv", index=False)
    for (name, part_split), part in split.groupby(["dataset", "split"]):
        part.to_csv(args.out / f"{name}_{part_split}.csv", index=False)

    print("\nSplit summary:")
    print(manifest.summarize(split).to_string())
    print("\nHeld-out sentences already seen in training:")
    print(text_leakage(split).to_string(index=False))
    print(f"\nWrote splits to {args.out}/")


if __name__ == "__main__":
    main()
