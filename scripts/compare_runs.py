"""Paired significance test between two evaluated models.

  python scripts/compare_runs.py results/both-best results/ashesi-only-best

For every test set both folders have (<name>_predictions.csv from
run_baseline.py), matches utterances by utt_id and reports WER(A) - WER(B)
with a 95% paired bootstrap interval and p-value, overall and per dialect.
Negative difference = the first model is better.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Let scripts run without `pip install -e .` (e.g. a fresh terminal or Kaggle).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akan_asr.metrics import paired_difference


def _compare(name: str, subset: str, df: pd.DataFrame, n_bootstrap: int) -> dict:
    res = paired_difference(df["reference_norm"].tolist(), df["prediction_norm_a"].tolist(),
                            df["prediction_norm_b"].tolist(), n_bootstrap=n_bootstrap)
    return {"test_set": name, "subset": subset, "n": res.n_utterances,
            "WER A %": round(100 * res.wer_a, 2), "WER B %": round(100 * res.wer_b, 2),
            "A - B": round(100 * res.diff, 2),
            "95% CI": f"[{100 * res.ci_low:.2f}, {100 * res.ci_high:.2f}]",
            "p": round(res.p_value, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    args = ap.parse_args()

    rows = []
    for pa in sorted(args.run_a.glob("*_predictions.csv")):
        pb = args.run_b / pa.name
        if not pb.exists():
            continue
        name = pa.name.removesuffix("_predictions.csv")
        a = pd.read_csv(pa, keep_default_na=False)
        b = pd.read_csv(pb, keep_default_na=False)
        df = a.merge(b[["utt_id", "prediction_norm"]], on="utt_id", suffixes=("_a", "_b"))
        if len(df) < len(a) or len(df) < len(b):
            print(f"{name}: comparing the {len(df)} utterances both runs scored "
                  f"(A has {len(a)}, B has {len(b)})")
        rows.append(_compare(name, "all", df, args.n_bootstrap))
        if df["dialect"].nunique() > 1:
            for dialect, part in df.groupby("dialect"):
                rows.append(_compare(name, dialect, part, args.n_bootstrap))
    if not rows:
        sys.exit(f"no test sets in common between {args.run_a} and {args.run_b}")
    print(f"A = {args.run_a.name}   B = {args.run_b.name}   (negative A - B: A is better)")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
