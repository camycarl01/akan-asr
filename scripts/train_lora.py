"""Train a Whisper LoRA adapter on one or more datasets.

  python scripts/train_lora.py data/splits/fin_incl_train.csv \
      --val data/splits/fin_incl_validation.csv data/splits/waxal_validation.csv

Validate on every dataset you will test on, not only the ones you train on:
the point is to see the cross-domain number move during training.

Writes under adapters/<run-name>/:
  best/          adapter with the lowest mean validation WER (+ train_config.json)
  last/          adapter at the final step
  history.csv    validation loss and WER per dataset at each evaluation
Then score it exactly like the baseline:
  python scripts/run_baseline.py data/splits/*_test.csv --adapter adapters/<run-name>/best
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Let scripts run without `pip install -e .` (e.g. a fresh terminal or Kaggle).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from akan_asr.finetune import TrainConfig, train


def _default_name(args) -> str:
    datasets = sorted({p.stem.removesuffix("_train") for p in args.train_manifests})
    return f"{args.model.split('/')[-1]}-{'+'.join(datasets)}-r{args.rank}"


def main() -> None:
    d = TrainConfig([], [], "")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train_manifests", nargs="+", type=Path)
    ap.add_argument("--val", nargs="+", type=Path, required=True,
                    help="validation manifests (include other domains too)")
    ap.add_argument("--model", default=d.model)
    ap.add_argument("--language", default=d.language,
                    help="Whisper language token used as a stand-in for Akan")
    ap.add_argument("--rank", type=int, default=d.rank)
    ap.add_argument("--alpha", type=int, default=d.alpha)
    ap.add_argument("--dropout", type=float, default=d.dropout)
    ap.add_argument("--target-modules", nargs="+", default=d.target_modules)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--epochs", type=float, default=d.epochs)
    ap.add_argument("--max-steps", type=int, help="overrides --epochs")
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--grad-accum", type=int, default=d.grad_accum)
    ap.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    ap.add_argument("--eval-every", type=int, default=d.eval_every)
    ap.add_argument("--val-utts", type=int, default=d.val_utts)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--limit", type=int, help="random N training clips (smoke test)")
    ap.add_argument("--balance", action="store_true",
                    help="draw each training dataset equally often, not by size")
    ap.add_argument("--num-workers", type=int, default=d.num_workers)
    ap.add_argument("--run-name")
    ap.add_argument("--adapters", type=Path, default=Path("adapters"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = TrainConfig(
        train_manifests=[str(p) for p in args.train_manifests],
        val_manifests=[str(p) for p in args.val],
        out_dir=str(args.adapters / (args.run_name or _default_name(args))),
        model=args.model, language=args.language, rank=args.rank, alpha=args.alpha,
        dropout=args.dropout, target_modules=args.target_modules, lr=args.lr,
        epochs=args.epochs, max_steps=args.max_steps, batch_size=args.batch_size,
        grad_accum=args.grad_accum, warmup_steps=args.warmup_steps,
        eval_every=args.eval_every, val_utts=args.val_utts, seed=args.seed,
        limit=args.limit, num_workers=args.num_workers, balance=args.balance,
    )
    train(cfg)


if __name__ == "__main__":
    main()
