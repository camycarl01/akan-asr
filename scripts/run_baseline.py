"""Zero-shot (or fine-tuned) Whisper evaluation on every test set.

  python scripts/run_baseline.py data/splits/*_test.csv --model openai/whisper-small

For each test file it writes, under results/<run-name>/:
  <dataset>_predictions.csv   reference, prediction and per-utterance WER
  <dataset>_metrics.json      WER/CER with 95% bootstrap intervals
and prints one summary table. Later runs (fine-tuned models, LoRA ranks)
use the same script with --adapter, so every number in your report is
produced the same way.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import jiwer
import pandas as pd

from akan_asr import manifest
from akan_asr.metrics import score
from akan_asr.text import normalize

log = logging.getLogger("baseline")
WHISPER_MAX_S = 30.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test_manifests", nargs="+", type=Path)
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--adapter", help="LoRA adapter folder (Week 2 onward)")
    ap.add_argument("--language", help="force a Whisper language token; default auto")
    ap.add_argument("--run-name", help="results subfolder; default derived from model")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--limit", type=int, help="first N utterances per test set (smoke test)")
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from akan_asr.transcribe import WhisperTranscriber

    run = args.run_name or "-".join(filter(None, [
        args.model.split("/")[-1],
        "lora" if args.adapter else "zeroshot",
        args.language,
    ]))
    out_dir = args.results / run
    out_dir.mkdir(parents=True, exist_ok=True)
    asr = WhisperTranscriber(args.model, language=args.language,
                             num_beams=args.num_beams, adapter=args.adapter)

    summary = []
    for path in args.test_manifests:
        df = manifest.load(path)
        name = path.stem.removesuffix("_test")
        too_long = df["duration_s"].notna() & (df["duration_s"] > WHISPER_MAX_S)
        if too_long.any():
            # Whisper sees only the first 30 s, so longer clips would be scored
            # as massive deletions. Exclude and report rather than hide it.
            log.warning("%s: excluding %d clips longer than %.0f s",
                        name, int(too_long.sum()), WHISPER_MAX_S)
            df = df[~too_long]
        if args.limit:
            df = df.head(args.limit)
        if df.empty:
            log.warning("%s: nothing to evaluate", name)
            continue

        preds = asr.transcribe_paths(df["audio_path"].tolist(), args.batch_size)
        res = score(df["text"].tolist(), preds)

        refs_n = [normalize(t) for t in df["text"]]
        hyps_n = [normalize(p) for p in preds]
        utt_wer = [jiwer.wer(r, h) if r else None for r, h in zip(refs_n, hyps_n)]
        pd.DataFrame({
            "utt_id": df["utt_id"], "dialect": df["dialect"], "gender": df["gender"],
            "reference": df["text"], "prediction": preds,
            "reference_norm": refs_n, "prediction_norm": hyps_n, "wer": utt_wer,
        }).to_csv(out_dir / f"{name}_predictions.csv", index=False)

        meta = {"model": args.model, "adapter": args.adapter, "language": args.language,
                "num_beams": args.num_beams, "test_manifest": str(path),
                "excluded_over_30s": int(too_long.sum()), **res.to_dict()}
        (out_dir / f"{name}_metrics.json").write_text(json.dumps(meta, indent=2))
        summary.append({
            "test_set": name, "n": res.n_utterances,
            "WER %": f"{100*res.wer.value:.1f} [{100*res.wer.ci_low:.1f}, {100*res.wer.ci_high:.1f}]",
            "CER %": f"{100*res.cer.value:.1f} [{100*res.cer.ci_low:.1f}, {100*res.cer.ci_high:.1f}]",
        })

    table = pd.DataFrame(summary)
    table.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n{run}  (95% bootstrap intervals in brackets)")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
