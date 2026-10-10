"""WER and CER with bootstrap confidence intervals.

Corpus WER = total word errors / total reference words (not the mean of
per-utterance WERs, which over-weights short clips). The 95% interval comes
from resampling utterances with replacement, which is how you tell whether a
2-point WER gap between two models is real or noise.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import jiwer
import numpy as np

from .text import normalize

log = logging.getLogger(__name__)


@dataclass
class ErrorRate:
    value: float
    ci_low: float
    ci_high: float
    errors: int
    ref_units: int


@dataclass
class EvalResult:
    n_utterances: int
    n_skipped_empty_refs: int
    wer: ErrorRate
    cer: ErrorRate

    def to_dict(self) -> dict:
        return asdict(self)


def _per_utt(refs: list[str], hyps: list[str], unit: str) -> tuple[np.ndarray, np.ndarray]:
    errors, lengths = [], []
    for r, h in zip(refs, hyps):
        out = jiwer.process_words(r, h) if unit == "word" else jiwer.process_characters(r, h)
        errors.append(out.substitutions + out.deletions + out.insertions)
        lengths.append(out.substitutions + out.deletions + out.hits)
    return np.asarray(errors), np.asarray(lengths)


def _bootstrap(errors: np.ndarray, lengths: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(errors), size=(n, len(errors)))
    rates = errors[idx].sum(axis=1) / np.maximum(lengths[idx].sum(axis=1), 1)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi)


def score(
    references: list[str],
    hypotheses: list[str],
    n_bootstrap: int = 1000,
    seed: int = 0,
    already_normalized: bool = False,
) -> EvalResult:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses differ in length")
    refs = references if already_normalized else [normalize(r) for r in references]
    hyps = hypotheses if already_normalized else [normalize(h) for h in hypotheses]

    keep = [i for i, r in enumerate(refs) if r]
    skipped = len(refs) - len(keep)
    if skipped:
        log.warning("skipping %d utterances whose reference is empty after normalising",
                    skipped)
    if not keep:
        raise ValueError("no non-empty references to score")
    refs = [refs[i] for i in keep]
    hyps = [hyps[i] for i in keep]

    def rate(unit: str, ref_list, hyp_list) -> ErrorRate:
        e, l = _per_utt(ref_list, hyp_list, unit)
        lo, hi = _bootstrap(e, l, n_bootstrap, seed)
        return ErrorRate(float(e.sum() / l.sum()), lo, hi, int(e.sum()), int(l.sum()))

    return EvalResult(
        n_utterances=len(refs),
        n_skipped_empty_refs=skipped,
        wer=rate("word", refs, hyps),
        # CER ignores spaces so word-boundary mistakes are not double-counted.
        cer=rate("char", [r.replace(" ", "") for r in refs],
                 [h.replace(" ", "") for h in hyps]),
    )


@dataclass
class PairedDifference:
    """WER(A) - WER(B) on the same utterances; negative means A is better."""
    n_utterances: int
    wer_a: float
    wer_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float  # two-sided: how often the resampled difference crosses zero

    def to_dict(self) -> dict:
        return asdict(self)


def paired_difference(
    references: list[str],
    hypotheses_a: list[str],
    hypotheses_b: list[str],
    n_bootstrap: int = 10_000,
    seed: int = 0,
    unit: str = "word",
) -> PairedDifference:
    """Paired bootstrap test for two systems scored on the same utterances.

    Each resample draws the same utterances for both systems, so per-utterance
    difficulty cancels out. This is far more sensitive than checking whether
    two separate confidence intervals overlap. Inputs must already be
    normalised and aligned (same order, same utterances).
    """
    if not (len(references) == len(hypotheses_a) == len(hypotheses_b)):
        raise ValueError("references and both hypothesis lists must be the same length")
    keep = [i for i, r in enumerate(references) if r]
    refs = [references[i] for i in keep]
    e_a, lengths = _per_utt(refs, [hypotheses_a[i] for i in keep], unit)
    e_b, _ = _per_utt(refs, [hypotheses_b[i] for i in keep], unit)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(refs), size=(n_bootstrap, len(refs)))
    total = np.maximum(lengths[idx].sum(axis=1), 1)
    diffs = (e_a[idx].sum(axis=1) - e_b[idx].sum(axis=1)) / total
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    wer_a, wer_b = e_a.sum() / lengths.sum(), e_b.sum() / lengths.sum()
    return PairedDifference(len(refs), float(wer_a), float(wer_b), float(wer_a - wer_b),
                            float(lo), float(hi), float(min(p, 1.0)))
