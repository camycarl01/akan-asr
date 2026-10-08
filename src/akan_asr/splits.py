"""Speaker-disjoint splits, leakage checks and cross-dataset overlap detection.

Why speaker-disjoint: if the same voice is in train and test, the model is
partly graded on recognising a person it has already heard, and WER looks
better than it will be for new users.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .text import normalize


def speaker_split(
    df: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 13,
    weight: str = "duration_s",
) -> pd.DataFrame:
    """Assign every utterance to train/validation/test with no speaker shared.

    Done separately per dataset so each dataset gets its own test set.
    Speakers are shuffled deterministically, then filled into test and
    validation until each reaches its share of audio (or of utterances if
    durations are missing). Same inputs + seed -> same split, always.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    out = []
    for name, part in df.groupby("dataset", sort=True):
        part = part.copy()
        w = part[weight] if weight in part and part[weight].notna().all() else None
        per_spk = (w.groupby(part["speaker_id"]).sum() if w is not None
                   else part.groupby("speaker_id").size()).astype(float)
        n_spk = len(per_spk)
        if n_spk < 3:
            raise ValueError(f"{name}: only {n_spk} speakers, cannot make 3 splits")

        rng = np.random.default_rng(_stable_seed(seed, name))
        order = per_spk.index.to_numpy().copy()
        order.sort()  # remove dependence on input row order
        rng.shuffle(order)

        total = per_spk.sum()
        targets = {"test": fractions[2] * total, "validation": fractions[1] * total}
        assign, filled = {}, {"test": 0.0, "validation": 0.0}
        for spk in order:
            for split in ("test", "validation"):
                if filled[split] < targets[split]:
                    assign[spk] = split
                    filled[split] += per_spk[spk]
                    break
            else:
                assign[spk] = "train"
        # Every split needs at least one speaker.
        for split in ("train", "validation", "test"):
            if split not in assign.values():
                raise ValueError(f"{name}: split '{split}' got no speakers; "
                                 "adjust fractions")
        part["split"] = part["speaker_id"].map(assign)
        out.append(part)
    result = pd.concat(out, ignore_index=True)
    assert_speaker_disjoint(result)
    return result


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()
    return int(digest[:8], 16)


def assert_speaker_disjoint(df: pd.DataFrame) -> None:
    per_spk = df.groupby("speaker_id")["split"].nunique()
    leaked = per_spk[per_spk > 1]
    if len(leaked):
        raise AssertionError(
            f"{len(leaked)} speakers appear in more than one split, "
            f"e.g. {leaked.index[0]!r}"
        )


def text_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Share of test/validation sentences whose exact text also occurs in train.

    Expect this near 100% for the Financial Inclusion data (everyone read the
    same sentences) and near 0% for spontaneous picture descriptions. Report
    it next to WER: a low WER on sentences the model trained on means little.
    """
    keyed = df.assign(_norm=df["text"].map(normalize))
    rows = []
    for name, part in keyed.groupby("dataset"):
        train_texts = set(part.loc[part["split"] == "train", "_norm"])
        for split in ("validation", "test"):
            held = part.loc[part["split"] == split, "_norm"]
            if len(held):
                rows.append({
                    "dataset": name, "split": split, "utterances": len(held),
                    "seen_in_train_pct": round(100 * held.isin(train_texts).mean(), 1),
                })
    return pd.DataFrame(rows)


def cross_dataset_overlap(a: pd.DataFrame, b: pd.DataFrame, min_words: int = 6
                          ) -> pd.DataFrame:
    """Utterances in `a` whose normalised transcript exactly matches one in `b`.

    For spontaneous speech, two people almost never produce the same 6+ word
    description, so an exact match very likely means the same recording was
    released in both datasets. Short phrases are ignored to avoid false alarms.
    """
    na = a.assign(_norm=a["text"].map(normalize))
    nb = b.assign(_norm=b["text"].map(normalize))
    na = na[na["_norm"].str.split().str.len() >= min_words]
    nb = nb[nb["_norm"].str.split().str.len() >= min_words]
    merged = na.merge(nb[["_norm", "utt_id"]], on="_norm", suffixes=("", "_other"))
    return merged[["utt_id", "utt_id_other", "_norm"]].rename(
        columns={"_norm": "normalized_text"})
