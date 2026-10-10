import pandas as pd
import pytest

from akan_asr.datasets.fin_incl import parse_filename
from akan_asr.manifest import make_manifest
from akan_asr.metrics import paired_difference, score
from akan_asr.splits import (assert_speaker_disjoint, cross_dataset_overlap,
                             speaker_split, text_leakage)


def _toy(dataset="fin_incl", n_speakers=30, per_speaker=5, same_sentences=False):
    rows = []
    for s in range(n_speakers):
        for u in range(per_speaker):
            text = f"sentence {u}" if same_sentences else f"spk{s} says thing {u}"
            rows.append({
                "utt_id": f"{dataset}:{s}-{u}", "audio_path": f"/x/{s}-{u}.wav",
                "text": text, "speaker_id": f"{dataset}:{s}", "dataset": dataset,
                "dialect": "asante", "gender": "female", "duration_s": 3.0 + s % 4,
            })
    return make_manifest(rows)


def test_split_is_speaker_disjoint_and_covers_all():
    df = _toy()
    out = speaker_split(df)
    assert len(out) == len(df)
    assert set(out["split"]) == {"train", "validation", "test"}
    assert_speaker_disjoint(out)


def test_split_is_deterministic_and_order_independent():
    df = _toy()
    a = speaker_split(df, seed=1).set_index("utt_id")["split"]
    b = speaker_split(df.sample(frac=1, random_state=0), seed=1).set_index("utt_id")["split"]
    assert a.sort_index().equals(b.sort_index())


def test_split_fractions_roughly_respected():
    out = speaker_split(_toy(n_speakers=100))
    share = out.groupby("split")["duration_s"].sum() / out["duration_s"].sum()
    assert 0.08 <= share["test"] <= 0.14
    assert 0.08 <= share["validation"] <= 0.14


def test_each_dataset_split_separately():
    df = pd.concat([_toy("fin_incl"), _toy("waxal")], ignore_index=True)
    out = speaker_split(df)
    for name in ("fin_incl", "waxal"):
        assert set(out.loc[out.dataset == name, "split"]) == {"train", "validation", "test"}


def test_too_few_speakers_raises():
    with pytest.raises(ValueError):
        speaker_split(_toy(n_speakers=2))


def test_text_leakage_detects_shared_prompts():
    leak = text_leakage(speaker_split(_toy(same_sentences=True)))
    assert (leak["seen_in_train_pct"] == 100.0).all()
    clean = text_leakage(speaker_split(_toy(same_sentences=False)))
    assert (clean["seen_in_train_pct"] == 0.0).all()


def test_cross_dataset_overlap_ignores_short_phrases():
    a = make_manifest([
        {"utt_id": "waxal:1", "audio_path": "a", "text": "Abofra no te dua no ase reni aduane",
         "speaker_id": "waxal:1", "dataset": "waxal"},
        {"utt_id": "waxal:2", "audio_path": "b", "text": "Ɛte sɛn",
         "speaker_id": "waxal:1", "dataset": "waxal"},
    ])
    b = make_manifest([
        {"utt_id": "ugspeech:9", "audio_path": "c", "text": "abofra no te dua no ase, reni aduane.",
         "speaker_id": "ugspeech:3", "dataset": "ugspeech"},
        {"utt_id": "ugspeech:8", "audio_path": "d", "text": "ɛte sɛn",
         "speaker_id": "ugspeech:3", "dataset": "ugspeech"},
    ])
    dup = cross_dataset_overlap(a, b)
    assert dup["utt_id"].tolist() == ["waxal:1"]


def test_manifest_rejects_duplicate_ids():
    row = {"utt_id": "x", "audio_path": "a", "text": "t", "speaker_id": "s", "dataset": "d"}
    with pytest.raises(ValueError):
        make_manifest([row, row])


def test_score_perfect_and_known_errors():
    perfect = score(["me pɛ sika"], ["Me pɛ sika."])
    assert perfect.wer.value == 0.0 and perfect.cer.value == 0.0
    # one substitution out of 3 words, and one empty ref skipped
    res = score(["me pɛ sika", "?"], ["me pɛ nsuo", "anything"])
    assert res.wer.value == pytest.approx(1 / 3)
    assert res.n_skipped_empty_refs == 1
    assert res.wer.ci_low <= res.wer.value <= res.wer.ci_high


def test_corpus_wer_weights_by_length():
    # mean-of-utterance WER would be 0.5; corpus WER is 1/5 = 0.2
    res = score(["a", "b c d e"], ["x", "b c d e"])
    assert res.wer.value == pytest.approx(0.2)


@pytest.mark.parametrize("name,speaker,gender,age", [
    ("GaFm21-ATuJLn5X-Tmp083-zykm34.ogg", "ATuJLn5X", "female", 21),
    # real filenames from the downloaded archives
    ("AsantiTwiFm23-MMuHe3cd-Tmp033-90pZBJ.ogg", "MMuHe3cd", "female", 23),
    ("AsantiTwiMa23-AOHqG4Hk-Tmp101-W90Pd5.ogg", "AOHqG4Hk", "male", 23),
    ("AkuapemTwiMa24-POxqPYmx-Tmp087-inx10P.ogg", "POxqPYmx", "male", 24),
    ("GaMa22-NK54l7ZF-Tmp060-wx410p.ogg", "NK54l7ZF", "male", 22),
    ("AsantiTwiFm20-A SLRKMb-Tmp010-o9jyxQ.ogg", "A SLRKMb", "female", 20),
    ("garbage.ogg", None, "unknown", None),
])
def test_fin_incl_filename_parsing(name, speaker, gender, age):
    info = parse_filename(name)
    assert (info["speaker"], info["gender"], info["age"]) == (speaker, gender, age)


def test_paired_difference_detects_consistent_gain():
    refs = ["me pɛ sika no"] * 50 + ["ɛte sɛn"] * 50
    better = list(refs)
    worse = ["me pɛ nsuo no"] * 50 + ["ɛte sɛn"] * 50  # one extra error on half the clips
    res = paired_difference(refs, better, worse, n_bootstrap=2000)
    assert res.diff == pytest.approx(-50 / 300)
    assert res.ci_high < 0 and res.p_value < 0.01


def test_paired_difference_identical_systems():
    refs = ["a b c", "d e"]
    res = paired_difference(refs, ["a x c", "d e"], ["a x c", "d e"], n_bootstrap=500)
    assert res.diff == 0 and res.p_value == 1.0
