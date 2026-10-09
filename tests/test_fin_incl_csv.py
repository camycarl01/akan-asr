import pytest

from akan_asr.datasets.fin_incl import read_metadata

ROWS = [
    ("a/b/AsantiTwiFm23-MMuHe3cd-Tmp033-90pZBJ.ogg", "Mepɛ sɛ, metua ka", "I want to, pay"),
    ("a/b/AsantiTwiMa23-AOHqG4Hk-Tmp101-W90Pd5.ogg", "Ɛhe na ɔwɔ", "Where is it"),
]


@pytest.mark.parametrize("sep", ["\t", "|", ";"])
def test_separator_detected_with_commas_in_text(tmp_path, sep):
    f = tmp_path / "data.csv"
    lines = [sep.join(["Audio Filepath", "Transcription", "Translation"])]
    lines += [sep.join(r) for r in ROWS]
    f.write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")  # with BOM
    df = read_metadata(f)
    assert list(df.columns) == ["Audio Filepath", "Transcription", "Translation"]
    assert df["Transcription"].tolist() == ["Mepɛ sɛ, metua ka", "Ɛhe na ɔwɔ"]


def test_plain_comma_csv_still_works(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text('path,transcription\nx.ogg,"Mepɛ sɛ, metua"\n', encoding="utf-8")
    df = read_metadata(f)
    assert df.shape == (1, 2)
    assert df.loc[0, "transcription"] == "Mepɛ sɛ, metua"
