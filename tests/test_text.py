from akan_asr.text import akan_charset, normalize


def test_lowercase_and_punctuation():
    assert normalize("Ɛte sɛn?  Me ho yɛ!") == "ɛte sɛn me ho yɛ"


def test_lookalike_letters_become_akan_letters():
    # Greek epsilon and reversed-c look-alikes are common in typed Twi.
    assert normalize("Mε pε sika") == "mɛ pɛ sika"
    assert normalize("ͻno ↄbae") == "ɔno ɔbae"


def test_tone_marks_removed_but_open_vowels_kept():
    assert normalize("ɔ́bɛ́ba") == "ɔbɛba"


def test_internal_apostrophe_kept_quotes_dropped():
    assert normalize("Ɔbue n’ani 'ɔkyena'") == "ɔbue n'ani ɔkyena"


def test_ascii_substitutes_only_when_asked():
    assert normalize("me p3 sika", map_ascii_substitutes=False) == "me p3 sika"
    assert normalize("me p3 sika", map_ascii_substitutes=True) == "me pɛ sika"
    # A standalone digit is a real number, not ɛ.
    assert normalize("ɛyɛ 3", map_ascii_substitutes=True) == "ɛyɛ 3"


def test_nfc_equivalence():
    composed = "é"            # é
    decomposed = "é"         # e + combining acute
    assert normalize(composed) == normalize(decomposed) == "e"


def test_empty_and_none():
    assert normalize("") == ""
    assert normalize(None) == ""
    assert normalize("?!.") == ""


def test_charset():
    assert akan_charset(["Ɛte sɛn", "ɔbaa"]) == set("ɛtesnɔba")
