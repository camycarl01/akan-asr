# Akan ASR: robust Twi speech recognition across domains

Current Akan speech models score well on the dataset they were trained on and
fall apart on others (Mensah et al., 2025: ~30% WER in-domain, 70–100% elsewhere).
This project tests whether training Whisper with LoRA adapters on several Akan
corpora together closes that gap.

Status: Week 1 — data pipeline and zero-shot baseline.

## What's in the repo

| Path | What it does |
| --- | --- |
| `src/akan_asr/text.py` | One Twi normaliser used for every transcript and every prediction (ɛ/ɔ look-alikes, tone marks, punctuation) |
| `src/akan_asr/datasets/` | One loader per dataset, all producing the same manifest format |
| `src/akan_asr/audio.py` | Converts any audio to 16 kHz mono WAV, resumable, logs failures |
| `src/akan_asr/splits.py` | Speaker-disjoint splits, sentence-leakage report, WAXAL/UGSpeechData duplicate check |
| `src/akan_asr/metrics.py` | Corpus WER/CER with 95% bootstrap confidence intervals |
| `src/akan_asr/transcribe.py` | Batched Whisper inference (zero-shot or with a LoRA adapter) |
| `scripts/` | The three commands you actually run |
| `tests/` | `pytest` — run it after any change to normalisation or splitting |

## Datasets

| Dataset | Get it | Licence | Notes |
| --- | --- | --- | --- |
| Ashesi Financial Inclusion | [Asante](https://adr.ashesi.edu.gh/datasets/10), [Akuapem](https://adr.ashesi.edu.gh/datasets/12), [Fante](https://adr.ashesi.edu.gh/datasets/13) | CC BY 4.0 | Download both the 10p and 90p archives per dialect into one folder. Everyone read the same ~130 sentences. |
| WAXAL Akan | Hugging Face `google/WaxalNLP`, config `aka_asr` (downloaded by the script) | CC BY / BY-SA 4.0 | Collected with the University of Ghana, may overlap UGSpeechData. No Fante ASR data. |
| UGSpeechData Akan | [Science Data Bank](https://doi.org/10.57760/sciencedb.22298) | CC BY-NC-ND 4.0 | Wait for the lab's reply before publishing any model trained on it. |

## Running it on Kaggle (Week 1)

Easiest: import `notebooks/week1_kaggle.ipynb` into Kaggle (File → Import Notebook) and run it top to bottom. The steps below are the same thing by hand.


1. New notebook → Settings → Accelerator: GPU T4 ×2, Internet: on.
2. Upload the extracted Ashesi folders as a Kaggle Dataset (e.g. `fin-incl-akan`) and attach it.
3. In the first cell:

```bash
!git clone https://github.com/<you>/akan-asr.git && cd akan-asr && pip install -q -e ".[model]"
%cd akan-asr
```

4. Prepare each dataset (a quick test first with `--limit 50`):

```bash
!python scripts/prepare_data.py fin_incl --root /kaggle/input/fin-incl-akan/asante  --dialect asante
!python scripts/prepare_data.py fin_incl --root /kaggle/input/fin-incl-akan/akuapem --dialect akuapem
!python scripts/prepare_data.py fin_incl --root /kaggle/input/fin-incl-akan/fante   --dialect fante
!python scripts/prepare_data.py waxal
```

Each prints hours, speakers, and every character in the transcripts. Read that
character list: anything odd (digits, `3`, `)`, stray symbols) is a cleaning job.

5. Make the splits:

```bash
!python scripts/make_splits.py data/manifests/*.csv
```

Read the two reports it prints. "Seen in train" will be near 100% for the Ashesi
data; say so next to its WER. When UGSpeechData is added, check the WAXAL
duplicate count and use `--drop-waxal-duplicates` if it is large.

6. Zero-shot baseline (the "before" numbers):

```bash
!python scripts/run_baseline.py data/splits/*_test.csv --model openai/whisper-small
```

Results land in `results/whisper-small-zeroshot/`: a summary table, per-dataset
metrics JSON, and per-utterance predictions for error analysis.

7. Copy `data/manifests` and `data/splits` (small CSVs) into a Kaggle Dataset
   so Week 2 doesn't redo this. The WAVs can stay in the notebook output.

## Decisions to record in the write-up

- Normalisation: lowercase, no punctuation, no tone marks, look-alikes mapped to ɛ/ɔ. Same for every model.
- Splits: 80/10/10 by audio duration, speaker-disjoint, per dataset, seed 13.
- Clips over 30 s are excluded from Whisper evaluation (Whisper only sees 30 s); the count is saved in each metrics file.
- Whisper has no Akan language token; the baseline uses auto-detect. Try `--language yoruba` or `--language swahili` as a side experiment.

## Tests

```bash
pip install -e ".[dev]" && pytest
```
