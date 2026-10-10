"""LoRA fine-tuning of Whisper on manifest CSVs.

Why LoRA: only the small adapter matrices train (~1-2% of the weights), so
whisper-small fits on a Kaggle T4, and each adapter is a few MB. We keep one
adapter per training mix (Ashesi only, WAXAL only, both) and score them all
with the same scripts/run_baseline.py --adapter, so the numbers are comparable.

Decisions that matter for the write-up:
- Targets are the *normalised* transcripts (text.normalize), so the model
  learns to output exactly the form WER is computed on.
- Language token: Whisper has no Akan token, so we pick one (default
  "yoruba") and use it both in training and decoding. A prefix the model never
  trained with at test time costs WER for no reason. The token is saved in the
  adapter folder (train_config.json) and transcribe.py reads it from there.
- Clips over 30 s are dropped (Whisper's window), same as in evaluation.
- The checkpoint kept as "best" has the lowest *mean* validation WER across
  datasets, so a large dataset cannot hide a drop on a small one.
- With balance=True, batches draw from each training dataset equally often
  (sampling with replacement), instead of in proportion to dataset size.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import manifest
from .metrics import score
from .text import normalize
from .transcribe import token_budget

log = logging.getLogger(__name__)

WHISPER_MAX_S = 30.0
MAX_LABEL_TOKENS = 448  # Whisper decoder's position limit
SAMPLE_RATE = 16_000
TRAIN_CONFIG = "train_config.json"


@dataclass
class TrainConfig:
    train_manifests: list[str]
    val_manifests: list[str]
    out_dir: str
    model: str = "openai/whisper-small"
    language: str = "yoruba"
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lr: float = 1e-3
    epochs: float = 3.0
    max_steps: int | None = None
    batch_size: int = 16
    grad_accum: int = 1
    warmup_steps: int = 100
    eval_every: int = 500
    val_utts: int = 400  # per evaluation, split evenly across val datasets
    seed: int = 13
    limit: int | None = None  # training utterances, for smoke tests
    balance: bool = False  # sample each training dataset equally often
    num_workers: int = 2


def load_split(paths: list[str], limit: int | None = None, seed: int = 13) -> pd.DataFrame:
    """Concatenate manifests, drop clips Whisper can't see whole, add `target`."""
    df = pd.concat([manifest.load(p) for p in paths], ignore_index=True)
    too_long = df["duration_s"].notna() & (df["duration_s"] > WHISPER_MAX_S)
    if too_long.any():
        log.warning("dropping %d clips longer than %.0f s", int(too_long.sum()), WHISPER_MAX_S)
        df = df[~too_long]
    df = df.assign(target=df["text"].map(normalize))
    empty = df["target"] == ""
    if empty.any():
        log.warning("dropping %d clips whose transcript is empty after normalising",
                    int(empty.sum()))
        df = df[~empty]
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=seed)
    return df.reset_index(drop=True)


def val_subset(df: pd.DataFrame, n: int, seed: int = 13) -> pd.DataFrame:
    """A fixed random sample with the same number of clips from each dataset."""
    per = max(1, n // df["dataset"].nunique())
    parts = [g.sample(n=min(per, len(g)), random_state=seed)
             for _, g in df.groupby("dataset", sort=True)]
    return pd.concat(parts, ignore_index=True)


class AudioTextDataset:
    """Indexable (path, target) pairs; audio is read lazily in the workers."""

    def __init__(self, df: pd.DataFrame):
        self.paths = df["audio_path"].tolist()
        self.targets = df["target"].tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        audio, sr = sf.read(self.paths[i], dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{self.paths[i]}: {sr} Hz, expected {SAMPLE_RATE}; "
                             "run prepare_data.py first")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return {"audio": audio, "text": self.targets[i]}


class WhisperCollator:
    """Log-mel features + label ids (padding masked with -100 so loss ignores it)."""

    def __init__(self, processor):
        self.processor = processor
        self.sot = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    def __call__(self, batch: list[dict]) -> dict:
        feats = self.processor.feature_extractor(
            [b["audio"] for b in batch], sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features
        tok = self.processor.tokenizer([b["text"] for b in batch], padding=True,
                                       return_tensors="pt")
        labels = tok.input_ids.masked_fill(tok.attention_mask.ne(1), -100)
        # The model prepends <|startoftranscript|> itself when shifting labels
        # right, so drop it here or it would be predicted twice.
        if (labels[:, 0] == self.sot).all():
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}


def read_train_config(adapter_dir: str | Path) -> dict | None:
    path = Path(adapter_dir) / TRAIN_CONFIG
    return json.loads(path.read_text()) if path.exists() else None


def train(cfg: TrainConfig) -> Path:
    """Train a LoRA adapter; returns the folder of the best checkpoint."""
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                              get_linear_schedule_with_warmup)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    if device == "cpu":
        log.warning("training on CPU: only useful as a smoke test")

    processor = WhisperProcessor.from_pretrained(cfg.model)
    processor.tokenizer.set_prefix_tokens(language=cfg.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model)
    model.generation_config.forced_decoder_ids = None
    model = get_peft_model(model, LoraConfig(
        r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules, bias="none"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log.info("trainable parameters: %d of %d (%.2f%%)", n_train, n_all, 100 * n_train / n_all)
    model.to(device)

    train_df = load_split(cfg.train_manifests, cfg.limit, cfg.seed)
    n_tok = train_df["target"].map(lambda t: len(processor.tokenizer(t).input_ids))
    if (n_tok > MAX_LABEL_TOKENS).any():
        log.warning("dropping %d clips with more than %d label tokens",
                    int((n_tok > MAX_LABEL_TOKENS).sum()), MAX_LABEL_TOKENS)
        train_df = train_df[n_tok <= MAX_LABEL_TOKENS].reset_index(drop=True)
    val_df = val_subset(load_split(cfg.val_manifests), cfg.val_utts, cfg.seed)
    for name, part in train_df.groupby("dataset"):
        log.info("train %-10s %6d clips  %.1f h", name, len(part),
                 part["duration_s"].sum() / 3600)
    log.info("validation subset: %s", val_df["dataset"].value_counts().to_dict())

    collate = WhisperCollator(processor)
    generator = torch.Generator().manual_seed(cfg.seed)
    sampler = None
    if cfg.balance and train_df["dataset"].nunique() > 1:
        sizes = train_df["dataset"].map(train_df["dataset"].value_counts())
        sampler = WeightedRandomSampler(torch.tensor((1.0 / sizes).to_numpy()),
                                        num_samples=len(train_df), replacement=True,
                                        generator=generator)
        log.info("balanced sampling: each dataset drawn equally often")
    train_dl = DataLoader(AudioTextDataset(train_df), batch_size=cfg.batch_size,
                          shuffle=sampler is None, sampler=sampler, collate_fn=collate,
                          num_workers=cfg.num_workers, generator=generator,
                          pin_memory=use_amp)
    val_dl = DataLoader(AudioTextDataset(val_df), batch_size=cfg.batch_size,
                        collate_fn=collate, num_workers=cfg.num_workers)

    steps_per_epoch = math.ceil(len(train_dl) / cfg.grad_accum)
    total = cfg.max_steps or math.ceil(steps_per_epoch * cfg.epochs)
    warmup = min(cfg.warmup_steps, max(1, total // 10))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, warmup, total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log.info("%d optimiser steps (%d per epoch), warmup %d", total, steps_per_epoch, warmup)

    def save(folder: Path, extra: dict) -> None:
        model.save_pretrained(folder)
        (folder / TRAIN_CONFIG).write_text(json.dumps({**asdict(cfg), **extra}, indent=2))

    history, best_wer, step, running = [], math.inf, 0, []
    t0 = time.time()
    model.train()
    while step < total:
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss / cfg.grad_accum
            scaler.scale(loss).backward()
            running.append(loss.item() * cfg.grad_accum)
            if (i + 1) % cfg.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % 50 == 0 or step == total:
                log.info("step %d/%d  loss %.3f  lr %.2e  %.0f min", step, total,
                         np.mean(running), sched.get_last_lr()[0], (time.time() - t0) / 60)
            if step % cfg.eval_every == 0 or step == total:
                ev = evaluate(model, processor, val_dl, val_df, cfg.language, device, use_amp)
                ev.update(step=step, train_loss=float(np.mean(running)))
                history.append(ev)
                pd.DataFrame(history).to_csv(out / "history.csv", index=False)
                log.info("eval step %d: val loss %.3f, val WER %.1f%% %s", step,
                         ev["val_loss"], 100 * ev["val_wer"],
                         {k: round(100 * v, 1) for k, v in ev.items() if k.startswith("wer_")})
                if ev["val_wer"] < best_wer:
                    best_wer = ev["val_wer"]
                    save(out / "best", {"best_step": step, "val_wer": best_wer})
                model.train()
            running = running if step % 50 else []
            if step >= total:
                break

    save(out / "last", {"step": step})
    log.info("done in %.0f min; best val WER %.1f%% -> %s",
             (time.time() - t0) / 60, 100 * best_wer, out / "best")
    return out / "best"


def evaluate(model, processor, val_dl, val_df: pd.DataFrame, language: str,
             device: str, use_amp: bool) -> dict:
    """Validation loss and greedy-decoding WER, overall and per dataset."""
    import torch

    model.eval()
    losses, preds = [], []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                enabled=use_amp):
        durations = val_df["duration_s"].fillna(30.0).tolist()
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            losses.append(model(**batch).loss.item())
            longest = max(durations[len(preds):len(preds) + len(batch["labels"])])
            ids = model.generate(input_features=batch["input_features"], language=language,
                                 task="transcribe", max_new_tokens=token_budget(longest, 225))
            preds.extend(processor.batch_decode(ids, skip_special_tokens=True))

    per_ds = {}
    for name in sorted(val_df["dataset"].unique()):
        mask = (val_df["dataset"] == name).to_numpy()
        res = score(val_df["target"][mask].tolist(),
                    [p for p, m in zip(preds, mask) if m], n_bootstrap=1)
        per_ds[f"wer_{name}"] = res.wer.value
    return {"val_loss": float(np.mean(losses)),
            "val_wer": float(np.mean(list(per_ds.values()))), **per_ds}
