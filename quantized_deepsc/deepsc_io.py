"""Thin wrapper around the frozen continuous DeepSC (../DeepSC-continous).

The reference repository is imported, never edited.  The transmitter and receiver
are exposed as `encode()` and `decode()` split exactly where `greedy_decode()` in
the repo applies the channel, so that any digital chain can be inserted between
them.  BLEU is computed with reference = target sentence, hypothesis = decoded
sentence (the repo's performance.py passes them the other way round).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent  # DeepSC-quantized/


def load_config(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path else HERE / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_like) -> Path:
    """Resolve a config path relative to DeepSC-quantized/."""
    p = Path(path_like)
    return p if p.is_absolute() else (PKG_ROOT / p)


def sha256_of(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


@contextlib.contextmanager
def working_dir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def import_repo(repo_dir):
    """Import models.transceiver, utils and dataset from the reference repo."""
    repo = resolve(repo_dir).resolve()
    if not (repo / "models" / "transceiver.py").exists():
        raise FileNotFoundError(f"DeepSC repo not found at {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import models.transceiver as transceiver  # noqa: E402
    import utils as repo_utils  # noqa: E402
    import dataset as repo_dataset  # noqa: E402
    return repo, transceiver, repo_utils, repo_dataset


_BLEU_WEIGHTS = {
    1: (1.0, 0.0, 0.0, 0.0),
    2: (0.5, 0.5, 0.0, 0.0),
    3: (1 / 3, 1 / 3, 1 / 3, 0.0),
    4: (0.25, 0.25, 0.25, 0.25),
}


def bleu_scores(targets: list[str], decoded: list[str], max_n: int = 4) -> dict:
    """Sentence-level BLEU-1..max_n averaged over sentences (no smoothing, as the repo)."""
    from nltk.translate.bleu_score import sentence_bleu
    from w3lib.html import remove_tags

    sums = {n: 0.0 for n in range(1, max_n + 1)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ref, hyp in zip(targets, decoded):
            r = remove_tags(ref).split()
            h = remove_tags(hyp).split()
            for n in range(1, max_n + 1):
                sums[n] += sentence_bleu([r], h, weights=_BLEU_WEIGHTS[n])
    cnt = max(len(targets), 1)
    return {f"bleu{n}": sums[n] / cnt for n in range(1, max_n + 1)}


class DeepSCLink:
    """Frozen DeepSC transmitter/receiver with the channel removed."""

    def __init__(self, cfg: dict | None = None, device: str | None = None):
        self.cfg = cfg or load_config()
        self.repo, tr, ru, rd = import_repo(self.cfg["repo_dir"])
        self.repo_utils = ru
        self.repo_dataset = rd
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        with open(self.repo / "europarl" / "vocab.json", "rb") as f:
            vocab = json.load(f)
        self.token_to_idx = vocab["token_to_idx"]
        self.num_vocab = len(self.token_to_idx)
        self.pad_idx = self.token_to_idx["<PAD>"]
        self.start_idx = self.token_to_idx["<START>"]
        self.end_idx = self.token_to_idx["<END>"]

        m = self.cfg["model"]
        V = self.num_vocab
        self.model = tr.DeepSC(m["num_layers"], V, V, V, V, m["d_model"], m["num_heads"], m["dff"], m["dropout"]).to(self.device)
        self.ckpt_path = self.repo / self.cfg["checkpoint"]
        state = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.ckpt_sha256 = sha256_of(self.ckpt_path)

        self.stot = ru.SeqtoText(self.token_to_idx, self.end_idx)
        self.max_len = int(self.cfg["max_len"])
        self.PowerNormalize = ru.PowerNormalize
        self.Channels = ru.Channels
        self.D = int(self.cfg["D"])

    # ------------------------------------------------------------------ data
    def snr_to_sigma(self, snr_db: float) -> float:
        """Repo convention: sigma per real dimension = 1/sqrt(2*10^(snr/10))."""
        return float(self.repo_utils.SNR_to_noise(snr_db))

    def test_dataset(self, subset: int | None = None):
        with working_dir(self.repo):
            ds = self.repo_dataset.EurDataset("test")
        if subset:
            ds = Subset(ds, range(min(int(subset), len(ds))))
        return ds

    def test_loader(self, subset: int | None = None, batch_size: int | None = None) -> DataLoader:
        ds = self.test_dataset(subset)
        return DataLoader(ds, batch_size=batch_size or int(self.cfg["batch_size"]), shuffle=False,
                          num_workers=0, collate_fn=self.repo_dataset.collate_data)

    # ------------------------------------------------------------- transmitter
    @torch.no_grad()
    def encode(self, sents: torch.Tensor):
        """tokens [B,L] -> (t [B,L,16] exactly as transmitted, pad_mask [B,L] True on pads)."""
        sents = sents.to(self.device)
        src_mask = (sents == self.pad_idx).unsqueeze(-2).float()
        enc = self.model.encoder(sents, src_mask)
        t = self.PowerNormalize(self.model.channel_encoder(enc))
        return t, (sents == self.pad_idx)

    # ---------------------------------------------------------------- receiver
    @torch.no_grad()
    def decode(self, t_hat: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        """t_hat [B,L,16] -> greedy token ids [B,max_len] (copy of utils.greedy_decode after the channel)."""
        max_len = max_len or self.max_len
        memory = self.model.channel_decoder(t_hat)
        B = memory.size(0)
        outputs = torch.full((B, 1), self.start_idx, dtype=torch.long, device=self.device)
        for _ in range(max_len - 1):
            size = outputs.size(1)
            trg_mask = (outputs == self.pad_idx).unsqueeze(-2).float()  # [B,1,size]
            look_ahead = torch.triu(torch.ones(size, size, device=self.device), diagonal=1).unsqueeze(0)
            combined = torch.max(trg_mask, look_ahead)
            dec = self.model.decoder(outputs, memory, combined, None)
            pred = self.model.dense(dec)
            next_word = pred[:, -1:, :].argmax(dim=-1)
            outputs = torch.cat([outputs, next_word], dim=1)
        return outputs

    def to_text(self, ids: torch.Tensor) -> list[str]:
        return [self.stot.sequence_to_text(row) for row in ids.cpu().numpy().tolist()]

    # ------------------------------------------------------------------ metrics
    @staticmethod
    def bleu(targets: list[str], decoded: list[str], max_n: int = 4) -> dict:
        return bleu_scores(targets, decoded, max_n)
