"""Short high-rate block code decoded with ORBGRAND (the decoder of ../../BitSemCom/grand).

The code and decoder are the ones used in the BitSemCom + GRAND experiments: a systematic
CRC or random linear code and the numpy port of K. R. Duffy's ORBGRAND (non-commercial
academic research licence, see BitSemCom/grand/orbgrand.py for the citations).

    code spec   'crc:0x9eb2:128:112'  CRC, Koopman polynomial, n, k
                'rlc:128:112:0'       random linear code, n, k, seed
    LLR sign    L > 0 <=> bit 0, as everywhere else in this package

A block whose hard decision already satisfies all parity checks is accepted without calling
the decoder (that is ORBGRAND's first query).  A block the decoder gives up on after
`max_queries` codebook queries is returned as its hard decision ("abandoned").
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_MODULE = "bitsemcom_grand"


def load_grand(grand_dir):
    """Import BitSemCom's grand package from its folder, without putting BitSemCom on sys.path
    (its top-level utils.py would shadow the DeepSC repo's)."""
    if _MODULE in sys.modules:
        return sys.modules[_MODULE]
    grand_dir = Path(grand_dir).resolve()
    init = grand_dir / "__init__.py"
    if not init.exists():
        raise FileNotFoundError(f"GRAND package not found at {grand_dir}")
    spec = importlib.util.spec_from_file_location(_MODULE, init, submodule_search_locations=[str(grand_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE] = mod
    spec.loader.exec_module(mod)
    return mod


class GrandCode:
    def __init__(self, grand_dir, spec: str = "crc:0x9eb2:128:112", max_queries: int = 2000, variant: str = "1line"):
        g = load_grand(grand_dir)
        parts = spec.split(":")
        if parts[0] == "crc":
            _, hexpoly, n, k = parts
            poly = g.koopman_to_poly(hexpoly)
            if len(poly) - 1 != int(n) - int(k):
                raise ValueError(f"polynomial degree {len(poly) - 1} != n-k = {int(n) - int(k)}")
            G, H = g.make_crc(int(k), poly)
        elif parts[0] == "rlc":
            _, n, k, seed = parts
            G, H = g.make_rlc(int(k), int(n), seed=int(seed))
        else:
            raise ValueError(f"unknown code spec {spec!r}")
        self.spec, self.variant, self.max_queries = spec, variant, int(max_queries)
        self.G, self.H = G.astype(np.uint8), H.astype(np.uint8)
        self.k, self.n = self.G.shape
        self.rate = self.k / self.n
        self.decoder = g.ORBGRAND(self.H, max_queries=self.max_queries, variant=variant)
        self._G_t = {}

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """u [Nblk, k] bits -> codewords [Nblk, n] (uint8, systematic: the first k bits are u)."""
        G = self._G_t.get(u.device)
        if G is None:
            G = self._G_t[u.device] = torch.from_numpy(self.G).float().to(u.device)
        return (u.float() @ G).remainder(2).to(torch.uint8)

    def decode(self, llr: torch.Tensor):
        """llr [Nblk, n] -> (words [Nblk, n] uint8, abandoned [Nblk] bool, queries [Nblk] long), on llr's device."""
        L = llr.detach().double().cpu().numpy()
        words = (L < 0).astype(np.uint8)
        queries = np.ones(L.shape[0], dtype=np.int64)
        abandoned = np.zeros(L.shape[0], dtype=bool)
        failing = ((words.astype(np.int64) @ self.H.T.astype(np.int64)) % 2).any(axis=1)
        for i in np.flatnonzero(failing):
            words[i], queries[i], abandoned[i] = self.decoder.decode(L[i])
        dev = llr.device
        return torch.from_numpy(words).to(dev), torch.from_numpy(abandoned).to(dev), torch.from_numpy(queries).to(dev)

    def __repr__(self):
        return f"GrandCode({self.spec}, n={self.n}, k={self.k}, R={self.rate:.3f}, ORBGRAND {self.variant}, budget={self.max_queries})"
