"""(dv,dc)-regular LDPC code with a systematic encoder and a batched BP decoder (torch).

Construction: random socket pairing (MacKay style), parallel edges removed, then
4-cycles removed by degree-preserving edge swaps.  H is column-permuted so that the
code is systematic: c = [u | p] with p = P u (mod 2), P obtained by GF(2) Gaussian
elimination.  A (3,6)-regular H of size 504x1008 typically has rank 502-504, so the
code dimension k_eff = n - rank can exceed the target k = 504; the extra k_eff - k
information bits are shortened (fixed to 0, known to the decoder).

Decoder: log-domain sum-product (tanh rule, exact leave-one-out products) or
normalised min-sum, batched over codewords, early stop on zero syndrome.
LLR convention: L > 0 <=> bit 0.  decode(llr) needs nothing but the LLRs.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


# ----------------------------------------------------------------------------- GF(2)
def gf2_rref(H: np.ndarray):
    """Reduced row echelon form over GF(2). Returns (reduced nonzero rows, pivot columns)."""
    A = H.copy().astype(np.uint8)
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        if r >= m:
            break
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        others = np.nonzero(A[:, c])[0]
        others = others[others != r]
        if others.size:
            A[others] ^= A[r]
        pivots.append(c)
        r += 1
    return A[:r], pivots


# ----------------------------------------------------------------------- construction
def construct_regular(n: int, dv: int, dc: int, rng: np.random.Generator, max_cycle_iters: int = 20000):
    """Random (dv,dc)-regular parity-check matrix without parallel edges and (if possible) without 4-cycles."""
    m = n * dv // dc
    assert m * dc == n * dv, "n*dv must equal m*dc"
    cols = np.repeat(np.arange(n), dv)
    rows = np.repeat(np.arange(m), dc)
    rng.shuffle(rows)
    for _ in range(200000):  # remove parallel edges by swapping row endpoints
        key = rows.astype(np.int64) * n + cols
        _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        dup = np.nonzero(counts[inv] > 1)[0]
        if dup.size == 0:
            break
        e = dup[0]
        f = rng.integers(len(rows))
        rows[e], rows[f] = rows[f], rows[e]
    else:
        raise RuntimeError("could not remove parallel edges")
    H = np.zeros((m, n), dtype=np.uint8)
    H[rows, cols] = 1
    assert (H.sum(0) == dv).all() and (H.sum(1) == dc).all()

    n4 = -1
    for _ in range(max_cycle_iters):  # 4-cycle removal by degree-preserving edge swaps
        Hf = H.astype(np.float32)
        S = Hf @ Hf.T
        np.fill_diagonal(S, 0)
        r1s, r2s = np.nonzero(np.triu(S) > 1)
        n4 = len(r1s)
        if n4 == 0:
            break
        j = rng.integers(n4)
        r1, r2 = int(r1s[j]), int(r2s[j])
        shared = np.nonzero(H[r1] & H[r2])[0]
        c = int(shared[rng.integers(len(shared))])
        for _ in range(200):
            r3 = int(rng.integers(m))
            if r3 == r1 or H[r3, c]:
                continue
            c3s = np.nonzero(H[r3])[0]
            c3 = int(c3s[rng.integers(len(c3s))])
            if H[r1, c3]:
                continue
            H[r1, c] = 0
            H[r1, c3] = 1
            H[r3, c3] = 0
            H[r3, c] = 1
            break
    assert (H.sum(0) == dv).all() and (H.sum(1) == dc).all()
    return H, n4


def count_4cycles(H: np.ndarray) -> int:
    Hf = H.astype(np.float32)
    S = Hf @ Hf.T
    np.fill_diagonal(S, 0)
    return int((np.triu(S) > 1).sum())


# --------------------------------------------------------------------------------- code
class LDPC:
    def __init__(self, n: int = 1008, dv: int = 3, dc: int = 6, k: int = 504, seed: int = 1234,
                 cache_dir=None, device=None, bp_iters: int = 50, decoder: str = "sum_product",
                 llr_clip: float = 20.0, min_sum_alpha: float = 0.8):
        self.n, self.dv, self.dc = int(n), int(dv), int(dc)
        self.m = self.n * self.dv // self.dc
        self.seed = int(seed)
        self.bp_iters = int(bp_iters)
        self.decoder = decoder
        self.llr_clip = float(llr_clip)
        self.alpha = float(min_sum_alpha)
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        cache = None
        if cache_dir is not None:
            cache = Path(cache_dir) / f"ldpc_n{self.n}_dv{self.dv}_dc{self.dc}_seed{self.seed}.npz"
        if cache is not None and cache.exists():
            z = np.load(cache)
            H, P, n4 = z["H"], z["P"], int(z["n4"])
        else:
            rng = np.random.default_rng(self.seed)
            H0, n4 = construct_regular(self.n, self.dv, self.dc, rng)
            R, pivots = gf2_rref(H0)
            nonpiv = [c for c in range(self.n) if c not in set(pivots)]
            order = nonpiv + list(pivots)
            H = np.ascontiguousarray(H0[:, order])
            P = np.ascontiguousarray(R[:, nonpiv])  # [rank, k_eff]: parity_i = sum_j P[i,j] u_j
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, H=H, P=P, n4=n4)
        self.H = H                       # [m, n] uint8, systematic column order [info | parity]
        self.n_4cycles = n4
        self.rank = P.shape[0]
        self.k_eff = self.n - self.rank
        if k > self.k_eff:
            raise ValueError(f"requested k={k} but code dimension is {self.k_eff}")
        self.k = int(k)
        self.n_short = self.k_eff - self.k
        self.rate = self.k / self.n
        self.P = torch.tensor(P, dtype=torch.float32, device=self.device)

        rows, cols = np.nonzero(H)                     # row-major -> check order
        E = len(rows)
        assert E == self.m * self.dc
        self.E = E
        self.chk_cols = torch.tensor(cols.reshape(self.m, self.dc), dtype=torch.long, device=self.device)
        order_by_var = np.lexsort((rows, cols))        # sort by column, then row
        pos_in_var = np.empty(E, dtype=np.int64)
        pos_in_var[order_by_var] = np.arange(E)
        self.order_by_var = torch.tensor(order_by_var, dtype=torch.long, device=self.device)
        self.pos_in_var = torch.tensor(pos_in_var, dtype=torch.long, device=self.device)
        self._slot = torch.arange(self.dc, device=self.device)

    # ---------------------------------------------------------------- encoding
    @torch.no_grad()
    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """u [Bc, k] bits -> codewords [Bc, n] (uint8)."""
        Bc = u.shape[0]
        u = u.to(self.device).float()
        if self.n_short:
            u = torch.cat([u, torch.zeros(Bc, self.n_short, device=self.device)], dim=1)
        p = torch.remainder(u @ self.P.T, 2.0)
        return torch.cat([u, p], dim=1).to(torch.uint8)

    @torch.no_grad()
    def syndrome(self, bits: torch.Tensor) -> torch.Tensor:
        """bits [Bc, n] -> syndrome [Bc, m]."""
        return bits.to(self.device).long()[:, self.chk_cols].sum(-1) % 2

    def syndrome_ok(self, bits: torch.Tensor) -> torch.Tensor:
        return (self.syndrome(bits) == 0).all(-1)

    # ---------------------------------------------------------------- decoding
    def _check_update(self, v2c: torch.Tensor) -> torch.Tensor:
        """v2c [Bc, m, dc] -> c2v [Bc, m, dc] (extrinsic)."""
        if self.decoder == "min_sum":
            mag = v2c.abs()
            sgn = torch.where(v2c < 0, -1.0, 1.0)
            sext = sgn.prod(-1, keepdim=True) * sgn
            min1, i1 = mag.min(-1, keepdim=True)
            mag2 = mag.scatter(-1, i1, float("inf"))
            min2 = mag2.min(-1, keepdim=True).values
            ext = torch.where(self._slot == i1, min2, min1)
            return self.alpha * sext * ext
        t = torch.tanh(v2c * 0.5)
        ones = torch.ones_like(t[..., :1])
        cp = torch.cumprod(t, dim=-1)
        cs = torch.flip(torch.cumprod(torch.flip(t, [-1]), dim=-1), [-1])
        pre = torch.cat([ones, cp[..., :-1]], dim=-1)
        suf = torch.cat([cs[..., 1:], ones], dim=-1)
        ext = torch.clamp(pre * suf, -1 + 1e-7, 1 - 1e-7)
        return 2.0 * torch.atanh(ext)

    @torch.no_grad()
    def _decode_chunk(self, llr: torch.Tensor, max_iters: int):
        Bc = llr.shape[0]
        llr = llr.to(self.device).float().clamp(-self.llr_clip, self.llr_clip).clone()
        if self.n_short:
            llr[:, self.k:self.k_eff] = self.llr_clip  # shortened bits are known zeros
        total = llr
        c2v_var = torch.zeros(Bc, self.n, self.dv, device=self.device)
        hard = (total < 0).to(torch.uint8)
        done = self.syndrome_ok(hard)
        out = hard.clone()
        n_it = torch.where(done, torch.zeros_like(done, dtype=torch.long), torch.full((Bc,), max_iters, dtype=torch.long, device=self.device))
        for it in range(1, max_iters + 1):
            if bool(done.all()):
                break
            v2c_var = total.unsqueeze(-1) - c2v_var                                  # [Bc, n, dv]
            v2c_chk = v2c_var.reshape(Bc, self.E)[:, self.pos_in_var].reshape(Bc, self.m, self.dc)
            c2v_chk = self._check_update(v2c_chk)
            c2v_var = c2v_chk.reshape(Bc, self.E)[:, self.order_by_var].reshape(Bc, self.n, self.dv)
            total = llr + c2v_var.sum(-1)
            hard = (total < 0).to(torch.uint8)
            ok = self.syndrome_ok(hard)
            newly = ok & ~done
            if bool(newly.any()):
                out[newly] = hard[newly]
                n_it[newly] = it
                done = done | newly
        out[~done] = hard[~done]
        return out, done, n_it

    @torch.no_grad()
    def decode(self, llr: torch.Tensor, max_iters: int | None = None, chunk: int = 4096):
        """llr [Bc, n] (L > 0 <=> bit 0) -> (bits [Bc, n] uint8, converged [Bc] bool, n_iters [Bc])."""
        max_iters = max_iters or self.bp_iters
        outs, dones, its = [], [], []
        for s in range(0, llr.shape[0], chunk):
            o, d, i = self._decode_chunk(llr[s:s + chunk], max_iters)
            outs.append(o)
            dones.append(d)
            its.append(i)
        return torch.cat(outs), torch.cat(dones), torch.cat(its)

    def __repr__(self):
        return (f"LDPC(n={self.n}, k={self.k} (k_eff={self.k_eff}, shortened={self.n_short}), "
                f"dv={self.dv}, dc={self.dc}, 4-cycles={self.n_4cycles}, decoder={self.decoder}, iters={self.bp_iters})")
