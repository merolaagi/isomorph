"""Comparison math shared by the lab and the hub analyses.

Everything here is symmetry-aware: raw weight cosine is reported only as a
baseline, because a network can be re-expressed (permute neurons, rotate the
residual basis) without changing a single output.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _t(x):
    return x.detach().cpu().double() if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64)


def cosine(a, b):
    a, b = _t(a).flatten(), _t(b).flatten()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else 0.0


def linear_cka(x, y):
    """Linear CKA between activation matrices (n, d1) and (n, d2). Invariant to
    orthogonal transforms and isotropic scaling of either side."""
    x, y = _t(x), _t(y)
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    # Use the n x n Gram form when features outnumber samples.
    if x.shape[1] > x.shape[0] or y.shape[1] > y.shape[0]:
        kx, ky = x @ x.T, y @ y.T
        hsic = (kx * ky).sum()
        den = torch.sqrt((kx * kx).sum() * (ky * ky).sum())
    else:
        hsic = (x.T @ y).pow(2).sum()
        den = torch.sqrt((x.T @ x).pow(2).sum() * (y.T @ y).pow(2).sum())
    return float(hsic / den) if den > 0 else 0.0


def mutual_knn(x, y, k=10):
    """Fraction of shared k-nearest neighbours (Huh et al. 2024, Platonic
    Representation Hypothesis). Works across different widths."""
    x, y = _t(x), _t(y)
    n = x.shape[0]
    k = max(1, min(k, n - 1))

    def knn(z):
        z = torch.nn.functional.normalize(z - z.mean(0), dim=1)
        s = z @ z.T
        s.fill_diagonal_(-1e9)
        return s.topk(k, dim=1).indices

    a, b = knn(x), knn(y)
    shared = 0
    for i in range(n):
        shared += len(set(a[i].tolist()) & set(b[i].tolist()))
    return shared / (n * k)


def procrustes(src, dst):
    """Orthogonal R minimising ||src @ R - dst||_F (src, dst: n x d)."""
    m = _t(src).T @ _t(dst)
    u, _, vt = torch.linalg.svd(m)
    return (u @ vt).float()


def affine_fit(src, dst, ridge=1e-4):
    """Least-squares affine map dst ~ [src, 1] @ W, with a small ridge."""
    s, d = _t(src), _t(dst)
    s1 = torch.cat([s, torch.ones(s.shape[0], 1, dtype=s.dtype)], 1)
    a = s1.T @ s1
    a += ridge * torch.eye(a.shape[0], dtype=a.dtype) * a.diagonal().mean()
    w = torch.linalg.solve(a, s1.T @ d)
    return w.float()


def apply_affine(src, w):
    s = src.float()
    return torch.cat([s, torch.ones(s.shape[0], 1)], 1) @ w


def correlation_matrix(a, b):
    """Pearson correlation between columns of a (n, m1) and b (n, m2)."""
    a, b = _t(a), _t(b)
    a = a - a.mean(0)
    b = b - b.mean(0)
    na, nb = a.norm(dim=0), b.norm(dim=0)
    c = (a.T @ b) / (na[:, None] * nb[None, :] + 1e-12)
    c[na < 1e-9, :] = 0
    c[:, nb < 1e-9] = 0
    return c.float()


def match_units(corr):
    """Hungarian matching that maximises total correlation. Returns perm such that
    unit perm[i] of B pairs with unit i of A, and the matched correlations."""
    c = corr.numpy() if isinstance(corr, torch.Tensor) else np.asarray(corr)
    rows, cols = linear_sum_assignment(-c)
    perm = np.empty(c.shape[0], dtype=np.int64)
    perm[rows] = cols
    return torch.as_tensor(perm), torch.as_tensor(c[rows, cols])


def svals(w, k=None):
    w = _t(w)
    if w.ndim != 2:
        w = w.reshape(w.shape[0], -1)
    s = torch.linalg.svdvals(w)
    return s[:k] if k else s


def effective_rank(s):
    """exp(entropy) of the normalised singular value distribution (Roy & Vetterli)."""
    s = _t(s)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def stable_rank(s):
    s = _t(s)
    return float((s ** 2).sum() / (s[0] ** 2)) if len(s) and s[0] > 0 else 0.0


def energy_rank(s, frac=0.9):
    """Smallest r such that the top-r singular values hold `frac` of the squared energy."""
    s = _t(s)
    e = (s ** 2).cumsum(0)
    if e[-1] <= 0:
        return 0
    return int((e / e[-1] < frac).sum().item()) + 1


def spectrum_distance(sa, sb):
    """L1 distance between normalised singular value profiles (0 = identical shape)."""
    sa, sb = _t(sa), _t(sb)
    n = min(len(sa), len(sb))
    if n == 0 or float(sa[:n].sum()) <= 0 or float(sb[:n].sum()) <= 0:
        return 0.0 if float(sa[:n].sum()) == float(sb[:n].sum()) else 1.0
    pa, pb = sa[:n] / sa[:n].sum(), sb[:n] / sb[:n].sum()
    return float((pa - pb).abs().sum() / 2)


def downsample(v, n=64):
    v = list(map(float, v))
    if len(v) <= n:
        return v
    idx = np.linspace(0, len(v) - 1, n).round().astype(int)
    return [v[i] for i in idx]
