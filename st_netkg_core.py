# -*- coding: utf-8 -*-
"""
ST-NetKG — Core-Mechanism Reference Implementation
==================================================

A *partial* reference implementation of the three coupled modules and the
training objective proposed in:

    "Spatio-Temporal Representation and Evolution Algorithm for Knowledge
     Graphs in Highly Dynamic Asymmetric Networks" (ST-NetKG).

This file implements the intellectual core of the method so that others can
read, verify, and adapt the mechanism:

    1. PC-DEC .................. Physics-Constrained Deep Embedded Clustering
                                (paper Sec. IV-A, Eq. (5)-(6))
    2. Complex rotational emb. . Asymmetric relation modelling in C^k
                                (paper Sec. IV-B, Eq. (7)-(8))
    3. EvolveGCN-O ............. Weight-evolving graph convolution; the GRU
                                evolves ONLY the fixed D x D weight, never the
                                |V|-sized node matrix (paper Sec. IV-C, Eq. (9)-(10))
    4. Objectives ............. Margin-ranking link loss + composite L_total
                                (paper Sec. IV-D)
    5. Time-aware sampler ..... ST-NetKG+ hard-negative sampler
                                (paper Sec. IV-D, Eq. (11)-(12), Algorithm 3)

--------------------------------------------------------------------------
DELIBERATELY NOT INCLUDED (kept in the authors' private codebase by design):
--------------------------------------------------------------------------
    * dataset loaders for UAV-Swarm-Sim / IoT-ISAC-Tele / OpenCity-Traffic /
      Syn-6G-Core, and the Syn-6G-Core NS-3 / MATLAB generation scripts;
    * the full training / evaluation harness that reproduces Tables IV-VI and
      the figures (logging, CD diagrams, per-degree slicing, OOV protocol, ...);
    * the baseline zoo (RotatE, TComplEx, DyERNIE, EvolveGCN, QSTGNN, ...);
    * the tuned per-dataset configuration files and pretrained checkpoints.

The `__main__` block runs a **shape + gradient smoke test on random toy data
only** (NOT the paper's datasets). It exists to prove the modules compose and
back-propagate, not to reproduce any reported number.

--------------------------------------------------------------------------
IMPLEMENTATION NOTES / DECISIONS (some paper details are underspecified):
--------------------------------------------------------------------------
  (a) Encoder / dims. We take the encoder literally as 128-64-32 (latent
      d_z = 32) used for clustering, plus a linear head d_z -> 2k that yields
      the complex embedding h in C^k (k = 128 => D = 2k = 256). If your actual
      code shares weights differently, adjust `PCDEC`.
  (b) Cannot-Link penalty. Implemented as a repulsive hinge
      lambda2 * max(0, m - d^2): a Cannot-Link pair is penalised when it lies
      closer than the margin m, pushing disjoint failure domains apart (Eq. (6)).
  (c) RotatE distance. `rotate_distance` returns the L1-over-dimensions of the
      per-dimension complex modulus (the standard RotatE score), matching the
      paper's ||h o r - t||_1 phrasing.
  (d) Weights are kept square (D x D) across GCN layers so the EvolveGCN-O
      GRU-as-recurrent-state trick is well defined; layer 0 already receives
      [Re||Im] in R^{|V| x 2k} = R^{|V| x D}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration (defaults follow Table III of the paper)
# ============================================================================
@dataclass
class STNetKGConfig:
    in_dim: int = 64                      # raw attribute dim a_i(t)
    # --- PC-DEC ---
    enc_hidden: Sequence[int] = (128, 64) # encoder trunk before latent
    d_z: int = 32                         # clustering latent dim
    n_clusters: int = 8                   # K
    alpha: float = 1.0                    # Student-t d.o.f.
    lambda1: float = 0.5                  # Must-Link coefficient
    lambda2: float = 0.3                  # Cannot-Link coefficient
    margin_m: float = 1.2                 # Cannot-Link repulsion margin
    # --- complex rotational embedding ---
    k: int = 128                          # complex dimension; D = 2k
    # --- evolutionary GCN ---
    n_layers: int = 3                     # L
    tau: float = 0.07                     # adjacency softmax temperature
    # --- objectives ---
    gamma: float = 2.0                    # margin-ranking margin
    beta: float = 0.1                     # cluster-loss weight
    eta: float = 0.05                     # modulus regularisation coefficient
    # --- time-aware negative sampler (ST-NetKG+) ---
    window_W: int = 5
    rho: float = 0.7
    sigma: float = 2.0

    @property
    def D(self) -> int:
        return 2 * self.k


# ============================================================================
# Complex algebra helpers (complex numbers kept as explicit (re, im) tensors)
# ============================================================================
def rotate(h_re: torch.Tensor, h_im: torch.Tensor, theta: torch.Tensor):
    """Hadamard phase rotation h o r with r = e^{j*theta} (|r| = 1). Eq. (7)."""
    r_re, r_im = torch.cos(theta), torch.sin(theta)
    out_re = h_re * r_re - h_im * r_im
    out_im = h_re * r_im + h_im * r_re
    return out_re, out_im


def rotate_distance(h_re, h_im, theta, t_re, t_im, eps: float = 1e-9):
    """d_r(h, t) = || h o r - t ||_1  (RotatE score). Eq. (8)."""
    hr_re, hr_im = rotate(h_re, h_im, theta)
    d_re, d_im = hr_re - t_re, hr_im - t_im
    modulus = torch.sqrt(d_re * d_re + d_im * d_im + eps)   # per-dim |.|
    return modulus.sum(dim=-1)                              # L1 over dims


def _pair_sqdist(z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance ||z_i - z_j||^2 for index pairs [2, P]."""
    zi, zj = z[pairs[0]], z[pairs[1]]
    return ((zi - zj) ** 2).sum(dim=-1)


# ============================================================================
# 1) Physics-Constrained Deep Embedded Clustering  (Sec. IV-A)
# ============================================================================
class PCDEC(nn.Module):
    """MLP autoencoder + DEC soft assignment + Must-/Cannot-Link penalties.

    The encoder doubles as the inductive map f_enc that projects raw physical
    attributes into C^k, enabling OOV nodes to be embedded by a single forward
    pass (no full-graph retraining). Domain priors act ONLY through the
    gradient of L_cluster; attributes never carry the pairwise priors as input
    features (standard constrained-DEC formulation).
    """

    def __init__(self, cfg: STNetKGConfig):
        super().__init__()
        self.cfg = cfg

        dims = [cfg.in_dim, *cfg.enc_hidden, cfg.d_z]
        enc: List[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        self.encoder = nn.Sequential(*enc[:-1])          # linear latent (drop last ReLU)

        rdims = dims[::-1]
        dec: List[nn.Module] = []
        for a, b in zip(rdims[:-1], rdims[1:]):
            dec += [nn.Linear(a, b), nn.ReLU()]
        self.decoder = nn.Sequential(*dec[:-1])          # linear reconstruction

        self.complex_proj = nn.Linear(cfg.d_z, 2 * cfg.k)  # -> [Re || Im]
        self.centers = nn.Parameter(torch.randn(cfg.n_clusters, cfg.d_z) * 0.1)

    # --- forward pieces ---
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction for the autoencoder pre-training stage (Alg. 1, line 1)."""
        return self.decoder(self.encoder(x))

    def to_complex(self, z: torch.Tensor):
        """Project latent -> complex embedding h in C^k, returned as (Re, Im)."""
        out = self.complex_proj(z)
        return out.chunk(2, dim=-1)

    # --- DEC distributions ---
    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        """Student-t soft assignment q_ij. Eq. (5)."""
        a = self.cfg.alpha
        dist2 = torch.cdist(z, self.centers) ** 2
        q = (1.0 + dist2 / a).pow(-(a + 1) / 2)
        return q / q.sum(dim=1, keepdim=True)

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """Sharpened auxiliary target p_ij."""
        f = q.sum(dim=0)
        p = (q ** 2) / f
        return p / p.sum(dim=1, keepdim=True)

    def cluster_loss(self, z, must_pairs, cannot_pairs):
        """L_cluster = KL(P||Q) + lambda1*L_M + lambda2*L_C (Eq. (6)).
        Cannot-Link L_C is a repulsive hinge (see module NOTE (b))."""
        cfg = self.cfg
        q = self.soft_assign(z)
        p = self.target_distribution(q).detach()
        kl = F.kl_div(q.clamp_min(1e-12).log(), p, reduction="batchmean")

        ml = _pair_sqdist(z, must_pairs).mean() if must_pairs.numel() else z.new_zeros(())
        if cannot_pairs.numel():
            cl = torch.clamp(cfg.margin_m - _pair_sqdist(z, cannot_pairs), min=0.0).mean()
        else:
            cl = z.new_zeros(())

        loss = kl + cfg.lambda1 * ml + cfg.lambda2 * cl
        return loss, {"kl": kl.detach(), "ml": ml.detach(), "cl": cl.detach()}


# ============================================================================
# 2) Complex-space relation rotation + asymmetric adjacency  (Sec. IV-B / IV-C)
# ============================================================================
class RelationRotation(nn.Module):
    """One learnable phase vector theta_r per relation (|r| = 1 by construction)."""

    def __init__(self, n_relations: int, k: int):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(n_relations, k))

    def score(self, h_re, h_im, t_re, t_im, rel_idx: torch.Tensor):
        """Asymmetric rotational distance for a batch of triplets."""
        return rotate_distance(h_re, h_im, self.theta[rel_idx], t_re, t_im)


def relation_adjacency(h_re, h_im, theta_r, edges, n_nodes, tau):
    """Calibrated relation-specific asymmetric adjacency (Sec. IV-C).

    `edges`: LongTensor [2, E_r] of observed directed (src -> dst) pairs of one
    relation. Softmax is taken over each source's observed out-neighbours,
    which enforces the O(|E|) sparse structure claimed in the paper (here
    realised densely on a small toy graph for readability).
    """
    src, dst = edges[0], edges[1]
    d = rotate_distance(h_re[src], h_im[src], theta_r.unsqueeze(0), h_re[dst], h_im[dst])
    logits = -d / tau

    A = h_re.new_full((n_nodes, n_nodes), float("-inf"))
    A[src, dst] = logits
    A = torch.softmax(A, dim=1)                 # normalise over tails
    return torch.nan_to_num(A, nan=0.0)         # isolated sources -> 0 row


class MultiRelationAdjacency(nn.Module):
    """Parameter-weighted aggregation of per-relation adjacencies (preserves
    multi-relational semantics instead of homogenising them)."""

    def __init__(self, n_relations: int):
        super().__init__()
        self.rel_weight = nn.Parameter(torch.zeros(n_relations))

    def build(self, h_re, h_im, theta, edges_per_rel, n_nodes, tau):
        w = torch.softmax(self.rel_weight, dim=0)
        A = h_re.new_zeros((n_nodes, n_nodes))
        for r, edges in enumerate(edges_per_rel):
            if edges.numel() == 0:
                continue
            A = A + w[r] * relation_adjacency(h_re, h_im, theta[r], edges, n_nodes, tau)
        return A


# ============================================================================
# 3) Weight-evolving graph convolution (EvolveGCN-O)  (Sec. IV-C, Eq. (9)-(10))
# ============================================================================
class MatrixGRU(nn.Module):
    """EvolveGCN-O recurrence: W_t = GRU(input=W_{t-1}, hidden=W_{t-1}).

    The D x D weight is treated as a batch of D row-vectors of dim D; a single
    small GRUCell (shared across rows) evolves it. Crucially the recurrence is
    independent of the node count |V|.
    """

    def __init__(self, D: int):
        super().__init__()
        self.cell = nn.GRUCell(D, D)

    def forward(self, W: torch.Tensor) -> torch.Tensor:  # W: [D, D]
        return self.cell(W, W)


class EvolveGCNO(nn.Module):
    """Stack of L weight-evolving GCN layers. Node features enter ONLY the GCN
    (Eq. (9)); the GRU never sees the |V|-sized node matrix (Eq. (10))."""

    def __init__(self, cfg: STNetKGConfig):
        super().__init__()
        D, L = cfg.D, cfg.n_layers
        self.n_layers = L
        self.grus = nn.ModuleList(MatrixGRU(D) for _ in range(L))
        self.W0 = nn.ParameterList(
            nn.Parameter(nn.init.orthogonal_(torch.empty(D, D))) for _ in range(L)
        )
        self._W: List[torch.Tensor] | None = None

    def reset_state(self):
        """Clear the recurrent weight state (call at the start of a window)."""
        self._W = None

    def step(self, H0: torch.Tensor, adj: torch.Tensor):
        """One snapshot. H0: [N, D] = [Re(H)||Im(H)]; adj: [N, N] normalised."""
        if self._W is None:
            W = [w for w in self.W0]                       # first step: init weights
        else:
            W = [gru(w) for gru, w in zip(self.grus, self._W)]  # evolve in time

        H = H0
        for l in range(self.n_layers):
            H = torch.relu(adj @ H @ W[l])                 # Eq. (9)
        self._W = W                                        # keep graph for BPTT in-window
        return H, W


# ============================================================================
# 4) Objectives  (Sec. IV-D)
# ============================================================================
def link_ranking_loss(pos_dist: torch.Tensor, neg_dist: torch.Tensor, gamma: float):
    """Margin ranking on distances: max(0, gamma + d(pos) - d(neg))."""
    return torch.clamp(gamma + pos_dist - neg_dist, min=0.0).mean()


def modulus_regularizer(h_re: torch.Tensor, h_im: torch.Tensor):
    """eta-term: keep entity modulus near 1 to prevent unbounded drift.
    Implemented as written in the paper, (||h||^2 - 1)^2 at the vector level."""
    mod2 = (h_re ** 2 + h_im ** 2).sum(dim=-1)
    return ((mod2 - 1.0) ** 2).mean()


# ============================================================================
# 5) Time-aware negative sampler (ST-NetKG+)  (Eq. (11)-(12), Algorithm 3)
# ============================================================================
class TimeAwareNegativeSampler:
    """Draws hard negatives from historically-active-but-currently-dormant tails.

    `history[tau]` is the iterable of tails active at snapshot `tau` for the
    query (h, r). This whole routine is already fully specified in the paper
    (Algorithm 3), so it leaks nothing beyond the manuscript.
    """

    def __init__(self, cfg: STNetKGConfig):
        self.cfg = cfg

    def sample(
        self,
        t_now: int,
        history: Dict[int, Iterable[int]],
        true_tails_now: set,
        n_entities: int,
        n_neg: int,
    ) -> List[int]:
        cfg = self.cfg
        recent, last_active = set(), {}
        for tau in range(max(0, t_now - cfg.window_W), t_now):
            for e in history.get(tau, ()):
                recent.add(e)
                last_active[e] = tau                       # most recent wins
        active_now = set(history.get(t_now, ()))
        pool = list(recent - active_now)                   # dormant pool D, Eq. (11)

        weights = None
        if pool:
            ages = torch.tensor([t_now - last_active[e] for e in pool], dtype=torch.float)
            weights = torch.softmax(-ages / cfg.sigma, dim=0)   # recency weights, Eq. (12)

        negatives: List[int] = []
        guard = 0
        while len(negatives) < n_neg and guard < 100 * n_neg:
            guard += 1
            if pool and torch.rand(()) < cfg.rho:
                e = pool[torch.multinomial(weights, 1).item()]  # recency-weighted hard neg
            else:
                e = int(torch.randint(n_entities, (1,)).item())  # uniform fallback
            if e in true_tails_now:                        # Filtered protocol
                continue
            negatives.append(e)
        return negatives


# ============================================================================
# Assembled model (thin wrapper wiring the three modules together)
# ============================================================================
class STNetKG(nn.Module):
    def __init__(self, cfg: STNetKGConfig, n_relations: int):
        super().__init__()
        self.cfg = cfg
        self.pcdec = PCDEC(cfg)
        self.relation = RelationRotation(n_relations, cfg.k)
        self.adjacency = MultiRelationAdjacency(n_relations)
        self.evolve = EvolveGCNO(cfg)
        # tiny forecasting head (Phase III): decode an edge weight from H^(L)
        self.forecast_head = nn.Sequential(
            nn.Linear(2 * cfg.D, cfg.D), nn.ReLU(), nn.Linear(cfg.D, 1)
        )

    def embed(self, x: torch.Tensor):
        """Raw attributes -> (latent z, complex embedding (h_re, h_im))."""
        z = self.pcdec.encode(x)
        h_re, h_im = self.pcdec.to_complex(z)
        return z, h_re, h_im

    def forecast_edge(self, H: torch.Tensor, edges: torch.Tensor):
        """Predict a scalar interaction state for candidate edges from H^(L)."""
        feat = torch.cat([H[edges[0]], H[edges[1]]], dim=-1)
        return self.forecast_head(feat).squeeze(-1)


# ============================================================================
# Smoke test on random TOY data (NOT the paper datasets)
# ============================================================================
def _toy_smoke_test():
    torch.manual_seed(0)
    cfg = STNetKGConfig(in_dim=64, k=128)      # paper-scale dims, tiny graph
    N, R, T = 40, 2, 4                          # nodes, relations, snapshots

    model = STNetKG(cfg, n_relations=R)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # random attributes and per-snapshot per-relation directed edges
    X = torch.randn(N, cfg.in_dim)
    def rand_edges(m):
        return torch.randint(0, N, (2, m))
    edges_seq = [[rand_edges(80) for _ in range(R)] for _ in range(T)]

    # random Must-/Cannot-Link priors (index pairs)
    must = torch.randint(0, N, (2, 20))
    cannot = torch.randint(0, N, (2, 20))

    sampler = TimeAwareNegativeSampler(cfg)
    # toy interaction history for one (h, r) query: tail sets per snapshot
    history = {t: set(int(v) for v in torch.randint(0, N, (10,))) for t in range(T)}

    model.train()
    opt.zero_grad()
    model.evolve.reset_state()

    z, h_re, h_im = model.embed(X)

    H_last = None
    for t in range(T):
        A = model.adjacency.build(h_re, h_im, model.relation.theta, edges_seq[t], N, cfg.tau)
        H0 = torch.cat([h_re, h_im], dim=-1)               # [N, D] = [Re||Im]
        H_last, _ = model.evolve.step(H0, A)               # weight-evolving GCN

    # --- link ranking loss on the current snapshot ---
    pos = edges_seq[-1][0]                                  # positives from relation 0
    rel0 = torch.zeros(pos.shape[1], dtype=torch.long)
    pos_d = model.relation.score(h_re[pos[0]], h_im[pos[0]],
                                 h_re[pos[1]], h_im[pos[1]], rel0)
    true_now = set(int(v) for v in pos[1])
    neg_tails = sampler.sample(T - 1, history, true_now, n_entities=N,
                               n_neg=pos.shape[1])
    neg_tails = torch.tensor(neg_tails, dtype=torch.long)
    neg_d = model.relation.score(h_re[pos[0]], h_im[pos[0]],
                                 h_re[neg_tails], h_im[neg_tails], rel0)
    l_link = link_ranking_loss(pos_d, neg_d, cfg.gamma)

    # --- composite objective L_total ---
    l_clu, parts = model.pcdec.cluster_loss(z, must, cannot)
    l_mod = modulus_regularizer(h_re, h_im)
    l_total = l_link + cfg.beta * l_clu + cfg.eta * l_mod

    l_total.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()

    # --- forecasting head sanity (Phase III wiring) ---
    fc = model.forecast_edge(H_last, pos)

    print("=== ST-NetKG core smoke test (toy random data) ===")
    print(f"nodes={N} relations={R} snapshots={T}  k={cfg.k}  D={cfg.D}")
    print(f"complex embedding h : Re{tuple(h_re.shape)} Im{tuple(h_im.shape)}")
    print(f"H^(L) node states   : {tuple(H_last.shape)}  (real, dim D)")
    print(f"forecast head out   : {tuple(fc.shape)}")
    print(f"L_link={l_link.item():.4f}  KL={parts['kl'].item():.4f} "
          f"ML={parts['ml'].item():.4f} CL={parts['cl'].item():.4f}")
    print(f"L_cluster={l_clu.item():.4f}  L_mod={l_mod.item():.4f}  "
          f"L_total={l_total.item():.4f}")
    print(f"grad-norm={gnorm.item():.4f}  -> backward + optimizer step OK")

    # a couple of invariants worth asserting in a real test suite
    assert H_last.shape == (N, cfg.D)
    assert torch.isfinite(l_total)
    assert gnorm.item() > 0, "no gradient reached the parameters"
    print("all shape/gradient assertions passed.")


if __name__ == "__main__":
    _toy_smoke_test()
