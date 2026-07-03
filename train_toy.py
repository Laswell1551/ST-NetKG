# -*- coding: utf-8 -*-
"""
ST-NetKG — Minimal Training-Loop Skeleton (TOY DATA ONLY)
=========================================================

Shows *how* ST-NetKG is trained, using the core modules from
`st_netkg_core.py`. It is intentionally minimal:

    Stage 0  -  autoencoder pre-training (reconstruction MSE) + K-Means
                center initialisation                (paper Alg. 1, line 1)
    Stage 1  -  end-to-end joint optimisation of L_total with early stopping
                                                     (paper Sec. IV-D, Table III)

--------------------------------------------------------------------------
WHAT THIS IS NOT (kept private by design, consistent with st_netkg_core.py):
--------------------------------------------------------------------------
  * NOT a reproduction script. The graph below is a random TOY generator, not
    UAV-Swarm-Sim / IoT-ISAC-Tele / OpenCity-Traffic / Syn-6G-Core, and there
    is no dataset loader or data preprocessing pipeline here.
  * NO full evaluation harness: the validation signal is a bare-bones filtered
    MRR on toy triplets, used only to drive early stopping. The paper's real
    metric suite (MRR/Hits@k/ACC/NMI/ARI/MAE/RMSE/MAPE/PTDD, CD diagrams,
    per-in-degree slicing, OOV protocol) is not reproduced.
  * NO baselines, no tuned per-dataset configs, no checkpoints.

--------------------------------------------------------------------------
NOTE on the training objective:
--------------------------------------------------------------------------
  Training combines the representation objective
      L_total = L_link + beta*L_cluster + eta*L_modulus        (Sec. IV-D)
  with an auxiliary forecasting regression `L_forecast` (weight `w_forecast`)
  that supervises the weight-evolving GCN / forecast head for the Phase-III
  topology-forecasting task. Set `w_forecast=0.0` to optimise the
  link + cluster + modulus objective alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from st_netkg_core import (
    STNetKG,
    STNetKGConfig,
    TimeAwareNegativeSampler,
    link_ranking_loss,
    modulus_regularizer,
)


# ============================================================================
# Toy dynamic-graph generator  (random data — NOT the paper datasets)
# ============================================================================
@dataclass
class ToyDynamicGraph:
    """A small synthetic dynamic multi-relational graph for the skeleton.

    Edges evolve with churn (persist a fraction, drop/add the rest) so that the
    time-aware sampler has a non-empty 'recently-active-but-now-dormant' pool.
    Attributes are static here for readability; in the real setting a_i(t) is
    time-varying.
    """
    n_nodes: int = 60
    n_relations: int = 2
    n_train_snapshots: int = 6      # history window used for training
    in_dim: int = 64
    edges_per_rel: int = 120
    churn: float = 0.3              # fraction of edges refreshed each snapshot
    n_must: int = 30
    n_cannot: int = 30
    n_val_pos: int = 40             # held-out positive triplets for val ranking
    seed: int = 0

    # populated in __post_init__
    X: torch.Tensor = field(init=False)
    edges_seq: List[List[torch.Tensor]] = field(init=False)     # [T][R] -> [2, E]
    future_edges: torch.Tensor = field(init=False)              # [2, E] next-step (rel 0)
    future_weights: torch.Tensor = field(init=False)            # [E] toy regression target
    must_pairs: torch.Tensor = field(init=False)                # [2, P]
    cannot_pairs: torch.Tensor = field(init=False)              # [2, P]
    val_pos: torch.Tensor = field(init=False)                   # [2, V] (head, tail) rel 0
    all_true_tails: Dict[int, set] = field(init=False)          # filter set per head, rel 0

    def __post_init__(self):
        g = torch.Generator().manual_seed(self.seed)
        N, R = self.n_nodes, self.n_relations
        self.X = torch.randn(N, self.in_dim, generator=g)

        def rand_edges(m):
            return torch.randint(0, N, (2, m), generator=g)

        # relation-0 edges evolve with churn; other relations resampled freely
        base0 = rand_edges(self.edges_per_rel)
        self.edges_seq = []
        for _ in range(self.n_train_snapshots):
            keep = int(self.edges_per_rel * (1 - self.churn))
            perm = torch.randperm(self.edges_per_rel, generator=g)
            kept = base0[:, perm[:keep]]
            fresh = rand_edges(self.edges_per_rel - keep)
            base0 = torch.cat([kept, fresh], dim=1)
            snap = [base0.clone()] + [rand_edges(self.edges_per_rel) for _ in range(R - 1)]
            self.edges_seq.append(snap)

        # a 'future' snapshot (rel 0) with a toy continuous target to forecast
        self.future_edges = rand_edges(self.edges_per_rel)
        h, t = self.future_edges
        self.future_weights = (
            0.01 * (h.float() - t.float()) + 0.1 * torch.randn(h.shape[0], generator=g)
        )

        self.must_pairs = torch.randint(0, N, (2, self.n_must), generator=g)
        self.cannot_pairs = torch.randint(0, N, (2, self.n_cannot), generator=g)

        # validation positives (rel 0) + filter set of all true tails per head
        self.val_pos = rand_edges(self.n_val_pos)
        self.all_true_tails = {}
        for snap in self.edges_seq:
            for a, b in snap[0].t().tolist():
                self.all_true_tails.setdefault(a, set()).add(b)
        for a, b in self.val_pos.t().tolist():
            self.all_true_tails.setdefault(a, set()).add(b)

    # history of active tails for a given head under relation 0, per snapshot
    def head_history(self, head: int) -> Dict[int, set]:
        hist: Dict[int, set] = {}
        for t, snap in enumerate(self.edges_seq):
            src, dst = snap[0]
            hist[t] = set(dst[src == head].tolist())
        return hist


# ============================================================================
# Helpers: K-Means center init, negative sampling, toy validation MRR
# ============================================================================
def kmeans_init(z: torch.Tensor, k: int, iters: int = 20) -> torch.Tensor:
    """Lightweight Lloyd's K-Means to initialise cluster centers (Alg. 1, line 1)."""
    idx = torch.randperm(z.shape[0])[:k]
    centers = z[idx].clone()
    for _ in range(iters):
        assign = torch.cdist(z, centers).argmin(dim=1)
        for c in range(k):
            m = assign == c
            if m.any():
                centers[c] = z[m].mean(dim=0)
    return centers


def sample_negatives(
    data: ToyDynamicGraph,
    heads: torch.Tensor,
    n_neg: int,
    t_now: int,
    sampler: TimeAwareNegativeSampler | None,
) -> torch.Tensor:
    """Return [P, n_neg] corrupted tails. If `sampler` is None -> uniform (base
    ST-NetKG); else -> time-aware hard negatives (ST-NetKG+)."""
    P, N = heads.shape[0], data.n_nodes
    out = torch.empty(P, n_neg, dtype=torch.long)
    for i, h in enumerate(heads.tolist()):
        true_now = data.all_true_tails.get(h, set())
        if sampler is None:
            negs = []
            while len(negs) < n_neg:
                e = int(torch.randint(N, (1,)).item())
                if e not in true_now:
                    negs.append(e)
        else:
            negs = sampler.sample(t_now, data.head_history(h), true_now, N, n_neg)
        out[i] = torch.tensor(negs, dtype=torch.long)
    return out


@torch.no_grad()
def toy_filtered_mrr(model: STNetKG, data: ToyDynamicGraph) -> float:
    """Bare-bones filtered MRR on toy val positives (NOT the paper's harness).

    Ranks each positive's true tail against all entities by the rotational
    distance d_r(h, .), removing other known-true tails (Filtered protocol)."""
    _, h_re, h_im = model.embed(data.X)
    N = data.n_nodes
    rel0 = torch.zeros(N, dtype=torch.long)
    rr = 0.0
    heads, tails = data.val_pos
    for h, t in zip(heads.tolist(), tails.tolist()):
        hr = h_re[h].expand(N, -1)
        hi = h_im[h].expand(N, -1)
        d = model.relation.score(hr, hi, h_re, h_im, rel0)      # distance to every tail
        filt = data.all_true_tails.get(h, set()) - {t}
        if filt:
            d[torch.tensor(sorted(filt))] = float("inf")        # mask other positives
        rank = 1 + (d < d[t]).sum().item()
        rr += 1.0 / rank
    return rr / heads.shape[0]


# ============================================================================
# Training
# ============================================================================
def pretrain_autoencoder(model: STNetKG, X: torch.Tensor, epochs: int, lr: float):
    """Stage 0: reconstruction pre-training, then K-Means center init."""
    opt = torch.optim.Adam(
        list(model.pcdec.encoder.parameters()) + list(model.pcdec.decoder.parameters()),
        lr=lr,
    )
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(model.pcdec.reconstruct(X), X)
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = model.pcdec.encode(X)
        model.pcdec.centers.data = kmeans_init(z, model.cfg.n_clusters)
    return loss.item()


def train(
    model: STNetKG,
    cfg: STNetKGConfig,
    data: ToyDynamicGraph,
    *,
    use_time_aware: bool = True,     # True -> ST-NetKG+, False -> base ST-NetKG
    n_neg: int = 5,                  # negative sampling ratio 1:5 (Table III)
    w_forecast: float = 1.0,         # 0.0 -> link+cluster+modulus objective only
    max_epochs: int = 100,           # Table III
    pretrain_epochs: int = 30,
    patience: int = 10,              # early stop (Table III)
    lr: float = 1e-3,                # Adam / 1e-3 (Table III)
    log_every: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    N, T = data.n_nodes, data.n_train_snapshots
    sampler = TimeAwareNegativeSampler(cfg) if use_time_aware else None

    variant = "ST-NetKG+ (time-aware)" if use_time_aware else "ST-NetKG (uniform)"
    print(f"\n=== training {variant} | seed={seed} | toy data N={N} T={T} ===")

    ae_loss = pretrain_autoencoder(model, data.X, pretrain_epochs, lr)
    print(f"stage-0 pre-train reconstruction MSE = {ae_loss:.4f}; centers K-Means-init done")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_mrr, best_epoch, wait = -1.0, -1, 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        model.evolve.reset_state()

        # --- forward over the temporal window (Alg. 2) ---
        z, h_re, h_im = model.embed(data.X)
        H_last = None
        for t in range(T):
            A = model.adjacency.build(
                h_re, h_im, model.relation.theta, data.edges_seq[t], N, cfg.tau
            )
            H0 = torch.cat([h_re, h_im], dim=-1)          # [N, D] = [Re||Im]
            H_last, _ = model.evolve.step(H0, A)          # evolve weights + message pass

        # --- L_link (rotational margin) with (time-aware) negatives ---
        pos = data.edges_seq[-1][0]                       # rel-0 positives, last snapshot
        if pos.shape[1] > 256:                            # batch cap (Table III: batch 256)
            pos = pos[:, torch.randperm(pos.shape[1])[:256]]
        rel0 = torch.zeros(pos.shape[1], dtype=torch.long)
        d_pos = model.relation.score(h_re[pos[0]], h_im[pos[0]],
                                     h_re[pos[1]], h_im[pos[1]], rel0)
        neg_tails = sample_negatives(data, pos[0], n_neg, T - 1, sampler)   # [P, n_neg]
        heads_rep = pos[0].unsqueeze(1).expand(-1, n_neg).reshape(-1)
        rel_rep = torch.zeros(heads_rep.shape[0], dtype=torch.long)
        d_neg = model.relation.score(
            h_re[heads_rep], h_im[heads_rep],
            h_re[neg_tails.reshape(-1)], h_im[neg_tails.reshape(-1)], rel_rep,
        ).view(pos.shape[1], n_neg)
        l_link = link_ranking_loss(
            d_pos.unsqueeze(1).expand_as(d_neg).reshape(-1), d_neg.reshape(-1), cfg.gamma
        )

        # --- L_cluster + modulus regulariser ---
        l_clu, parts = model.pcdec.cluster_loss(z, data.must_pairs, data.cannot_pairs)
        l_mod = modulus_regularizer(h_re, h_im)

        # --- auxiliary forecasting regression (Phase-III topology forecasting) ---
        pred = model.forecast_edge(H_last, data.future_edges)
        l_fc = F.mse_loss(pred, data.future_weights)

        l_total = l_link + cfg.beta * l_clu + cfg.eta * l_mod + w_forecast * l_fc
        l_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        # --- validation + early stopping ---
        val_mrr = toy_filtered_mrr(model, data)
        if val_mrr > best_mrr:
            best_mrr, best_epoch, wait = val_mrr, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if epoch % log_every == 0 or wait == 0:
            print(f"epoch {epoch:3d} | L_total {l_total.item():.4f} "
                  f"(link {l_link.item():.3f}, clu {l_clu.item():.3f}, "
                  f"mod {l_mod.item():.3f}, fc {l_fc.item():.3f}) | "
                  f"val MRR {val_mrr:.4f}{'  *' if wait == 0 else ''}")
        if wait >= patience:
            print(f"early stop at epoch {epoch} (no val gain for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"best val MRR {best_mrr:.4f} @ epoch {best_epoch}")
    return {"best_val_mrr": best_mrr, "best_epoch": best_epoch}


if __name__ == "__main__":
    cfg = STNetKGConfig(in_dim=64, k=64)          # k=64 keeps the toy run snappy
    data = ToyDynamicGraph(in_dim=cfg.in_dim, seed=0)

    # base ST-NetKG (uniform negatives)
    model = STNetKG(cfg, n_relations=data.n_relations)
    train(model, cfg, data, use_time_aware=False, max_epochs=40, patience=8, seed=0)

    # ST-NetKG+ (time-aware hard negatives) -- fresh model, same toy graph
    model = STNetKG(cfg, n_relations=data.n_relations)
    train(model, cfg, data, use_time_aware=True, max_epochs=40, patience=8, seed=0)
