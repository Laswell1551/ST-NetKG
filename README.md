# ST-NetKG

Reference implementation of the **core mechanisms** of **ST-NetKG**, a
spatio-temporal network knowledge graph framework for highly dynamic and
asymmetric 6G networks, from the paper:

> J. Lin, S. Yan, Q. Qian, B. Ren, M. Peng, X. Zhong, Y. Song,
> *"Spatio-Temporal Representation and Evolution Algorithm for Knowledge Graphs
> in Highly Dynamic Asymmetric Networks."*

ST-NetKG bridges physical-layer ISAC sensing and logical network deduction
through three coupled modules:

1. **Physics-Constrained Deep Embedded Clustering (PC-DEC)** — purifies noisy
   multi-modal attributes into semantic entities using Must-Link / Cannot-Link
   domain priors.
2. **Complex-space (ℂᵏ) rotational embedding** — models directed, non-reciprocal
   interactions (e.g. unidirectional interference) as relation-specific phase
   rotations, so asymmetry `d_r(h,t) ≠ d_r(t,h)` is preserved.
3. **Weight-evolving graph convolution (EvolveGCN-O)** — tracks topological
   phase transitions by evolving the fixed-size GCN weight matrix over time
   instead of aggregating stale node states. The `|V|`-sized node matrix never
   enters the recurrence, so the model is well-defined under arbitrary
   topological churn and supports out-of-vocabulary nodes.

---

## Scope of this repository

This repository is a **clean, self-contained reference implementation of the
method and its training recipe**, meant for readers who want to understand,
verify, and build on the mechanism.

**Included**

| File | Purpose |
| --- | --- |
| `st_netkg_core.py` | The three core modules, the training objectives, and the time-aware negative sampler. Runs a shape/gradient smoke test. |
| `train_toy.py` | A minimal end-to-end training loop (pre-training + K-Means init + joint optimisation with early stopping) on a synthetic toy graph. |

**Not included (by design).** To keep this release focused on the method, it
does **not** ship the full experimental reproduction package:

- the simulated / proprietary datasets (**UAV-Swarm-Sim, IoT-ISAC-Tele,
  OpenCity-Traffic, Syn-6G-Core**) and their NS-3 / MATLAB generators;
- dataset loaders and preprocessing pipelines;
- the complete evaluation harness that produces the reported tables and figures
  (full MRR / Hits@k / ACC / NMI / ARI / MAE / RMSE / MAPE / PTDD, Friedman–Nemenyi
  significance tests and CD diagrams, per-in-degree slicing, the OOV protocol);
- the baseline models and the tuned per-dataset configurations / checkpoints.

The scripts here operate on **randomly generated toy data only**. They
demonstrate the *mechanics* of the method.

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
# 1) Core-module shape/gradient smoke test on toy data
python st_netkg_core.py

# 2) Minimal end-to-end training on a toy dynamic graph
#    (runs both the base uniform-negative variant and ST-NetKG+)
python train_toy.py
```

The smoke test prints the tensor shapes flowing through the three modules and
confirms that gradients propagate. The training script pre-trains the
autoencoder, initialises the cluster centers via K-Means, then runs joint
optimisation with early stopping. Because the toy graph is random noise with no
learnable structure, the validation MRR stays near chance — this is expected;
the goal is to show that the loop trains and converges.

---

## Method ↔ code map

| Component | Paper | Code |
| --- | --- | --- |
| Physics-Constrained Deep Embedded Clustering | Sec. IV-A, Eq. (5)–(6) | `PCDEC` |
| Complex rotational embedding (phase rotation, RotatE score) | Sec. IV-B, Eq. (7)–(8) | `RelationRotation`, `rotate`, `rotate_distance` |
| Relation-specific asymmetric adjacency (calibrated softmax) | Sec. IV-C | `relation_adjacency`, `MultiRelationAdjacency` |
| Weight-evolving graph convolution (EvolveGCN-O) | Sec. IV-C, Eq. (9)–(10) | `MatrixGRU`, `EvolveGCNO` |
| Margin-ranking link loss + modulus regulariser (`L_total`) | Sec. IV-D | `link_ranking_loss`, `modulus_regularizer` |
| Time-aware hard-negative sampler (ST-NetKG+) | Sec. IV-D, Eq. (11)–(12), Alg. 3 | `TimeAwareNegativeSampler` |
| Training recipe (pre-train → joint optimisation) | Alg. 1 (line 1), Sec. IV-D, Table III | `train_toy.py` |

---

## Configuration

All hyperparameters live in the `STNetKGConfig` dataclass and default to the
values in Table III of the paper (e.g. `k=128`, `L=3`, `tau=0.07`, `gamma=2.0`,
`beta=0.1`, `eta=0.05`, Adam `1e-3`, negative-sampling ratio `1:5`). The toy
scripts shrink a few dimensions purely for speed.

## Implementation notes

- Complex numbers are represented as explicit `(real, imag)` tensor pairs for
  readability and portability.
- `train_toy.py` includes an auxiliary forecasting regression term
  (`w_forecast`) that supervises the weight-evolving GCN / forecast head for the
  Phase-III topology-forecasting task; set `w_forecast=0.0` to optimise the
  link + cluster + modulus objective alone.
- `use_time_aware=False` selects the base ST-NetKG (uniform negatives);
  `True` selects **ST-NetKG+** (time-aware hard negatives).

---

## Citation

If you find this reference useful, please cite the paper:

```bibtex
@article{lin2026stnetkg,
  title   = {Spatio-Temporal Representation and Evolution Algorithm for
             Knowledge Graphs in Highly Dynamic Asymmetric Networks},
  author  = {Lin, Jiaqi and Yan, Shi and Qian, Qijie and Ren, Baoquan and
             Peng, Mugen and Zhong, Xudong and Song, Yangzi},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2026},
  note    = {Under review}
}
```

## License

Add your preferred license here before publishing (e.g. MIT, Apache-2.0, or a
custom research license).
