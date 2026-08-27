"""
Shared module for Week 2 — the tidy versions of what you derive in 02 and 03.

Imported by 04 and 05. Same pattern as gnn_layers.py last week: the numbered
files are where you work things out, this is what you build on.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ======================================================================
# Configuration
#
# CUTOFF is 5.2 rather than 5.0 deliberately. A cutoff that exactly equals a
# lattice parameter puts bonds precisely on the boundary, where inclusive vs
# strict comparison changes your edge count (see 02, part D). 5.2 avoids the
# common round-number collisions.
# ======================================================================

CUTOFF = 5.2
MAX_NEIGHBOURS: int | None = None     # None = uncapped; CGCNN uses 12
N_GAUSSIAN_BASIS = 40
MAX_Z = 100                           # atomic numbers 1..99


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ======================================================================
# The graph
# ======================================================================

@dataclass
class CrystalGraph:
    """One crystal, as tensors ready for a GNN.

    Attributes:
        z:          (N,)      atomic numbers
        edge_index: (2, E)    row 0 source, row 1 destination
        edge_dist:  (E,)      interatomic distance in Angstroms
        edge_offset:(E, 3)    periodic image offset, integers
        n_atoms:    int
        target:     float     the property we're predicting
        material_id:str
    """

    z: torch.Tensor
    edge_index: torch.Tensor
    edge_dist: torch.Tensor
    edge_offset: torch.Tensor
    n_atoms: int
    target: float
    material_id: str = ""

    def __repr__(self) -> str:
        return (f"CrystalGraph({self.material_id}, {self.n_atoms} atoms, "
                f"{self.edge_index.shape[1]} edges, "
                f"{self.edge_index.shape[1] / self.n_atoms:.1f} per atom)")


def structure_to_graph(structure, target: float, material_id: str = "",
                       cutoff: float = CUTOFF,
                       max_neighbours: int | None = MAX_NEIGHBOURS
                       ) -> CrystalGraph | None:
    """Convert a pymatgen Structure into a CrystalGraph.

    Uses pymatgen's get_neighbor_list, which is the vectorised, correct,
    periodic neighbour search you reimplemented from scratch in file 02.

    Returns None for structures that produce no edges — isolated atoms in a
    huge cell. Those are useless for a GNN (no messages can flow) and would
    poison a batch, so they're dropped at construction time rather than
    causing a confusing NaN later.
    """
    src, dst, offset, dist = structure.get_neighbor_list(cutoff)

    if len(src) == 0:
        return None

    if max_neighbours is not None:
        src, dst, offset, dist = _cap_neighbours(
            src, dst, offset, dist, len(structure), max_neighbours
        )

    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)

    return CrystalGraph(
        z=z,
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_dist=torch.tensor(dist, dtype=torch.float32),
        edge_offset=torch.tensor(np.asarray(offset), dtype=torch.float32),
        n_atoms=len(structure),
        target=float(target),
        material_id=material_id,
    )


def _cap_neighbours(src, dst, offset, dist, n_atoms: int, k: int):
    """Keep only the k nearest neighbours of each atom (CGCNN's approach)."""
    keep = []
    for atom in range(n_atoms):
        idx = np.nonzero(src == atom)[0]
        if len(idx) > k:
            idx = idx[np.argsort(dist[idx])[:k]]
        keep.append(idx)
    keep = np.concatenate(keep) if keep else np.array([], dtype=int)
    return src[keep], dst[keep], offset[keep], dist[keep]


# ======================================================================
# Batching — same trick as Week 1, now with edge offsets carried along
# ======================================================================

def collate(graphs: list[CrystalGraph]) -> dict:
    """Merge graphs into one big disconnected graph.

    Identical logic to Week 1's collate: concatenate nodes, offset each
    graph's edge indices by the running node count, record membership in
    `batch`. See 05_graph_regression.py for the full explanation.
    """
    z, dists, offsets, edges, batch, targets, n_atoms = [], [], [], [], [], [], []
    running = 0

    for gid, g in enumerate(graphs):
        z.append(g.z)
        dists.append(g.edge_dist)
        offsets.append(g.edge_offset)
        edges.append(g.edge_index + running)
        batch.append(torch.full((g.n_atoms,), gid, dtype=torch.long))
        targets.append(g.target)
        n_atoms.append(g.n_atoms)
        running += g.n_atoms

    return {
        "z": torch.cat(z),
        "edge_index": torch.cat(edges, dim=1),
        "edge_dist": torch.cat(dists),
        "edge_offset": torch.cat(offsets),
        "batch": torch.cat(batch),
        "y": torch.tensor(targets, dtype=torch.float32),
        "n_atoms": torch.tensor(n_atoms, dtype=torch.float32),
        "num_graphs": len(graphs),
    }


def to_device(batch: dict, dev: torch.device) -> dict:
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


# ======================================================================
# Featurisation
# ======================================================================

def gaussian_expansion(distances: torch.Tensor, centres: torch.Tensor,
                       width: float) -> torch.Tensor:
    """Expand scalar distances into smooth radial basis functions."""
    diff = distances.unsqueeze(-1) - centres.unsqueeze(0)
    return torch.exp(-(diff ** 2) / width ** 2)


# ======================================================================
# Model
# ======================================================================

class CGConvLayer(nn.Module):
    """Crystal graph convolution (Xie & Grossman, 2018).

        z_ij  = [ x_i , x_j , e_ij ]
        x_i' = x_i + sum_j  sigmoid(W_f z_ij + b_f) * softplus(W_s z_ij + b_s)
    """

    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        concat = 2 * node_dim + edge_dim
        self.filter_lin = nn.Linear(concat, node_dim)
        self.core_lin = nn.Linear(concat, node_dim)
        self.bn = nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        z = torch.cat([x[dst], x[src], edge_attr], dim=-1)

        messages = torch.sigmoid(self.filter_lin(z)) * nn.functional.softplus(
            self.core_lin(z)
        )

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages)
        # Mean aggregation: normalise by in-degree so a node's aggregated signal
        # is the AVERAGE neighbour message, independent of coordination number.
        # Unnormalised sum let high-coordination cells push activations outside
        # BatchNorm's eval-time running stats and explode predictions on a few
        # structures (see the Week 3 matbench_log_gvrh runs). clamp(min=1)
        # leaves edge-less nodes untouched — their agg is already zero.
        deg = torch.bincount(dst, minlength=x.size(0)).clamp(min=1)
        agg = agg / deg.unsqueeze(-1).to(agg.dtype)
        return nn.functional.softplus(x + self.bn(agg))


def global_pool(x, batch, num_graphs, mode: str = "mean"):
    out = torch.zeros(num_graphs, x.shape[1], dtype=x.dtype, device=x.device)
    out.index_add_(0, batch, x)
    if mode == "sum":
        return out
    if mode == "mean":
        counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1)
        return out / counts.unsqueeze(-1).to(x.dtype)
    raise ValueError(mode)


class CGCNN(nn.Module):
    """Crystal Graph Convolutional Neural Network.

    Pooling defaults to "mean" because formation energy is reported PER ATOM —
    an intensive property. See Week 1, file 05, part D for why that isn't a
    detail.
    """

    def __init__(self, node_dim: int = 64, n_layers: int = 3,
                 n_basis: int = N_GAUSSIAN_BASIS, cutoff: float = CUTOFF,
                 pool: str = "mean", readout_dim: int = 128) -> None:
        super().__init__()

        self.embedding = nn.Embedding(MAX_Z, node_dim)
        self.register_buffer("centres", torch.linspace(0.0, cutoff, n_basis))
        self.basis_width = cutoff / n_basis * 2

        self.convs = nn.ModuleList(
            [CGConvLayer(node_dim, n_basis) for _ in range(n_layers)]
        )
        self.pool = pool
        self.readout = nn.Sequential(
            nn.Linear(node_dim, readout_dim),
            nn.Softplus(),
            nn.Linear(readout_dim, 1),
        )

    def embed(self, batch: dict) -> torch.Tensor:
        x = self.embedding(batch["z"])
        edge_attr = gaussian_expansion(
            batch["edge_dist"], self.centres, self.basis_width
        )
        for conv in self.convs:
            x = conv(x, batch["edge_index"], edge_attr)
        return global_pool(x, batch["batch"], batch["num_graphs"], self.pool)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.readout(self.embed(batch)).squeeze(-1)


# ======================================================================
# Caching
# ======================================================================

def cache_key(**params) -> str:
    """Hash of the parameters that affect graph construction.

    This is the answer to README checkpoint question 6. Cached graphs are only
    valid for the settings that built them. Change CUTOFF and every cached file
    is silently wrong — the graphs load fine, they're just built to the old
    radius. Putting the parameters in the filename makes that impossible:
    change a parameter, get a cache miss, rebuild.

    Caching by content, not by name, is the general habit. Adopt it now.
    """
    blob = json.dumps(params, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest()[:10]


def save_graphs(graphs: list[CrystalGraph], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graphs, f)


def load_graphs(path: Path) -> list[CrystalGraph]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ======================================================================
# Baselines — carried over from Week 1, because they still matter
# ======================================================================

def mean_baseline(train: list[CrystalGraph], test: list[CrystalGraph]) -> float:
    mu = float(np.mean([g.target for g in train]))
    return float(np.mean([abs(g.target - mu) for g in test]))


def composition_baseline(train: list[CrystalGraph],
                         test: list[CrystalGraph]) -> float:
    """Least squares on element FRACTIONS. No geometry at all.

    Fractions rather than raw counts, because the target is per-atom
    (intensive). Regressing a per-atom quantity on raw counts would let the
    model cheat via cell size.

    This is the bar your GNN must clear. In real materials informatics,
    composition-only models are surprisingly strong, and papers have been
    embarrassed by not checking.
    """
    def featurise(graphs):
        X = np.zeros((len(graphs), MAX_Z + 1))
        for i, g in enumerate(graphs):
            counts = np.bincount(g.z.numpy(), minlength=MAX_Z)
            X[i, :MAX_Z] = counts / counts.sum()
            X[i, MAX_Z] = 1.0
        return X

    X_tr, X_te = featurise(train), featurise(test)
    y_tr = np.array([g.target for g in train])
    y_te = np.array([g.target for g in test])

    coef, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    return float(np.mean(np.abs(y_te - X_te @ coef)))
