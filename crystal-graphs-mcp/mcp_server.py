"""
MCP server exposing this project's crystal-graph tooling to an agent.

Week 2 of the AI-skills track. The point is not that querying the Materials
Project through an agent is revolutionary — it's that writing an MCP server
teaches you the protocol that every agent integration now runs on, and you
happen to have a codebase worth exposing sitting right here.

The tools that matter are the last two. Anyone can wrap a REST API; wrapping
YOUR pipeline means you can ask an agent "what's the edge count per atom for
mp-149 at a 6.0 Å cutoff" and it computes the real answer using the same
`structure_to_graph` your model trains on, rather than guessing from memory.

    pip install "mcp[cli]"
    export MP_API_KEY="..."
    python mcp_server.py                 # stdio transport

Register with Hermes via `hermes tools`, or in an MCP client config:

    {"mcpServers": {"crystal-graphs": {
        "command": "python",
        "args": ["/Users/aaronlin/Documents/ML/blog/crystal-graphs-mcp/mcp_server.py"],
        "env": {"MP_API_KEY": "..."}
    }}}

STATUS — READ THIS
------------------
Be precise about what has and hasn't been checked, because this file's whole
value is that its numbers come from your real pipeline.

  - The *underlying pymatgen behaviour* the local tools depend on — neighbour
    lists, shell distances, self-image edge counts, edges per atom — was
    verified directly against pymatgen 2025.10.7. That check is what caught
    the fcc Cu error in README_Week2.md.
  - `test_mcp_server.py` covers `graph_stats` and `compare_cutoffs` end to end.
    It HAS now been executed against pymatgen 2025.10.7 and torch 2.5.1, and
    every check passes — including the fcc Cu 42/54 cutoff discontinuity. Rerun
    it after any change to `crystal_graph.py`:
        python test_mcp_server.py
  - The three MP-backed tools have NOT been run against the live API, for the
    same reason `04_download_mp.py` hasn't. Field names and the `chemsys` /
    `elements` filters are the likely drift points. Run `00_check_setup.py`
    before trusting any output from them.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from crystal_graph import CUTOFF, structure_to_graph

mcp = FastMCP("crystal-graphs")


# Chemistries that show up in biomedical materials. Not a rigorous list —
# a starting filter, so searches can be narrowed to things worth looking at.
# Ca-P for bone mineral, Mg for biodegradable implants, Ti/Zr for permanent
# implants and dental, Ta/Nb as radiopaque and alloying additions.
BIOMEDICAL_ELEMENT_SETS = {
    "calcium_phosphate": ["Ca", "P", "O"],
    "biodegradable_mg": ["Mg"],
    "titanium_implant": ["Ti"],
    "zirconia_dental": ["Zr", "O"],
    "tantalum_niobium": ["Ta", "Nb"],
}


def _client():
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError(
            "MP_API_KEY is not set. `export MP_API_KEY=...` before starting "
            "the server — do not paste it into a tool call."
        )
    from mp_api.client import MPRester
    return MPRester(key)


# ======================================================================
# Materials Project — unverified against the live API, see header
# ======================================================================


@mcp.tool()
def search_materials(elements: list[str], max_sites: int = 30,
                     max_results: int = 20,
                     require_exact_system: bool = False) -> list[dict[str, Any]]:
    """Search the Materials Project for structures containing given elements.

    Args:
        elements: element symbols, e.g. ["Mg", "O"]
        max_sites: skip cells larger than this — big cells are expensive to
            graph and add little to a first model
        max_results: cap on returned entries
        require_exact_system: if True, return only structures whose element set
            is exactly `elements`. If False, return anything containing them.

    Returns summary records with material_id, formula, formation energy per
    atom, energy above hull, band gap, and site count.
    """
    with _client() as mpr:
        kwargs: dict[str, Any] = {
            "num_sites": (1, max_sites),
            "fields": ["material_id", "formula_pretty", "elements",
                       "formation_energy_per_atom", "energy_above_hull",
                       "band_gap", "nsites", "symmetry"],
        }
        if require_exact_system:
            kwargs["chemsys"] = "-".join(sorted(elements))
        else:
            kwargs["elements"] = elements

        docs = mpr.materials.summary.search(**kwargs)

    out = []
    for d in docs[:max_results]:
        out.append({
            "material_id": str(d.material_id),
            "formula": d.formula_pretty,
            "elements": [str(e) for e in d.elements],
            "formation_energy_per_atom": d.formation_energy_per_atom,
            "energy_above_hull": d.energy_above_hull,
            "band_gap": d.band_gap,
            "n_sites": d.nsites,
            "spacegroup": getattr(d.symmetry, "symbol", None),
        })
    return out


@mcp.tool()
def find_biomedical_candidates(family: str, max_results: int = 20,
                               stable_only: bool = True) -> dict[str, Any]:
    """Search a predefined biomedically relevant chemistry family.

    Args:
        family: one of calcium_phosphate, biodegradable_mg, titanium_implant,
            zirconia_dental, tantalum_niobium
        stable_only: restrict to energy_above_hull <= 0.05 eV/atom

    This is a convenience filter, not a biocompatibility assessment. Thermo-
    dynamic stability in vacuum at 0 K says nothing about behaviour in
    physiological solution — it is a first screen for what to look at, and
    should never be reported as a biological claim.
    """
    if family not in BIOMEDICAL_ELEMENT_SETS:
        return {"error": f"unknown family '{family}'",
                "available": sorted(BIOMEDICAL_ELEMENT_SETS)}

    elements = BIOMEDICAL_ELEMENT_SETS[family]
    results = search_materials(elements, max_results=max_results * 3)

    if stable_only:
        results = [r for r in results
                   if r["energy_above_hull"] is not None
                   and r["energy_above_hull"] <= 0.05]

    return {
        "family": family,
        "elements_searched": elements,
        "n_returned": len(results[:max_results]),
        "caveat": "0 K vacuum stability only — not a biocompatibility claim.",
        "results": results[:max_results],
    }


@mcp.tool()
def get_structure_summary(material_id: str) -> dict[str, Any]:
    """Fetch one structure and describe its lattice, sites, and symmetry."""
    with _client() as mpr:
        structure = mpr.get_structure_by_material_id(material_id)

    lattice = structure.lattice
    return {
        "material_id": material_id,
        "formula": structure.composition.reduced_formula,
        "n_sites": len(structure),
        "lattice_abc": [round(x, 4) for x in lattice.abc],
        "lattice_angles": [round(x, 2) for x in lattice.angles],
        "volume": round(lattice.volume, 3),
        "density": round(float(structure.density), 3),
        "elements": sorted({str(s.specie) for s in structure}),
    }


# ======================================================================
# Local pipeline — verified by test_mcp_server.py
# ======================================================================


@mcp.tool()
def graph_stats(material_id: str, cutoff: float = CUTOFF) -> dict[str, Any]:
    """Build the crystal graph for a material and report its statistics.

    Uses this project's own `structure_to_graph`, so the numbers are exactly
    what the model would see. Expect 25–60 edges per atom at a 5.2 A cutoff;
    single-digit values mean periodic boundary handling is broken somewhere.
    """
    with _client() as mpr:
        structure = mpr.get_structure_by_material_id(material_id)

    graph = structure_to_graph(structure, target=0.0,
                               material_id=material_id, cutoff=cutoff)
    if graph is None:
        return {"material_id": material_id, "error":
                "no edges at this cutoff — isolated atoms in a large cell"}

    n_edges = int(graph.edge_index.shape[1])
    self_image = int((graph.edge_index[0] == graph.edge_index[1]).sum())

    return {
        "material_id": material_id,
        "cutoff": cutoff,
        "n_atoms": graph.n_atoms,
        "n_edges": n_edges,
        "edges_per_atom": round(n_edges / graph.n_atoms, 2),
        "self_image_edges": self_image,
        "min_distance": round(float(graph.edge_dist.min()), 4),
        "max_distance": round(float(graph.edge_dist.max()), 4),
        "sanity": ("plausible" if 15 <= n_edges / graph.n_atoms <= 90
                   else "SUSPICIOUS — check periodic boundary handling"),
    }


@mcp.tool()
def compare_cutoffs(material_id: str,
                    cutoffs: list[float] = [3.0, 4.0, 5.2, 6.5]
                    ) -> dict[str, Any]:
    """Show how edge count scales with cutoff radius for one material.

    Useful for building intuition about why 5.2 A reaches the third or fourth
    coordination shell rather than the first, and for spotting the radius at
    which self-image edges start appearing.
    """
    with _client() as mpr:
        structure = mpr.get_structure_by_material_id(material_id)

    rows = []
    for c in sorted(cutoffs):
        g = structure_to_graph(structure, target=0.0, cutoff=c)
        if g is None:
            rows.append({"cutoff": c, "edges_per_atom": 0.0,
                         "self_image_edges": 0})
            continue
        rows.append({
            "cutoff": c,
            "edges_per_atom": round(g.edge_index.shape[1] / g.n_atoms, 2),
            "self_image_edges": int((g.edge_index[0] == g.edge_index[1]).sum()),
        })

    return {"material_id": material_id,
            "formula": structure.composition.reduced_formula,
            "n_atoms": len(structure), "by_cutoff": rows}


if __name__ == "__main__":
    mcp.run()
