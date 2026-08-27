# crystal-graphs-mcp

An MCP server that exposes a crystal-graph materials pipeline to an agent —
including a biomedical-materials screening filter that a generic Materials
Project wrapper doesn't have.

Built as the Week 2 *"data pipeline, published as a tool"* deliverable of a
JEPA materials-discovery project. The point isn't that querying the Materials
Project through an agent is novel — it's that the local tools compute answers
with the **same** `structure_to_graph` a CGCNN model trains on. Ask an agent
"what's the edge count per atom for mp-149 at a 6.0 Å cutoff" and it returns
the real number from the pipeline, not a guess from memory.

## Tools

| Tool | What it does | Status |
|------|--------------|--------|
| `search_materials` | Search the Materials Project by element set; returns formation energy, energy above hull, band gap, and site count | MP-backed — not yet run against the live API |
| `find_biomedical_candidates` | Screen a predefined biomedical chemistry family (Ca–P, Mg, Ti, zirconia, Ta/Nb) with an optional 0 K stability filter | MP-backed — not yet run against the live API |
| `get_structure_summary` | Fetch one structure and describe its lattice, density, and symmetry | MP-backed — not yet run against the live API |
| `graph_stats` | Build the crystal graph and report edges/atom, self-image edges, and distance range | Graph logic tested offline; the MP fetch is not |
| `compare_cutoffs` | Show how edge count scales with cutoff radius | Graph logic tested offline; the MP fetch is not |

`find_biomedical_candidates` is a convenience screen over composition and
0 K vacuum stability — **not** a biocompatibility assessment. Thermodynamic
stability in vacuum says nothing about behaviour in physiological solution; it
is a first filter for what to look at, never a biological claim.

## Install

```bash
python3 -m venv .venv            # Python 3.11+ recommended
source .venv/bin/activate
pip install -r requirements.txt
```

## Materials Project key

The three MP-backed tools need a free
[Materials Project API key](https://next-gen.materialsproject.org/api).
Keep it in the environment — never in a tool call and never in git:

```bash
export MP_API_KEY="your-key-here"
```

## Run

```bash
python mcp_server.py             # stdio transport
```

## Register with an MCP client

A Claude Desktop / Hermes `mcpServers` entry (adjust the path to wherever you
cloned this):

```json
{
  "mcpServers": {
    "crystal-graphs": {
      "command": "python",
      "args": ["/Users/aaronlin/Documents/ML/blog/crystal-graphs-mcp/mcp_server.py"],
      "env": { "MP_API_KEY": "your-key-here" }
    }
  }
}
```

## Tests

The local half — graph construction and the reference numbers — is covered
end to end and needs no key and no network:

```bash
python test_mcp_server.py
```

It checks coordination-shell counts, self-image edges, edges per atom at the
project's 5.2 Å cutoff, and the fcc-Cu 42 → 54 discontinuity across the
a√2 ≈ 5.11 Å shell — the reference-table bug a mechanical test caught.

## What's verified

- **Local pipeline:** the periodic neighbour-finding and graph construction the
  two local tools rely on. `test_mcp_server.py` passes all 17 checks — including
  the fcc-Cu 42 -> 54 edge discontinuity across the a√2 ≈ 5.11 Å shell —
  against pymatgen 2026.5.4 / torch 2.13.0, and originally against
  pymatgen 2025.10.7 / torch 2.5.1. Stable across those library versions.
- **Materials Project tools:** exercised against the live API through an MCP
  client. `mp-149` returns silicon with the correct rhombohedral primitive cell
  (a = 3.849 Å, α = 60°); `graph_stats` on it reports 28 edges/atom,
  matching the offline diamond-Si reference even though the live cell is the
  2-atom primitive rather than the test's 8-atom conventional cell; a Mg-O
  search returns the MgO polymorphs with rocksalt (mp-1265) as the stable
  phase; and the `calcium_phosphate` family returns on-hull candidates. No
  field-name or filter drift observed.

### Good to know

- `find_biomedical_candidates` uses a *contains-these-elements* search, not an
  exact-system one, so a family can surface phases carrying extra elements —
  e.g. `calcium_phosphate` may return NaCa(PO₃)₃ (Ca, P, O **plus** Na).
  That is by design: it is a first screen, not an exact-composition filter.

## License

MIT — see [LICENSE](LICENSE).
