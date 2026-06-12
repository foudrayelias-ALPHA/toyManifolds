# ATLAS-1

**A dictionary of charts, not directions** — mapping transformer features as manifolds instead of SAE latents.

Part of [toyManifolds](https://github.com/foudrayelias-ALPHA/toyManifolds). Companion to LOOP-1 (shared `topology.py` telemetry planned).

> Replace the SAE's dictionary of **directions** with a dictionary of **feature families**: an over-provisioned group-sparse autoencoder whose surviving groups are read out as manifolds (topology, intrinsic dimension, and a human-usable coordinate), validated against planted ground truth and grokked modular-addition circuits.

Full specification: [`ATLAS-1_project_description.md`](ATLAS-1_project_description.md)

## Status

**v0.1 scaffold** — repository structure and module stubs. Implementation follows the phased plan in §7 of the spec.

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Telemetry sanity (persistence + coordinates) | Not started |
| 1 | Stage 0 planted recovery | Scaffold |
| 2 | Stage 1 mod-add + steering | Scaffold |
| 3 | Knob demo | Placeholder |
| 4 | Stage 2 ViT stretch | Placeholder |
| 5 | Writeup | — |

## Layout

```
Atlas-1/
├── atlas1/              # Core library
│   ├── featurizer.py    # Group-sparse autoencoder (§3.1)
│   ├── topology.py      # Manifold readout (§3.2)
│   ├── metrics.py       # M1–M7 (§6)
│   └── baselines.py     # B1–B3, controls, ablations (§5)
├── testbeds/
│   ├── stage0_planted.py
│   ├── stage1_modadd.py
│   └── stage2_vit.py
├── configs/             # YAML experiment configs
├── scripts/             # CLI entry points
├── demo/                # Interactive knob demo (Phase 3)
├── tests/
└── runs/                # Saved experiment outputs (gitignored)
```

## Quick start

```bash
cd Atlas-1
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,topology]"
pytest
python scripts/run_stage0.py --dry-run
```

## Hypotheses

| ID | One-liner |
|----|-----------|
| H1 | Planted recovery on Stage 0 |
| H2 | Loop recovery on grokked mod-add |
| H3 | Beat SAE dilution at matched budget |
| H4 | End-to-end beats SAE-then-stitch |
| H5 | Families stabler than atoms across seeds |
| H6 | Chart steering shifts predictions |

## License

MIT (inherits from parent `toyManifolds` repo unless noted otherwise).
