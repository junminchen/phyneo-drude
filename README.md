# phyneo-drude

This repository collects the Drude-focused work extracted from the larger
`test_MPID_DMFF` workspace.

Contents:

- `drude_openmm_baselines`
  Small self-contained OpenMM Drude baselines using
  `charmm_polar_2023.xml`. This currently includes:
  - water dimer sanity check
  - bulk water NPT baseline

- `drude_dmc_bottomup`
  Bottom-up Drude workflow for `DMC`:
  - DMC monomer response
  - DMC-DMC dimer `lr_es + lr_pol` targets
  - simple Drude parameter fitting
  - posterior-only bulk validation
  - exploratory hybrid bulk script combining Drude espol with the existing
    PhyNEO short-range, dispersion, and bonded terms

- `mlff_dmc_bottomup`
  Machine-learned bottom-up workflow for `DMC`:
  - monomer graph encoder with 2D attention-style pair bias
  - continuous nonbonded parameter heads
  - SAPT decomposed dimer supervision
  - posterior plots and exported parameter JSON

- `PATENT_DESIGN_AROUND_NOTES.md`
  Engineering notes for a future lower-risk design-around implementation.

Notes:

- The DMC hybrid bulk script
  [run_dmc_drude_hybrid_bulk.py](./drude_dmc_bottomup/scripts/run_dmc_drude_hybrid_bulk.py)
  depends on helper modules from `openmm-phyneo-plugin`.
- By default it looks for a sibling checkout named `openmm-phyneo-plugin` or
  `openmm-phyneo-plugin-amoeba`.
- You can also point it explicitly with:

```bash
export OPENMM_PHYNEO_PLUGIN_ROOT=/path/to/openmm-phyneo-plugin
```

- Large trajectories and checkpoint files are intentionally not tracked here.
