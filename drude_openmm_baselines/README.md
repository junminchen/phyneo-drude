# Drude OpenMM Baselines

This directory is a self-contained OpenMM Drude reference area under
`test_MPID_DMFF`.  It uses a local copy of
`inputs/charmm_polar_2023.xml` so the examples do not depend on the
OpenMM `site-packages` data path at runtime.

Current contents:

- `inputs/charmm_polar_2023.xml`
  OpenMM's CHARMM polar 2023 force field copied from the active conda
  environment.
- `inputs/water_dimer.pdb`
  Two-water SWM4-style starting geometry for a small Drude sanity check.
- `scripts/run_water_dimer.py`
  Short Drude dimer energy/minimization check.
- `scripts/run_bulk_water_npt.py`
  Small bulk-water Drude NPT baseline using `SWM4-NDP`.

## Recommended runs

Water dimer sanity check:

```bash
cd /home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/drude_openmm_baselines
python scripts/run_water_dimer.py
```

Short bulk-water Drude NPT:

```bash
cd /home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/drude_openmm_baselines
python scripts/run_bulk_water_npt.py --nvt-steps 1000 --npt-steps 2000
```

## Notes

- The bulk-water script builds the box from scratch with
  `Modeller.addSolvent(..., model='swm4ndp')`.
- The integrator path is intentionally conservative:
  `DrudeNoseHooverIntegrator` with a small timestep and a Drude
  displacement cap.
- Outputs go under `output/`.
