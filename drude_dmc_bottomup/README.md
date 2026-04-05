# DMC Drude Bottom-Up

This directory is a self-contained, bottom-up Drude workflow for `DMC` under:

`/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/drude_dmc_bottomup`

The design goal is deliberately narrow:

- fit only `espol` (`fixed charge + Drude polarization`)
- use QM/SAPT dimer data as the primary target
- keep `bulk` as posterior validation only
- avoid using bulk density as a fitting loss

## Inputs

- `inputs/structures/DMC.pdb`
  DMC monomer structure.
- `inputs/structures/dimer_001_DMC_DMC.pdb`
  DMC dimer reference geometry.
- `inputs/structures/dmc_100mol_box.pdb`
  DMC bulk box for posterior validation only.
- `inputs/params_results/DMC.json`
  Existing charge/alpha initial guess source.
- `inputs/params_results/DMC.itp`
  Existing bonded template, reused only for posterior bulk validation.

## Workflow

1. Extract bottom-up dimer targets and generate an initial Drude model:

```bash
cd /home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/drude_dmc_bottomup
python scripts/extract_dmc_targets.py
```

This creates:

- `inputs/targets/dmc_dimer_batch000_targets.npz`
- `inputs/targets/dmc_dimer_batch000_targets.csv`
- `inputs/dmc_drude_initial.json`
- `inputs/targets/monomer_targets.template.json`

The default dimer target used by this workflow is:

- `lr_es + lr_pol`

because the first version only models long-range `espol`.

2. Compute monomer response for a Drude parameter file:

```bash
python scripts/run_dmc_drude_monomer_response.py
```

This reports:

- zero-field dipole
- finite-field polarizability tensor

3. Evaluate the DMC-DMC dimer scan against bottom-up targets:

```bash
python scripts/run_dmc_drude_dimer_scan.py --target-mode lr_espol
```

4. Fit a minimal set of Drude scaling parameters:

```bash
python scripts/fit_dmc_drude_espol.py --target-mode lr_espol
```

The first fitting pass only adjusts:

- `alpha_scale`
- `drude_charge_scale`
- `thole_scale`

It keeps the fixed charges from `DMC.json`.

5. Posterior-only bulk validation:

```bash
python scripts/validate_dmc_bulk_drude.py --param-file inputs/dmc_drude_initial.json
```

This is a posterior stress test only. It is not part of the fitting loss.
The current validator only adds:

- bonded terms from `DMC.itp`
- fixed charges
- Drude polarization

It intentionally does not add LJ/exchange/dispersion. Expect it to behave as a
numerical stress test, not as a production liquid model.

6. Exploratory hybrid bulk MD:

```bash
python scripts/run_dmc_drude_hybrid_bulk.py --param-file output/fit_smoke_joint/fit_smoke_joint_lr_espol_model.json
```

This path combines:

- Drude `espol`
- plugin Slater short-range
- plugin dispersion
- plugin intra bonded terms

It requires access to `openmm-phyneo-plugin`. The script looks for either:

- a sibling checkout named `openmm-phyneo-plugin`
- a sibling checkout named `openmm-phyneo-plugin-amoeba`
- or an explicit `OPENMM_PHYNEO_PLUGIN_ROOT`

## Model Notes

- The initial Drude guess uses:
  - charges from `DMC.json`
  - polarizabilities from `DMC.json`
  - `q_D^2 = 3000 * alpha`
  - `thole = 1.3`
- The monomer and dimer scripts build only `espol`:
  - fixed charges
  - Drude particles
  - Drude screened pairs for `1-2/1-3`
- No LJ, exchange, or dispersion are included in the dimer fitting path.

## Bottom-Up Scope

This directory is intentionally bottom-up only:

- dimer SAPT data drives the main target
- monomer QM targets can be inserted into `inputs/targets/monomer_targets.template.json`
- bulk runs are for posterior validation, not parameter optimization
