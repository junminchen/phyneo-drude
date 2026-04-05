# DMC MLFF Bottom-Up

This directory contains a first-pass machine-learned, bottom-up workflow for
`DMC` full nonbonded parameter prediction under:

`/home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/phyneo-drude/mlff_dmc_bottomup`

The design goal is:

- use a `polff`-style 2D attention encoder on the **monomer** graph
- predict **continuous transferable nonbonded parameters**
- supervise on **SAPT decomposed dimer targets**
- keep `bulk` out of the training loss

This is an MVP, not a production force field. There are now two training paths:

- the original `surrogate` path, which uses a lightweight differentiable
  approximation for all nonbonded terms
- a `real Slater` path, which keeps the Drude long-range surrogate but replaces
  the short-range and dispersion terms with the actual PhyNEO
  `QqTt/Slater/Tang-Toennies` functional form

## Inputs

- `inputs/structures/DMC.pdb`
  Local DMC monomer structure.
- `inputs/structures/dimer_001_DMC_DMC.pdb`
  Local DMC dimer reference structure.
- `inputs/params_results/DMC.json`
  Base nonbonded parameter source used for initialization and regularization.
- `inputs/targets/monomer_targets.json`
  QM monomer anchors:
  - dipole
  - isotropic polarizability
- `inputs/targets/dmc_dimer_conf001_600_full_targets.npz`
  Full SAPT decomposition bundle for `conf_001_DMC_DMC`, extracted across all
  50 batches (`600` total dimer points).

## Workflow

1. Extract the full DMC-DMC SAPT decomposition bundle:

```bash
cd /home/am3-peichenzhong-group/Documents/project/test_MPID_DMFF/phyneo-drude
python mlff_dmc_bottomup/scripts/extract_full_dmc_targets.py
```

This writes:

- `inputs/targets/dmc_dimer_conf001_600_full_targets.npz`
- `inputs/targets/dmc_dimer_conf001_600_full_targets.csv`

2. Train the monomer-parameter MLFF in `espol` warm-start mode:

```bash
python mlff_dmc_bottomup/scripts/train_dmc_mlff.py --stage espol --steps 1000
```

3. Train the full nonbonded surrogate:

```bash
python mlff_dmc_bottomup/scripts/train_dmc_mlff.py --stage full_nonbonded --steps 3000 --batch-size 128
```

4. Train against the real PhyNEO Slater/QqTt/dispersion force terms on the
12-point batch:

```bash
python mlff_dmc_bottomup/scripts/train_dmc_real_slater.py \
  --batch-index 0 \
  --steps 1200 \
  --output-dir mlff_dmc_bottomup/output/train_real_slater_12pt \
  --prefix dmc_mlff_real_slater_12pt
```

5. Plot the calibrated 12-point real-Slater scan:

```bash
python mlff_dmc_bottomup/scripts/plot_real_slater_12pt.py \
  --model-json mlff_dmc_bottomup/output/train_real_slater_12pt/dmc_mlff_real_slater_12pt_model.json \
  --batch-index 0
```

## Model Notes

- The network reads only the monomer graph.
- It predicts continuous atom-wise heads for:
  - `charge`
  - `alpha`
  - `thole`
  - `c6`
  - short-range repulsion amplitude/range
  - `ct-like` amplitude/range
  - auxiliary short-range `es/pol` scales
- Dimer energies are computed by a differentiable surrogate evaluator.
- In the `real Slater` route the physical grouping is fixed to the plugin terms:
  - `sr_es_total = QqTtDampingForce + SlaterSrEsForce`
  - `sr_pol_total = SlaterSrPolForce`
  - `exchange = SlaterExForce`
  - `dispersion_total = ADMPDispPmeForce + SlaterDampingForce + SlaterSrDispForce`
  - `ct_like = SlaterDhfForce`
- The evaluator currently exposes these supervised terms:
  - `lr_es`
  - `lr_pol`
  - `sr_es_total`
  - `sr_pol_total`
  - `exchange`
  - `dispersion_total`
  - `ct_like`
- By default the trainer uses the full `600`-point `conf_001_DMC_DMC` bundle,
  not a single 12-point batch.

## Outputs

Each training run writes:

- `*_model.json`
  Final predicted atom-wise parameters.
- `*_summary.json`
  Final loss, per-term normalized RMSE, and monomer prediction summary.
- `*_weights.npz`
  Raw JAX model weights.
- `*_curves.csv`
  Target/predicted dimer curves for each supervised term.
- `*_curves.png`
  Matplotlib comparison plot for the supervised terms.

## Current Limits

- The current evaluator is a differentiable surrogate, not the plugin/OpenMM
  production evaluator.
- The monomer anchor uses dipole and isotropic polarizability, not the full
  tensor.
- This directory only covers `DMC`.
- `bulk` is still posterior-only and is intentionally not part of the loss.
