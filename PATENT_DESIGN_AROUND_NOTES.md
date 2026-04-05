# Patent Design-Around Notes

This note is an engineering document, not legal advice.

It records:

- what the current implementation is doing
- why that implementation may resemble published "molecular force field
  parameter determination" patent language
- how to pivot later to a lower-risk design without changing the scientific
  goal

## Current Implementation

The current `mlff_dmc_bottomup` line intentionally does **not** try to avoid the
published patent language up front.

The implementation currently uses:

- a monomer graph encoder
- a small 2D attention-style message-passing block with pair bias from graph
  distance
- continuous parameter heads
- a physics evaluator that turns predicted parameters into decomposed dimer
  energies

This is closer to:

- continuous parameter regression

than to:

- a hierarchical lookup-table parameter assignment engine

That distinction matters for future design-around work.

## Higher-Risk Technical Language To Avoid Later

If this project later needs a lower-risk implementation or paper wording,
avoid describing it as:

- "hierarchical encoding followed by parameter lookup"
- "matching encoded molecular features to a parameter table"
- "progressively searching levels until a parameter type is determined"
- "using a matching table as the source of final force-field parameters"

Those phrases are closer to the semantics of the published application than the
current continuous-regression implementation.

## Lower-Risk Design-Around Direction

If we need to pivot later, keep the scientific workflow but tighten the
implementation around these constraints:

- the network outputs **continuous parameters directly**
- no discrete template ID is predicted
- no parameter table is consulted to select final values
- no staged "fallback to another hierarchy level" is used
- the physics layer consumes predicted parameters directly
- training is described as response-matching / SAPT-term regression, not
  template assignment

The recommended future wording is:

- `monomer graph encoder`
- `continuous parameter heads`
- `physics-based decomposed-energy supervision`

and not:

- `hierarchical feature matching`
- `table-based parameter determination`

## Clean Separation For Future Refactor

If a design-around refactor is needed later, keep the current directory split:

- `drude_dmc_bottomup`
  hand-fit / explicit-physics bottom-up reference
- `mlff_dmc_bottomup`
  learned parameter predictor

Then make the design-around changes only inside the ML directory:

- swap the encoder if needed
- keep the parameter head interface stable
- keep the evaluator interface stable
- keep the output JSON/CSV/PNG formats stable

That minimizes scientific churn while changing the implementation narrative.

## Publication Guidance

For technical writing, the safer framing is:

- a graph neural network predicts transferable nonbonded parameters
- the parameters are trained against QM/SAPT decomposed targets
- the evaluator is physics-based

Avoid describing the method as:

- classifying atoms into parameter templates
- selecting parameters from a lookup table
- determining parameters by hierarchical table matching
