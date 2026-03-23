# Paper Reproduction Status

This file summarizes what still goes beyond the paper text even after the repository was cleaned back to the paper-style single-environment setup.

## What now follows the paper directly

- Chemical default config now matches the original repository setup:
  - `num_env = 10`
  - `total_steps = 16100`
  - `init_steps = 1000`
  - `random_action_steps = 1000`
- Magnetic default config uses:
  - `num_env = 1`
  - `total_steps = 200000`
  - `init_steps = 2000`
  - `random_action_steps = 2000`
- Both default configs use:
  - `K = 16`
  - `lambda = 0.001`
  - `beta = 0.25`
  - `total_test_episode_num = 10`
- The launcher scripts still run 8 seeds, matching the paper's averaging protocol.

## Remaining implementation details not literally specified by the paper

### Chemical default config no longer matches the paper literally

The default Chemical config has been reverted to the original repository behavior:

- `num_env = 10`
- `total_steps = 16100`
- `code_labeling = false`

So the default Chemical config is now repository-faithful rather than paper-literal.

### Magnetic environment reconstruction

The Magnetic environment is not author-released code. The current implementation is a Robosuite-backed reconstruction based on the paper text.

That means the following are implementation choices, not values explicitly given by the paper:

- exact Robosuite task wiring
- object sizes
- object densities and friction coefficients
- the exact magnetic-force law
- the scalar `magnetic_force = 0.5`

The paper specifies the task structure, reward, success condition, table size, and OOD test protocol, but not every simulator constant.

### EMA VQ updates

The configs use:

- `vqvae_ema = true`
- `ema = 0.99`

These are implementation choices. The paper tables do not explicitly specify EMA versus non-EMA VQ updates.

### Evaluation scheduling

The paper states:

- 10 test episodes
- every 40 training episodes

In code, this is implemented through:

- `total_test_episode_num = 10`
- `test_freq = 1000`

With `max_steps = 25` and `num_env = 1`, that is consistent with evaluation every 40 episodes, but it is still encoded as a step-frequency parameter in the trainer.

### Convenience code kept in the repo

The following are still present because they are useful and do not change the paper experiment itself:

- `--config=...` support in `main_policy.py`
- separate Markdown documentation files
- Robosuite logger suppression to reduce repeated console spam

## Still not fully reproducible from this repo alone

The repository still does not contain all paper baselines or all figure-generation scripts.

In particular, this codebase currently contains:

- `Ours`
- `MLP`
- `GNN`
- `NCD`
- `Oracle`

But the paper discusses additional baselines beyond what is present here. So although the FCDL / Chemical / Magnetic runs are now aligned much more closely with the paper, the repository is still not sufficient to regenerate every main-text figure exactly as published without adding the missing baselines and figure scripts.
