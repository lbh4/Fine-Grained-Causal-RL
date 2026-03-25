# WandB Graph Reference

## Scope

This file explains the WandB graphs produced by the current training code in [`main_policy.py`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py).

It covers:

- which metrics are logged
- how each metric is computed
- which metrics are Chemical-only or Magnetic-only
- why the two experiments show different graph types

## High-Level Structure

The code logs four kinds of information:

1. run-progress signals
2. online training signals
3. offline dynamics-evaluation signals
4. downstream test-policy signals

The main logging sites are:

- [`main_policy.py`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py)
- [`fcdl/model/inference.py`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/model/inference.py)
- [`fcdl/model/inference_fcdl_base.py`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/model/inference_fcdl_base.py)

## Graphs Logged In Both Chemical And Magnetic

### `init_stage`

Source:

- [`main_policy.py:349`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L349)
- [`main_policy.py:351`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L351)

Meaning:

- `1.0` during warmup, when the code is still in the initial random-data-collection phase
- `0.0` after warmup ends

Why it matters:

- This graph marks the point where the run switches from random action collection to learned-model updates and model-based control.

### `n_samples`

Source:

- [`main_policy.py:407`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L407)
- [`main_policy.py:408`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L408)

Meaning:

- cumulative number of environment transitions collected so far

How it is computed:

- each loop adds `num_env`
- Chemical default: `num_env = 10`
- Magnetic default: `num_env = 1`

Why it matters:

- this is the closest logged quantity to actual sample count
- it is often more comparable than raw loop step when `num_env` differs

### `policy_stat/episode_reward`

Source:

- vectorized case: [`main_policy.py:365`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L365)
- single-env case: [`main_policy.py:383`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L383)

Meaning:

- episodic return from the live training environment

How it is computed:

- rewards are accumulated during an episode
- when an episode ends, the total return is logged
- in vectorized Chemical training, the code logs the mean over the currently tracked parallel environments

Why it matters:

- this is the online training-return curve, not the held-out test curve

### `policy_stat/success`

Source:

- vectorized case: [`main_policy.py:368`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L368)
- single-env case: [`main_policy.py:385`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L385)

Meaning:

- whether the current training episode succeeded

How it is computed:

- each step exposes `info["success"]` from the environment
- the code ORs this across the episode
- when the episode ends:
  - Chemical vectorized training logs the mean success across active env slots
  - Magnetic single-env training logs `0.0` or `1.0`

Why it matters:

- this is the online training success signal
- it is noisier than the separate held-out test success graphs

### `episode_num`

Source:

- vectorized case: [`main_policy.py:367`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L367)
- single-env case: [`main_policy.py:386`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L386)

Meaning:

- running count of completed training episodes

Why it matters:

- useful for interpreting `test_freq = 1000` as “roughly every 40 episodes” when `max_steps = 25` and `num_env = 1`
- less direct in vectorized Chemical because multiple envs finish asynchronously

### `inference/pred_loss`

Source:

- generic inference update: [`fcdl/model/inference.py:230`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/model/inference.py#L230)
- `fcdl` update path: [`fcdl/model/inference_fcdl_base.py:276`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/model/inference_fcdl_base.py#L276)
- logging: [`main_policy.py:448`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L448)

Meaning:

- training loss for one-step or multi-step next-state prediction

How it is computed:

- the model predicts a distribution over the next latent / factorized state
- the loss is negative log-likelihood or KL-style prediction loss, averaged across variables and batch

Why it matters:

- this is the main learned-dynamics training objective

### `inference/grad_norm`

Source:

- [`fcdl/model/inference.py:219`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/model/inference.py#L219)

Meaning:

- gradient norm of the inference model before optimizer step, after clipping logic is applied

Why it matters:

- useful for spotting optimization instability or exploding gradients

### `inference_eval/pred_loss`

Source:

- [`fcdl/model/inference.py:325`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/model/inference.py#L325)
- logging cadence: [`main_policy.py:434`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L434)

Meaning:

- held-out prediction loss on replay-buffer evaluation samples

How it is computed:

- same basic prediction objective as `inference/pred_loss`
- but run in eval mode on `use_part="eval"` samples rather than training samples

Why it matters:

- this is the closest thing to a validation-loss curve for the dynamics model

## Graphs Usually Seen With `fcdl`

These appear when the current inference method is the FCDL masking / codebook model.

### `inference/full_pred_loss`

Source:

- [`fcdl/model/inference_fcdl_base.py:278`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/model/inference_fcdl_base.py#L278)

Meaning:

- placeholder full-model prediction loss

Current behavior:

- always zero in the current code

Interpretation:

- this is not a meaningful learning curve right now

### `inference/reg_loss`

Source:

- [`fcdl/model/inference_fcdl_base.py:286`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/model/inference_fcdl_base.py#L286)

Meaning:

- local-causal-model regularization term

Interpretation:

- reflects the structural regularizer controlled by `reg_coef`

### `inference/vq_loss`

Source:

- [`fcdl/model/inference_fcdl_base.py:287`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/model/inference_fcdl_base.py#L287)

Meaning:

- vector-quantization loss from the codebook module

Interpretation:

- measures how the learned subgroup assignment machinery is fitting the latent codes

### `inference/commit_loss`

Source:

- [`fcdl/model/inference_fcdl_base.py:288`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/model/inference_fcdl_base.py#L288)

Meaning:

- VQ commitment loss

Interpretation:

- encourages encoder outputs to stay close to selected codebook entries

## Chemical-Only Graphs

### `inference_eval/accuracy`

Source:

- [`fcdl/model/inference.py:328`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/model/inference.py#L328)

Meaning:

- prediction accuracy for categorical state variables during evaluation

How it is computed:

- for each categorical next-state variable, take `argmax` of the predicted logits
- compare with the `argmax` of the target one-hot variable
- average over samples and variables

Why Chemical has it:

- Chemical state is dominated by discrete color variables, so categorical accuracy is natural and interpretable

Why Magnetic does not:

- Magnetic state is mixed discrete plus continuous, so a single categorical accuracy number is not a good summary

### `test/test1/inference/accuracy`, `test/test2/inference/accuracy`, `test/test3/inference/accuracy`

Source:

- [`main_policy.py:40`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L40)
- [`fcdl/model/inference.py:282`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/model/inference.py#L282)

Meaning:

- offline OOD prediction accuracy on Chemical test conditions

How it is computed:

- the code samples OOD transitions from replay-buffer states whose root color is fixed to the test regime
- then it injects the Chemical test-noise setting through the encoder
- then it computes categorical next-state prediction accuracy on the matched variables

Chemical test environments:

- `test1`: lower OOD severity
- `test2`: medium OOD severity
- `test3`: higher OOD severity

These correspond to different predefined noisy-node settings in the Chemical environment and encoder logic.

Relevant code:

- OOD replay selection: [`fcdl/utils/replay_buffer.py:142`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/utils/replay_buffer.py#L142)
- test env list: [`policy_params_chemical.json:78`](/home/byeonghui/Fine-Grained-Causal-RL/policy_params_chemical.json#L78)

### `test/test1/policy/episode_reward_mean`, `test/test2/policy/episode_reward_mean`, `test/test3/policy/episode_reward_mean`

Source:

- [`main_policy.py:153`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L153)

Meaning:

- average return over held-out test episodes for each Chemical OOD regime

How it is computed:

- run `total_test_episode_num` episodes with deterministic policy actions
- collect total episode reward
- average over the finished test episodes

### `test/test1/policy/success_ratio`, `test/test2/policy/success_ratio`, `test/test3/policy/success_ratio`

Source:

- [`main_policy.py:220`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L220)

Meaning:

- fraction of held-out test episodes that succeeded in each Chemical OOD regime

How it is computed:

- each episode tracks whether `info["success"]` was ever true
- average those binary episode outcomes over the test episodes

## Magnetic-Only Graphs

### `test/test/inference/shd`

Source:

- [`main_policy.py:88`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L88)
- [`fcdl/model/inference.py:299`](/home/byeonghui/Fine-Grained-Causal-RL/fcdl/model/inference.py#L299)

Meaning:

- offline OOD structural Hamming distance between the inferred local causal mask and the ground-truth local causal mask

How it is computed:

- collect random-action transitions from the Magnetic OOD test env
- get the environment-provided ground-truth local causal mask `info["lcm"]`
- infer the model’s local mask with `infer_local_mask(...)`
- compute elementwise absolute difference and sum over the matrix
- average across the batch

Interpretation:

- lower is better
- `0` means the inferred local graph matches the env’s ground-truth local graph exactly for the sampled transitions

Why Magnetic has SHD instead of accuracy:

- the current Magnetic offline OOD evaluation is about graph recovery under changed context, not categorical node prediction
- this matches the way the current repo implements Magnetic OOD evaluation

### `test/test/policy/episode_reward_mean`

Source:

- [`main_policy.py:111`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L111)

Meaning:

- mean held-out return in the Magnetic OOD test environment

How it is computed:

- deterministic policy evaluation over `total_test_episode_num` episodes
- average total reward per episode

### `test/test/policy/success_ratio`

Source:

- [`main_policy.py:150`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L150)

Meaning:

- fraction of held-out Magnetic OOD test episodes that succeed

How it is computed:

- success is taken from `info["success"]`
- the code ORs success across the episode
- the final binary result is averaged across test episodes

Magnetic test environment:

- there is currently one named test condition, `test`
- it uses OOD box sampling via `test_box_sigma = 100.0`

Relevant config:

- [`policy_params_magnetic.json:75`](/home/byeonghui/Fine-Grained-Causal-RL/policy_params_magnetic.json#L75)

## Automatically Generated WandB Panels

### Gradient panels from `wandb.watch(...)`

Source:

- [`main_policy.py:307`](/home/byeonghui/Fine-Grained-Causal-RL/main_policy.py#L307)

Meaning:

- WandB automatically records gradient statistics for the inference model every `100` logging steps

Important note:

- these panels are generated by WandB itself
- their exact names and layout depend on the WandB UI
- they are not manually named in the repo like the scalar curves above

## Why Chemical And Magnetic Show Different Graph Types

There are three main reasons.

### 1. The OOD definitions are different

Chemical OOD:

- same basic environment family
- test-time node corruption / spurious-noise conditions
- natural offline summary: next-state prediction accuracy on affected discrete variables

Magnetic OOD:

- changed context for local causality
- one object is black, magnetic interaction turns off, and box position is sampled from a broad OOD distribution
- natural offline summary: whether the inferred local causal graph matches the true local graph

### 2. The state representations are different

Chemical:

- mostly discrete factorized variables
- accuracy on one-hot categorical predictions is meaningful

Magnetic:

- mixed discrete and continuous state
- graph-structure quality is more informative than a single mixed-state “accuracy” number

### 3. The test suites are different

Chemical logs three named OOD test families:

- `test1`
- `test2`
- `test3`

Magnetic currently logs one named OOD test family:

- `test`

So the Chemical WandB page naturally has more per-test panels.

## Practical Reading Guide

If you are reading WandB for model quality:

- look at `inference_eval/pred_loss` for generic validation-style prediction quality
- on Chemical, also look at `inference_eval/accuracy`
- on Chemical OOD, look at `test/test*/inference/accuracy`
- on Magnetic OOD, look at `test/test/inference/shd`

If you are reading WandB for control performance:

- look at `test/.../policy/episode_reward_mean`
- look at `test/.../policy/success_ratio`

If you are reading WandB for optimization stability:

- look at `inference/pred_loss`
- look at `inference/grad_norm`
- for FCDL runs, also look at `inference/reg_loss`, `inference/vq_loss`, and `inference/commit_loss`

## Important Caveat

The graph names described here are for the current code in this workspace.

They are not guaranteed to match:

- the original upstream CDL repo
- older local revisions of this repo
- every possible inference algorithm if additional methods are added later
