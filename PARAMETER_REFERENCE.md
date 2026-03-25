# Parameter Reference

This file documents the parameters used by:

- `policy_params_chemical.json`
- `policy_params_magnetic.json`

Primary paper source:

- Appendix C, Table 4, Table 5, and Table 6 of FCDL (`arXiv:2406.03234`)

## Important note on Chemical training-step counts

The default Chemical config now follows the original repository-style setup:

- `num_env = 10`
- `total_steps = 16100`
- `init_steps = 1000`
- `random_action_steps = 1000`

This is not the paper's literal single-environment schedule. It is the repository default.

## Top-level fields

### `info`

Short run name prefix used for logging and result directories.

### `obs_keys`

Observation keys consumed by the encoder.

- Chemical: explicit object keys are listed in the JSON.
- Magnetic: `main_policy.py` overwrites this to `["ball", "box", "eef"]`.

### `goal_keys`

Goal observation keys used by planning / reward code.

- Chemical: target object colors.
- Magnetic: empty, because the reward is computed directly from the current state.

### `seed`

Global random seed.

- `-1` means "sample a fresh seed".

### `cuda_id`

CUDA device index passed to the training code.

### `wandb_dir`

Directory used by WandB for local run files.

### `loglevel`

Python logging level.

## `fcdl_params`

These configure FCDL itself.

### `feature_fc_dims`

Shared hidden layers before masked prediction.

- Chemical: `[128, 128]`
- Magnetic: `[128, 128]`
- Paper source: Table 6 hidden dim `128`

### `generative_fc_dims`

Hidden layers in the masked per-variable predictor.

- Chemical: `[128, 128]`
- Magnetic: `[128, 128, 128]`

In this codebase, the total FCDL hidden-layer depth is split across `feature_fc_dims` and `generative_fc_dims`.

- Chemical total depth: `4` hidden layers, matching Table 6.
- Magnetic total depth: `5` hidden layers, matching Table 6.

### `vq_encode_fc_dims`

VQ encoder widths.

- Both configs: `[128, 64]`
- Paper source: Table 6

### `vq_decode_fc_dims`

VQ decoder widths.

- Both configs: `[32]`
- Paper source: Table 6

### `ncd_fc_dims`

Auxiliary-network widths for the NCD baseline implementation.

- Chemical: `[128, 128]`
- Magnetic: `[128, 128, 128]`

These are set so the baseline depth matches the Table 6 hidden-layer counts in this implementation.

### `code_labeling`

Whether the masked dynamics model receives the subgroup label `z` as an additional one-hot input.

- Chemical config: `false`
- Magnetic config: `true`
- Paper source: Appendix C.4.1 says the authors use the one-hot encoding of `z`

### `vqvae_ema`

Whether to update the VQ codebook with EMA.

- Implementation detail
- Not explicitly specified in the paper tables

### `ema`

EMA decay used when `vqvae_ema` is enabled.

### `codebook_size`

Number of quantization codes `K`.

- Both configs: `16`
- Paper source: Table 6

### `code_dim`

Latent code dimension.

- Both configs: `16`
- Paper source: Table 6

### `reg_coef`

Regularization weight for sparsifying local causal masks.

### `vq_coef`

Weight of the vector-quantization loss.

### `commit_coef`

Weight of the VQ commitment loss.

### `local_mask_sampling_num`

Number of local-mask samples drawn during training.

### `eval_local_mask_sampling_num`

Number of local-mask samples drawn during evaluation.

## `training_params`

### `inference_algo`

Dynamics / causal inference method.

- This repo uses `"fcdl"` for FCDL runs.

### `rl_algo`

RL algorithm family.

- This repo uses `"model_based"`.

### `load_id`

Run identifier for checkpoint loading.

### `load_inference`

Optional path suffix for loading a pretrained inference checkpoint.

### `load_model_based`

Optional path suffix for loading a pretrained model-based policy checkpoint.

### `load_policy`

Legacy loading hook, currently unused here.

### `load_replay_buffer`

Optional replay-buffer load path.

### `total_steps`

Total training interaction budget.

- Chemical: `16100`
- Magnetic: `200000`
- Paper source: Table 4

### `init_steps`

Initial random-data collection budget before learning starts.

- Chemical: `1000`
- Magnetic: `2000`
- Paper source: Table 4

### `random_action_steps`

How long the policy stays random during initial collection.

- Set equal to `init_steps`
- Matches the paper's "random policy for the initial data collection" protocol in Appendix C.2

### `inference_gradient_steps`

Number of inference updates per training loop.

### `inference_update_freq`

How often inference updates run.

### `policy_update_freq`

How often policy updates run.

### `test_freq`

How often downstream evaluation runs.

- Repo scheduling parameter
- Not specified in the paper tables

### `ood_eval_freq`

How often offline OOD evaluation runs.

- Repo scheduling parameter
- Not specified in the paper tables

### `ood_eval_batch_size`

Batch size for offline OOD evaluation.

- Repo evaluation parameter

### `total_test_episode_num`

Number of episodes used per evaluation call.

### `saving_freq`

Model checkpoint interval.

### `plot_freq`

Planner logging / plotting interval.

### `replay_buffer_params.capacity`

Replay buffer capacity.

### `replay_buffer_params.max_sample_time`

Maximum number of times one stored transition can be resampled.

- Chemical: `128`
- Magnetic: `1024`

The Magnetic value is a repo-side stability adjustment for long runs. It is not a paper hyperparameter.

### `replay_buffer_params.saving_freq`

Replay-buffer snapshot interval.

## `env_params`

### `env_name`

Environment selector.

- Chemical config: `"Chemical"`
- Magnetic config: `"Magnetic"`

### `num_env`

Number of parallel environments.

- Chemical: `10`
- Magnetic: `1`

This is a codebase execution choice, not a value specified in the paper.

## `env_params.chemical_env_params`

### `gt_local_mask`

Whether the environment exposes its known mask structure.

### `local_causal_rule`

Chemical graph family.

- Main-paper config here uses `"full_fork"`

### `use_position`

Whether positions are included in the state.

- `false` for the paper's color-based Chemical setup

### `num_objects`

Number of nodes / state variables.

- `10`
- Paper source: Appendix C.1.1

### `num_colors`

Number of colors.

- `5`
- Together with 10 objects this yields the 50-way action space described in Appendix C.1.1

### `continuous_pos`

Internal Chemical-environment flag for position handling.

### `width_std`, `height_std`, `width`, `height`

Chemical environment layout settings used by the repo implementation.

### `render_image`, `render_type`, `shape_size`

Rendering settings.

### `movement`

Chemical transition mode.

### `use_cuda`

Whether the Chemical env internals use CUDA.

### `max_steps`

Episode horizon.

- `25`
- Paper source: Table 4

### `num_target_interventions`

Number of target interventions sampled for goal generation.

### `g`

Internal graph-specification string for the Chemical simulator.

### `match_type`

Which observable nodes contribute to the reward.

### `dense_reward`

Whether to use dense reward.

- This matches Eq. (40)-style reward accumulation in the implementation

### `num_action_variable`

Number of factorized action variables.

- Chemical uses `1` categorical action variable

### `name`

Current environment split name.

### `test_params`

Named OOD test splits used by the repo.

- `test1`
- `test2`
- `test3`

These map to the predefined noisy-node subsets inside the Chemical environment implementation.

## `env_params.magnetic_env_params`

### `name`

Current environment split name.

### `max_steps`

Episode horizon.

- `25`
- Paper source: Table 4

### `robot`

Robosuite robot arm.

- The reconstruction uses `"Panda"`

### `control_freq`

Robosuite controller frequency.

### `table_full_size`

Table dimensions.

- `[0.6, 0.9, 0.05]`
- Paper source: Appendix C.1.2 says x-length `0.6`, y-width `0.9`

### `table_friction`

Robosuite table friction coefficients.

### `table_offset`

Table origin / offset.

- The z value `0.8` aligns with the reward target in Eq. (41)

### `ball_radius`, `box_size`

Robosuite object geometry settings.

### `ball_density`, `ball_friction`, `box_density`, `box_friction`

Robosuite physical constants for the reconstruction.

These are implementation details, not specified in the paper.

### `magnetic_force`

Force magnitude used in the reconstructed Magnetic environment.

The paper specifies the qualitative magnetic effect, but not this exact coefficient.

### `reach_threshold`

Success threshold.

- `0.05`
- Paper source: Appendix C.1.2

### `reward_scale`

Scale inside the Magnetic reward function.

- `5.0`
- Paper source: Eq. (41)

### `goal_height`

Target z-coordinate in the Magnetic reward function.

- `0.8`
- Paper source: Eq. (41)

### `action_low`, `action_high`

Lower and upper action bounds for the 3D end-effector delta action.

### `test_params[0].name`

Name of the Magnetic OOD test split.

### `test_params[0].ood_box`

Whether to use the OOD box-position distribution at test time.

### `test_params[0].test_box_sigma`

Standard deviation of the test-time OOD box-position Gaussian.

- `100.0`
- Paper source: Appendix C.1.2

## `encoder_params`

### `encoder_type`

Observation encoder type.

- Both configs use `"identity"` because the state is already factorized

## `inference_params`

### `learn_action`

Whether the local causal mask learns action-to-state dependencies.

- Chemical: `false`
- Magnetic: `true`

### `learn_upper`

Whether the full mask structure is learned rather than only the constrained lower-triangular part.

- Chemical: `false`
- Magnetic: `true`

### `learn_std`

Whether continuous outputs predict both mean and standard deviation.

- Chemical: `false`
- Magnetic: `true`
- Paper source: Appendix C.3 says continuous variables output mean and standard deviation

### `n_pred_step`

Prediction horizon used in dynamics training.

- Chemical: `3`
- Magnetic: `1`

### `batch_size`

Training batch size.

- `256`
- Paper source: Table 4

### `lr`

Learning rate.

- `1e-4`
- Paper source: Table 4

### `eval_freq`

Inference validation frequency.

### `train_prop`

Fraction of data assigned to the training partition in replay.

### `residual`

Whether the model predicts residual updates instead of absolute next-state values.

### `log_std_min`, `log_std_max`

Clamps applied to predicted log standard deviations.

### `grad_clip_norm`

Gradient clipping threshold for inference training.

### `eval_batch_size`

Batch size used for inference evaluation.

### `use_gt_global_mask`

Whether to bypass learned masks and use the environment's ground-truth global mask.

### `mlp_params.fc_dims`

MLP baseline hidden widths.

- Chemical: `[1024, 1024, 1024]`
- Magnetic: `[512, 512, 512, 512]`
- Paper source: Table 6

### `gnn_params.node_attr_dim`

GNN node-attribute dimension.

- `256`
- Paper source: Table 6

### `gnn_params.edge_attr_dim`

GNN edge-attribute dimension.

- `256`
- Paper source: Table 6

### `gnn_params.embedder_dims`

Optional extra embedder layers.

### `gnn_params.edge_net_dims`

GNN edge-network hidden widths.

- `[512, 512, 512]`
- Paper source: Table 6

### `gnn_params.node_net_dims`

GNN node-network hidden widths.

- `[512, 512, 512]`
- Paper source: Table 6

### `gnn_params.projector_dims`

Optional output projection layers.

## `policy_params`

### `batch_size`

Policy-related batch size.

- `256`
- Paper source: Table 4

### `lr`

Policy-related learning rate.

- `1e-4`
- Paper source: Table 4

### `n_reward_step`

Reward-prediction rollout length used by the model-based policy code.

### `discount`

Discount factor.

### `model_based_params.fc_dims`

Reward-model MLP widths used by this codebase's model-based planner.

These are implementation details and are not listed in the paper tables.

### `model_based_params.activations`

Reward-model activation functions.

### `model_based_params.planner_type`

Planner type.

- `"cem"`
- Paper source: Appendix C.2

### `model_based_params.std_scale`

Initial scale of the continuous CEM sampling distribution.

### `model_based_params.n_horizon_step`

Planning length.

- Chemical: `3`
- Magnetic: `1`
- Paper source: Table 5

### `model_based_params.n_iter`

Number of CEM iterations.

- `5`
- Paper source: Table 5

### `model_based_params.n_candidate`

Number of sampled CEM candidates.

- `64`
- Paper source: Table 5

### `model_based_params.n_top_candidate`

Number of top candidates retained per CEM iteration.

- `32`
- Paper source: Table 5

### `model_based_params.action_noise`

Action-space exploration noise.

- Chemical: kept for API compatibility, effectively unused because Chemical actions are discrete
- Magnetic: `1e-4`
- Paper source for Magnetic: Table 5

### `model_based_params.action_noise_eps`

Exploration probability for discrete action selection.

- Chemical: `0.05`
- Magnetic: unused for continuous actions
- Paper source for Chemical: Table 5
