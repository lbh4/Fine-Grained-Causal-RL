for seed in 3 4; do
    python main_policy.py \
        --config=policy_params_magnetic.json \
        --training_params.inference_algo=oracle --inference_params.use_gt_global_mask=True \
        --fcdl_params.code_labeling=False --cuda_id=2 --seed=$seed
done
