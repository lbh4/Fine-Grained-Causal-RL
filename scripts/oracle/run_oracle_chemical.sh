for seed in 1 2 3 4 5 6 7 8; do
    python main_policy.py \
        --config=policy_params_chemical.json \
        --training_params.inference_algo=oracle --inference_params.use_gt_global_mask=True \
        --fcdl_params.code_labeling=False --cuda_id=0 --seed=$seed
done
