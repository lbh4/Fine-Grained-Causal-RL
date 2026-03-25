for seed in 7 8; do
    python main_policy.py \
        --config=policy_params_magnetic.json \
        --training_params.inference_algo=oracle --inference_params.use_gt_global_mask=True \
        --fcdl_params.code_labeling=False --cuda_id=4 --seed=$seed
done
