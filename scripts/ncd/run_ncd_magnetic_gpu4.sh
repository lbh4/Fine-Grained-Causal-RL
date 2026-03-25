for seed in 7 8; do
    python main_policy.py \
        --config=policy_params_magnetic.json \
        --training_params.inference_algo=ncd --fcdl_params.code_labeling=True --cuda_id=4 --seed=$seed
done
