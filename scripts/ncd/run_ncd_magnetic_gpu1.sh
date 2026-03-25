for seed in 1 2; do
    python main_policy.py \
        --config=policy_params_magnetic.json \
        --training_params.inference_algo=ncd --fcdl_params.code_labeling=True --cuda_id=1 --seed=$seed
done
