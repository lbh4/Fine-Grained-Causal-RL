for seed in 5 6; do
    python main_policy.py \
        --config=policy_params_magnetic.json \
        --training_params.inference_algo=ncd --fcdl_params.code_labeling=True --cuda_id=3 --seed=$seed
done
