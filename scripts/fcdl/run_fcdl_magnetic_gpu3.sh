for seed in 5 6; do
    python main_policy.py \
        --config=policy_params_magnetic.json \
        --training_params.inference_algo=fcdl --cuda_id=3 --seed=$seed
done
