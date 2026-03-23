for seed in 1 2 3 4 5 6 7 8; do
    python main_policy.py \
        --config=policy_params_magnetic.json \
        --training_params.inference_algo=ours --cuda_id=5 --seed=$seed
done
