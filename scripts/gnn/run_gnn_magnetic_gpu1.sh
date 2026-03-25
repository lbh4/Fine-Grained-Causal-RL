for seed in 1 2; do
    python main_policy.py \
        --config=policy_params_magnetic.json \
        --training_params.inference_algo=gnn --cuda_id=1 --seed=$seed
done
