for seed in 3 4; do
    python main_policy.py \
        --config=policy_params_magnetic.json \
        --training_params.inference_algo=gnn --cuda_id=2 --seed=$seed
done
