for seed in 7 8; do
    python main_policy.py \
        --config=policy_params_magnetic.json \
        --training_params.inference_algo=gnn --cuda_id=4 --seed=$seed
done
