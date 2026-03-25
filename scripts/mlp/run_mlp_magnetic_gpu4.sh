for seed in 7 8; do
    python main_policy.py \
        --config=policy_params_magnetic.json \
        --training_params.inference_algo=mlp --cuda_id=4 --seed=$seed
done
