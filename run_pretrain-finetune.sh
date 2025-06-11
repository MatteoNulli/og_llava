python submit.py \
    ./scripts/train/pretrain-finetune_thesis-llama31-8b.sh \
    --ems_project thesis-training \
    --experiment_name llama31-8B-2d_sinusoidal_encoding_fixed_aftermlp_noglob-sft \
    --cluster tess137 \
    -n chatgpt-training-slc-a100 \
    -i hub.tess.io/vorshulevich/vllm:latest \
    --gpu_per_node 8 \
    --num_nodes 1 \
    --cpu 60 \
    --memory 512 \
    --pvc