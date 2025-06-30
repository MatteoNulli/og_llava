python submit.py \
    ./scripts/train/pretrain-finetune_thesis-llama32-Cambrian.sh \
    --ems_project thesis-training \
    --experiment_name llama32-3B-cambrian-nomasktoken_no_globalview-sft \
    --cluster tess137 \
    -n chatgpt-training-slc-a100 \
    -i hub.tess.io/vorshulevich/vllm:latest \
    --gpu_per_node 8 \
    --num_nodes 1 \
    --cpu 60 \
    --memory 512 \
    --pvc