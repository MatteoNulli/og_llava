python submit.py \
    automatic_masks.sh \
    --ems_project thesis-training \
    --experiment_name automatic_masks_aro \
    --cluster tess137 \
    -n chatgpt-training-slc-a100 \
    -i hub.tess.io/vorshulevich/vllm:latest \
    --gpu_per_node 2 \
    --num_nodes 1 \
    --cpu 16 \
    --memory 128 \
    --pvc