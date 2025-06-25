python ./krylov/submit.py \
    ./krylov/scripts/eval_lmms_workflow.sh \
    --ems_project thesis-benchmarking \
    --experiment_name aro_cambrian \
    --cluster tess137 \
    -n chatgpt-training-slc-a100 \
    -i hub.tess.io/vorshulevich/vllm:latest \
    --gpu_per_node 2 \
    --num_nodes 1 \
    --cpu 16 \
    --memory 512 \
    --pvc