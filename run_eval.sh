python submit.py \
    ./llava/eval/conme/conme_eval_workflow.sh\
    --ems_project thesis-benchmarking \
    --experiment_name conme_llama3b_learnable_encoding \
    --cluster tess137 \
    -n chatgpt \
    -i hub.tess.io/vorshulevich/vllm:latest \
    --gpu_per_node 1 \
    --num_nodes 1 \
    --cpu 60 \
    --memory 512 \
    --pvc