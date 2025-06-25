#!/bin/bash


export http_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export https_proxy=http://httpproxy-tcop.vip.ebay.com:80 
export no_proxy=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com
export HTTP_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export HTTPS_PROXY=http://httpproxy-tcop.vip.ebay.com:80
export NO_PROXY=krylov,ams,ems,mms,localhost,127.0.0.1,.vip.hadoop.ebay.com,.vip.ebay.com,github.ebay.com,.tess.io,.corp.ebay.com,.ebayc3.com,.qa.ebay.com,.dev.ebay.com


# pip install --proxy http://httpproxy-tcop.vip.ebay.com:80 open-clip-torch
# pip install --proxy http://httpproxy-tcop.vip.ebay.com:80 decord
# pip install --proxy http://httpproxy-tcop.vip.ebay.com:80 hydra-core
# pip install --proxy http://httpproxy-tcop.vip.ebay.com:80 iopath
# pip install --proxy http://httpproxy-tcop.vip.ebay.com:80 pycocotools

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1

NUM_MACHINES=${NUM_MACHINES:-1}
NUM_GPUS=${NUM_GPUS:-1}

# MODEL_DIR=/mnt/nushare2/data/mnulli/model_zoos/opensource-vlms/models--OpenGVLab--InternVL2_5-8B/snapshots/d64b85a1392275381ddbb7525db05e587303d59e
# MODEL_NAME='InternVL2_5-8B'
# CONV_MODE='llava_llama_3'
# MODEL_BASE=$MODEL_DIR


# MODEL_DIR=/mnt/nushare2/data/mnulli/model_zoos/opensource-vlms/models--llava-hf--llama3-llava-next-8b-hf/snapshots/b041c0d0ea0dd0196d147206c210c8d1752fc2da
# MODEL_NAME='llava-hf'
# CONV_MODE='llava_llama_3'
# MODEL_BASE=$MODEL_DIR


# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_global_view_llava-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_global_view_masklimiting20v2_llava-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/4b_global_view_llava-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/no_global_view_llava-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_standard_llava/standard_llava15-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_no_global_view_llava-Meta-Llama-3_1-8B-Instruct-openclip-bliplaion-lora
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_no_masktoken-Meta-Llama-3_1-8B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-EPOCHS
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_avg_global-view-Meta-Llama-3_1-8B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-EPOCHS
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_slidingwindow_no_globview-Meta-Llama-3_1-8B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-EPOCHS
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_avg_global-view-Meta-Llama-3_1-8B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-2-capEpochs-1-EPOCHS
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_avg_global-view-Meta-Llama-3_1-8B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-2-capEpochs-2-sftEpochs
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_no_global_view_oldllavacodebase-meta-llama--Llama-3.2-1B-Instruct-openclip-bliplaion_llava-lora-1-EPOCHS
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_no_global_view_oldllavacodebase-meta-llama--Llama-3.2-1B-Instruct-openclip-bliplaion_llava-lora-3-EPOCHS
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_standard_llava/standard_llava15-meta-llama--Llama-3.2-1B-Instruct-openclip-bliplaion-lora-1-EPOCHS
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/subobject_tokenization-second_run-Meta-Llama-3_1-8B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-capEpochs-1-sftEpochs
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_global-view-Meta-Llama-3_1-8B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-2-capEpochs-1-sftEpochs
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/sliding_window_5-Meta-Llama-3_1-8B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-capEpochs-1-sftEpochs
# # MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/sliding_window_10-notokens-Meta-Llama-3_1-8B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-capEpochs-1-sftEpochs
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_3b/custom_rot-noglob-meta-llama--Llama-3.2-3B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-EPOCHS
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_standard_llava/standard-meta-llama--Llama-3.2-3B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-EPOCHS

# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_3b/abs_pos_emb_noglob-meta-llama--Llama-3.2-3B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-EPOCHS
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft/8bs_avg_global-view-Meta-Llama-3_1-8B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-capEpochs-1-sftEpochs

# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_3b/abs_pos_emb_fixed_aftermlp_noglob-meta-llama--Llama-3.2-3B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-EPOCHS
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_3b/learnable_encoding_aftermlp_noglob-meta-llama--Llama-3.2-3B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-EPOCHS

# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_standard_llava/standard-meta-llama--Llama-3.2-3B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-Cambrian7M_withsystemprompt.json-v2-lora-1-EPOCHS
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_3b/noglob-meta-llama--Llama-3.2-3B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-Cambrian7M_withsystemprompt.json-lora-1-EPOCHS

# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_8b/sinusoidal_encoding_fixed_aftermlp_noglob-Meta-Llama-3_1-8B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-EPOCHS
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_8b/learnable_encoding_aftermlp_noglob-Meta-Llama-3_1-8B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-EPOCHS
MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_8b/2d_sinusoidal_encoding_fixed_aftermlp_noglob-Meta-Llama-3_1-8B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-EPOCHS
# MODEL_DIR=/mnt/nushare2/data/mnulli/thesis/testruns/sft_3b/global_view-meta-llama--Llama-3.2-3B-Instruct-openai-clip-vit-large-patch14-336-blip_laion-llava_mix665k-lora-1-EPOCHS


MODEL_NAME=$(basename "$MODEL_DIR")
MODEL_BASE=/mnt/mtrepo/data/wwalentynowicz/models/Meta-Llama-3_1-8B-Instruct
# MODEL_BASE=/mnt/nushare2/data/mnulli/model_zoos/language_models/meta-llama--Llama-3.2-3B-Instruct
# MODEL_BASE=/mnt/nushare2/data/mnulli/model_zoos/language_models/meta-llama--Llama-3.2-1B-Instruct
CONV_MODE='llama3'


# MODEL_DIR=/mnt/nushare2/data/mnulli/model_zoos/opensource-vlms/models--ByteDance--Sa2VA-8B/snapshots/43ee408e24e7fc571a4e33862f663c2dbc6e11da
# MODEL_NAME=$(basename "$MODEL_DIR")
# MODEL_BASE=/mnt/mtrepo/data/wwalentynowicz/models/Meta-Llama-3_1-8B-Instruct
# # # MODEL_BASE=/mnt/nushare2/data/mnulli/model_zoos/language_models/meta-llama--Llama-3.2-1B-Instruct
# CONV_MODE='llama3'



cd llava/eval/conme/

accelerate launch --num_machines $NUM_MACHINES --num_processes $NUM_GPUS --main_process_port 12380 --mixed_precision no --dynamo_backend no \
    eval_conme.py \
    --model-path $MODEL_DIR \
    --model-base $MODEL_BASE \
    --question-file /mnt/nushare2/data/mnulli/thesis/data/benchmarks/conme/data \
    --image-folder /mnt/nushare2/data/mnulli/thesis/data/benchmarks/conme/ \
    --answers-file /mnt/nushare2/data/mnulli/thesis/data/benchmarks/conme/answers/1_$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE \
    --result-dir /mnt/nushare2/data/mnulli/thesis/data/benchmarks/conme/answers \
    --upload-dir /mnt/nushare2/data/mnulli/thesis/data/benchmarks/conme/answers_upload/$MODEL_NAME \
    --experiment $MODEL_NAME \
    # --sam2 True \
    # --custom_rotary_embedding True \