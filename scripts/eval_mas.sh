#!/bin/bash
# This script evaluates the MAS model on the test set.
export MAX_WORKERS_NUM=10

MODELS=(
    # /cpfs01/mj/eval_models/Qwen2.5-32B-Instruct
    # /cpfs01/mj/eval_models/Qwen2.5-72B-Instruct
    # "gpt-5-2025-08-07"
    # "gemini-2.5-pro"
    # /cpfs01/mj/eval_models/Llama-3.3-70B-Instruct
    # /cpfs01/zileq/models/openai/gpt-oss-120b
    /cpfs01/zhuochen.zc/models/ZhipuAI/GLM-4___5
)
IPS=(
    "0.0.0.0" 
    "0.0.0.0"
)
PORT_NUMBERS=(
    "9090"
    "9091"
)
MODE="reflection"  # mode can empowered or reflection
TAG="0917"  # dont use - or _ in tag
SAVE_ROOT="mas_eval"
LOG_FILE="${SAVE_ROOT}/EVAL_${TAG}-${MODE}.log"

# create save root
if [ ! -d "${SAVE_ROOT}" ]; then
    mkdir -p "${SAVE_ROOT}"
fi

# Create directories for each model and run evaluation in parallel
for i in "${!MODELS[@]}"; do
    model=${MODELS[$i]}
    ip=${IPS[$i]}
    port=${PORT_NUMBERS[$i]}
    model_dir="${SAVE_ROOT}/$(echo ${model} | cut -d'/' -f2- | tr '/' '_')"
    mkdir -p "${model_dir}"

    ( 
        python src/MAS/eval_mas.py \
        --models ${model} \
        --ips ${ip} \
        --port_numbers ${port} \
        --temperature 0.7 \
        --save_root ${SAVE_ROOT} \
        --dataset_path data/final_test \
        --mode ${MODE} \
        --tag ${TAG} 2>&1 | tee -a ${LOG_FILE} 
    
        python src/MAS/eval_analysis.py \
            --input_dir ${model_dir} 2>&1 | tee ${model_dir}/ANALYSIS_${TAG}-${MODE}.log 
    ) &
done

# --testing