#!/bin/bash
# This script evaluates the MAS model on the test set.
export MAX_WORKERS_NUM=60

MODELS=(
    saved/Qwen2.5-3B-Instruct
)
IPS=(
    "0.0.0.0" 
)
PORT_NUMBERS=(
    "9090"
)
MODE="reflection"  # mode can empowered or reflection
TAG="xxxx"  # dont use - or _ in tag
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