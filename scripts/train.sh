#!/bin/bash
CONFIGS=(
    grpo-llama32-3b-MAS-DS-DR.yaml
)

for dir in saved logs; do
  [ -d "$dir" ] || mkdir "$dir"
done



for config in "${CONFIGS[@]}"; do
    training=$(basename "$config" .yaml | cut -d'-' -f1)
    task=$(basename "$config" .yaml)

    if [[ "$training" == "grpo" ]]; then
        export NCCL_IB_DISABLE=1
        export NCCL_P2P_DISABLE=1
        export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
        MODEL_NAME=$(awk -F': ' '/model_name_or_path:/ {print $2; exit}' "recipes/train_configs/$config" | awk '{print $1}')
        CUDA_VISIBLE_DEVICES="0" trl vllm-serve \
            --model "$MODEL_NAME" \
            --gpu_memory_utilization 0.9 \
            --tensor_parallel_size 1 \
            --data_parallel_size 1 \
            --port 8000 \
            --host 0.0.0.0 &
    else
        export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    fi

    echo "Starting training for $task"

    if [[ "$task" == *"14b"* ]]; then
        config_file="recipes/accelerate_configs/ds3.yaml"
    else
        config_file="recipes/accelerate_configs/ds2.yaml"
    fi

    ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file "$config_file" \
    --num_processes $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
    --main_process_port 29512 \
    src/MAS/"$training".py \
    recipes/train_configs/"$config" 2>&1 | tee "logs/${task}.log"

    if [[ "$training" == "grpo" ]]; then
        sleep 5
        bash scripts/kill_vllm.sh
    fi

    sleep 5
done
