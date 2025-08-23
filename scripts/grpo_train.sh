#!/bin/bash
# A wrapper script to start a vLLM server on a remote machine and then initiate local training.

# --- Configuration ---
# The SSH target for the remote vLLM server (e.g., user@192.168.1.100).
# You can override this by passing it as the first argument to the script.
export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

VLLM_MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
)
REMOTE_SSH_TARGETS=(
    "ubuntu@152.69.171.170"
)
VLLM_PORT="8000"

TASKS=(
    "grpo-qwen25-3b-NORMAL" 
)


for i in "${!TASKS[@]}"; do
    vllm_model=${VLLM_MODELS[$i]}
    remote_ssh_target=${REMOTE_SSH_TARGETS[$i]}
    task=${TASKS[$i]}
    vllm_host=$(echo ${remote_ssh_target} | cut -d'@' -f2)
    echo "--> Starting training for ${task}"

    echo "--> Attempting to start vLLM server on remote host: ${remote_ssh_target}"
    # We use -o "StrictHostKeyChecking=no" to avoid interactive prompts for new hosts.
    ssh -o "StrictHostKeyChecking=no" ${remote_ssh_target} '
        echo "--> [Remote] Starting new vLLM server in the background..."
        nohup CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" trl vllm-serve \
            --model ${vllm_model} \
            --gpu_memory_utilization 0.9 \
            --tensor-parallel-size 8 \
            > ~/vllm_server.log 2>&1 &
        echo "--> [Remote] vLLM server started. Check ~/vllm_server.log for output."
    '

    echo "--> Proceeding with local training setup. Waiting 60s for server to initialize..."
    sleep 60

    ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/ds2.yaml \
    --num_processes $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
    --main_process_port 29511 \
    src/MAS/grpo.py \
    recipes/train_configs/${task}.yaml \
    --vllm_server_host=${vllm_host} \
    --vllm_server_port=${VLLM_PORT} 2>&1 | tee logs/${task}.log

    sleep 5
    echo "--> Training finished for ${task}. Stopping vLLM server on ${remote_ssh_target}..."
    ssh -o "StrictHostKeyChecking=no" ${remote_ssh_target} '
        pkill -f "trl vllm-serve"
        pkill -f "from multiprocessing.spawn import spawn_main"
        echo "--> [Remote] vLLM server stopped."
    '
    sleep 5
done

# --debug="underflow_overflow"