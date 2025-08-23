# Start vllm server
MODELS=(
    saved/sft-llama32-3b/checkpoint-47
)
IPS=(
    "0.0.0.0" 
)
PORT_NUMBERS=(
    "9090"     
)
CUDA_DEVICES=(
    "0"
)
MAX_LENGTH=4096

for i in ${!MODELS[@]}; do
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES[$i]}" vllm serve \
        --model ${MODELS[$i]} \
        --host ${IPS[$i]} \
        --port ${PORT_NUMBERS[$i]} \
        --tensor-parallel-size $(echo ${CUDA_DEVICES[$i]} | tr -cd ',' | wc -c | xargs -I {} expr {} + 1) \
        --gpu-memory-utilization 0.8 \
        --max-model-len ${MAX_LENGTH} \
        --dtype bfloat16 &
done