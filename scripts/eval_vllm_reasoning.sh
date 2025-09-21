# Start vllm server
MODELS=(
    /cpfs01/zhuochen.zc/models/ZhipuAI/GLM-4___5
)
IPS=(
    "0.0.0.0" 
)
PORT_NUMBERS=(
    "9090"     
)
CUDA_DEVICES=(
    "0,1,2,3,4,5,6,7"
)
MAX_LENGTH=16384 # 8192

for i in ${!MODELS[@]}; do
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES[$i]}" vllm serve \
        ${MODELS[$i]} \
        --host ${IPS[$i]} \
        --port ${PORT_NUMBERS[$i]} \
        --tensor-parallel-size $(echo ${CUDA_DEVICES[$i]} | tr -cd ',' | wc -c | xargs -I {} expr {} + 1) \
        --gpu-memory-utilization 0.8 \
        --max-model-len ${MAX_LENGTH} \
        --reasoning-parser glm45 \
        --dtype bfloat16 &
done