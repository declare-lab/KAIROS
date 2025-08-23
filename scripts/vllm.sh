#!/bin/bash

CUDA_VISIBLE_DEVICES="0" trl vllm-serve \
    --model Qwen/Qwen2.5-3B-Instruct \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 1 \
    --data_parallel_size 1 \
    --port 8000 \
    --host 0.0.0.0