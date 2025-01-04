#!/bin/bash

TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=3 src/llm.py --prediction_horizon=12 --model_name='bert'
TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 TORCH_NCCL_BLOCKING_WAIT=1 torchrun --nproc_per_node=3 src/llm.py --prediction_horizon=12 --model_name='bert' --max_steps=4 --num_workers_loader=0
TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 TORCH_NCCL_BLOCKING_WAIT=1 torchrun --rdzv_backend=gloo --nproc_per_node=3 src/llm.py --prediction_horizon=12 --model_name='bert' --max_steps=4 --num_workers_loader=0

python3 src/llm.py --prediction_horizon=12 --model_name='bert'
python3 src/llm.py --prediction_horizon=18 --model_name='bert'
python3 src/llm.py --prediction_horizon=24 --model_name='bert'
python3 src/llm.py --prediction_horizon=12 --model_name='gpt'
python3 src/llm.py --prediction_horizon=18 --model_name='gpt'
python3 src/llm.py --prediction_horizon=24 --model_name='gpt'
python3 src/classical_models.py --prediction_horizon=12
python3 src/classical_models.py --prediction_horizon=18
python3 src/classical_models.py --prediction_horizon=24


export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
