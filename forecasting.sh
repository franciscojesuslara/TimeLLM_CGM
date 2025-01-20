#!/bin/bash

#TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=3 src/llm.py --prediction_horizon=12 --model_name='bert'
#TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 TORCH_NCCL_BLOCKING_WAIT=1 torchrun --nproc_per_node=3 src/llm.py --prediction_horizon=12 --model_name='bert' --max_steps=4 --num_workers_loader=0
#TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 TORCH_NCCL_BLOCKING_WAIT=1 torchrun --rdzv_backend=gloo --nproc_per_node=3 src/llm.py --prediction_horizon=12 --model_name='bert' --max_steps=4 --num_workers_loader=0



#


python3 src/llm.py --prediction_horizon=4 --model_name='bert' --dataset_name='vivli_pump' --freq_sample=15 --ts_length=96 --gpu_device=0
python3 src/llm.py --prediction_horizon=6 --model_name='bert '--dataset_name='vivli_pump' --freq_sample=15 --ts_length=96 --gpu_device=0
python3 src/llm.py --prediction_horizon=8 --model_name='bert' --dataset_name='vivli_pump' --freq_sample=15 --ts_length=96 --gpu_device=0
python3 src/llm.py --prediction_horizon=4 --model_name='gpt' --dataset_name='vivli_pump' --freq_sample=15 --ts_length=96 --gpu_device=0
python3 src/llm.py --prediction_horizon=6 --model_name='gpt' --dataset_name='vivli_pump' --freq_sample=15 --ts_length=96 --gpu_device=0
python3 src/llm.py --prediction_horizon=8 --model_name='gpt' --dataset_name='vivli_pump' --freq_sample=15 --ts_length=96 --gpu_device=0

python3 src/llm.py --prediction_horizon=4 --model_name='bert' --dataset_name='vivli_mdi' --freq_sample=15 --ts_length=96 --gpu_device=0
python3 src/llm.py --prediction_horizon=6 --model_name='bert '--dataset_name='vivli_mdi' --freq_sample=15 --ts_length=96 --gpu_device=0
python3 src/llm.py --prediction_horizon=8 --model_name='bert' --dataset_name='vivli_mdi' --freq_sample=15 --ts_length=96 --gpu_device=0
python3 src/llm.py --prediction_horizon=4 --model_name='gpt' --dataset_name='vivli_mdi' --freq_sample=15 --ts_length=96 --gpu_device=0
python3 src/llm.py --prediction_horizon=6 --model_name='gpt' --dataset_name='vivli_mdi' --freq_sample=15 --ts_length=96 --gpu_device=0
python3 src/llm.py --prediction_horizon=8 --model_name='gpt' --dataset_name='vivli_mdi' --freq_sample=15 --ts_length=96 --gpu_device=0

python3 src/classical_models.py --prediction_horizon=4
python3 src/classical_models.py --prediction_horizon=6
python3 src/classical_models.py --prediction_horizon=8


python3 src/llm.py --prediction_horizon=12 --model_name='bert' --dataset_name='Ohio' --freq_sample=15 --ts_length=288 --gpu_device=0
python3 src/llm.py --prediction_horizon=16 --model_name='bert '--dataset_name='Ohio' --freq_sample=15 --ts_length=288 --gpu_device=0
python3 src/llm.py --prediction_horizon=18 --model_name='bert' --dataset_name='Ohio' --freq_sample=15 --ts_length=288 --gpu_device=0
python3 src/llm.py --prediction_horizon=12 --model_name='gpt' --dataset_name='Ohio' --freq_sample=15 --ts_length=288 --gpu_device=0
python3 src/llm.py --prediction_horizon=16 --model_name='gpt' --dataset_name='Ohio' --freq_sample=15 --ts_length=288 --gpu_device=0
python3 src/llm.py --prediction_horizon=18 --model_name='gpt' --dataset_name='Ohio' --freq_sample=15 --ts_length=288 --gpu_device=0

python3 src/classical_models.py --prediction_horizon=12 --dataset_name='Ohio'
python3 src/classical_models.py --prediction_horizon=16 --dataset_name='Ohio'
python3 src/classical_models.py --prediction_horizon=18 --dataset_name='Ohio'


#export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
