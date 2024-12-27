#!/bin/bash

python3 src/classical_models.py --prediction_horizon =12
python3 src/classical_models.py --prediction_horizon =18
python3 src/classical_models.py --prediction_horizon =24


python3 src/llm.py --prediction_horizon =12 --model_name ='bert'
python3 src/llm.py --prediction_horizon =18 --model_name ='bert'
python3 src/llm.py --prediction_horizon =24 --model_name ='bert'

python3 src/llm.py --prediction_horizon =12 --model_name ='gpt'
python3 src/llm.py --prediction_horizon =18 --model_name ='gpt'
python3 src/llm.py --prediction_horizon =24 --model_name ='gpt'

