# TimeLLM_CGM
====

Python library that provides functions for forecasting the glucose levels in a variable prediction horizon.

## Installation and setup

To download the source code, you can clone it from the Github repository.
```console
git clone git@github.com:franciscojesuslara/TimeLLM_CGM.git
```

Before installing libraries, ensuring that a Python virtual environment is activated (using conda o virtualenv). To install Python libraries run: 

```console
pip install -r requirements.txt 
```

If you have any problem by installing libraries, first execute:

```console
curl https://bootstrap.pypa.io/get-pip.py | python -
```

## Execute scripts for training forecasting models

To train LLM-based models:
```console
python src/llm.py --model_name='gpt' --input_seq_len=96 --batch_size=5 --windows_batch_size=5 --n_samples=10 --num_workers=8 --validation_size=200 --scaler='minmax' --max_steps=10
```
To train classical models:
```console
python src/classical_models.py --dataset_name='vivli_mdi' --prediction_horizon == 6 --input_seq_len=96 
