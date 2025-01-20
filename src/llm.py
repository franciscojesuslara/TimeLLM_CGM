from numpy import count_nonzero
from neuralforecast import NeuralForecast
from neuralforecast.models import TimeLLM
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, LlamaTokenizer, LlamaModel, LlamaConfig, AutoTokenizer, \
    BertModel, BertTokenizer,BertConfig
from utils.extract_series_llm import extract_series_general
from utils.metrics import check_model_performance_llm, save_results_forecasting
import argparse
import time
import numpy as np
import os
import utils.constants as cons
from utils.metrics import evaluate_performance, evaluate_performance_llm, evaluate_performance_intrapatient
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
def select_llm(name_llm: str):
    if name_llm == 'gpt':
        llm_config = GPT2Config.from_pretrained('openai-community/gpt2')
        llm_model = GPT2Model.from_pretrained('openai-community/gpt2', config=llm_config)
        llm_tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')

    elif name_llm == 'llama3':
        llm_config = LlamaConfig.from_pretrained(
            'meta-llama/Meta-Llama-3-8B',
            token='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',
            trust_remote_code=True,
            local_files_only=False
        )
        llm_model = LlamaModel.from_pretrained(
                            'meta-llama/Meta-Llama-3-8B',
                    token='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',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=llm_config)
        llm_tokenizer = AutoTokenizer.from_pretrained(
                            'meta-llama/Meta-Llama-3-8B',
                    token='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',
                    trust_remote_code=True,
                    local_files_only=False)
    elif name_llm == 'llama2':
        llm_config = LlamaConfig.from_pretrained(
            'meta-llama/Llama-2-7b-chat-hf',
            token='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',
            trust_remote_code=True,
            local_files_only=False)
        llm_model = LlamaModel.from_pretrained(
            'meta-llama/Llama-2-7b-chat-hf',
            token='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',
            trust_remote_code=True,
            local_files_only=False,
            config=llm_config)
        llm_tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Llama-2-7b-chat-hf',
            token='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',
            trust_remote_code=True,
            local_files_only=False)

    elif name_llm == 'bert':
        llm_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

        llm_model = BertModel.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False,
                config=llm_config,
            )
        llm_tokenizer= BertTokenizer.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False)
    return llm_config, llm_model, llm_tokenizer


def parse_arguments(parser):
    parser.add_argument('--model_name', type=str, default='bert')
    parser.add_argument('--scaler', type=str, default='minmax')
    parser.add_argument('--dataset_name', type=str, default='Ohio')
    parser.add_argument('--prediction_horizon', type=int, default=12)
    parser.add_argument('--ts_length', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--windows_batch_size', type=int, default=5)
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers_loader', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=3000)
    parser.add_argument('--read_plot', type=bool, default=False),
    parser.add_argument('--n_windows', type=int, default=50)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--freq_sample', type=int, default=15)
    parser.add_argument('--test_iterations', type=int, default=5)
    parser.add_argument('--test_step_size', type=int, default=24)
    parser.add_argument('--gpu_device', type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM forecasting')
    args = parse_arguments(parser)

    df, train_all, test_all = extract_series_general(dataset_name=args.dataset_name,
                                             n_samples=args.n_samples,
                                             prediction_horizon=args.prediction_horizon,
                                             ts_length=args.ts_length,
                                             freq_sample=args.freq_sample,
                                             step_size=args.step_size,
                                             n_windows=args.n_windows,
                                             )
    print(test_all['unique_id'].value_counts())

    if args.read_plot:
        forecasts = pd.read_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                    f'forecasts_{args.dataset_name}_{args.model_name}_{args.prediction_horizon}.csv'))
        aggregated_losses_test = pd.read_csv(
            os.path.join(cons.PATH_PROJECT_REPORTS,
                         f'aggregated_losses_test_{args.dataset_name}_{args.prediction_horizon}.csv'))
        best_model_per_patient_test = pd.read_csv(
            os.path.join(cons.PATH_PROJECT_REPORTS,
                         f'best_model_per_patient_test_{args.dataset_name}_{args.prediction_horizon}.csv'))
        losses_test = pd.read_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                               f'losses_test_{args.dataset_name}_{args.prediction_horizon}.csv'))
        aggregated_losses_val = pd.read_csv(
            os.path.join(cons.PATH_PROJECT_REPORTS,
                         f'aggregated_losses_val_{args.dataset_name}_{args.prediction_horizon}.csv'))
        best_model_per_patient_val = pd.read_csv(
            os.path.join(cons.PATH_PROJECT_REPORTS,
                         f'best_model_per_patient_val_{args.dataset_name}_{args.prediction_horizon}.csv'))
        losses_val = pd.read_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                              f'losses_val_{args.dataset_name}_{args.prediction_horizon}.csv'))
    else:
        print(args.dataset_name, args.model_name)

        count=0
        id = list(set(train_all.unique_id.values))
        llm_config, llm_model, llm_tokenizer = select_llm(name_llm=args.model_name)

        prompt_prefix = "{}_llm".format(args.model_name)
        start_time = time.time()
        if args.gpu_device > -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)



        if 'vivli' in args.dataset_name and int(args.freq_sample) == 5:
            for e in range(int(args.n_windows/10)):
                upper= count + 10
                if upper > len(train_all):
                    upper = -1
                train = train_all[train_all['unique_id'].isin(id[count: upper])]
                test = test_all[test_all['unique_id'].isin(id[count: upper])]
                count+=10
                timellm = TimeLLM(h=args.prediction_horizon,
                                  patch_len=64,
                                  d_model=64,
                                  top_k=10,
                                  input_size=args.ts_length,
                                  llm=llm_model,
                                  llm_config=llm_config,
                                  llm_tokenizer=llm_tokenizer,
                                  prompt_prefix=prompt_prefix,
                                  batch_size=args.batch_size,
                                  windows_batch_size=args.windows_batch_size,
                                  random_seed=args.seed,
                                  # num_workers_loader=args.num_workers_loader,
                                  max_steps=args.max_steps,
                                  )

                nf = NeuralForecast(
                    models=[timellm],
                    freq=f'{args.freq_sample}min',
                    local_scaler_type=args.scaler
                )

                cv_df = nf.cross_validation(
                    df=train,
                    id_col="unique_id",
                    time_col="time",
                    target_col="cgm",
                    verbose=True,
                    n_windows=args.n_windows,
                    step_size=args.step_size
                )

                total_time = time.time() - start_time
                print(f'Tiempo total de entrenamiento: {total_time:.2f} sec')
                columns_list = ['TimeLLM', 'cgm', 'unique_id']
                losses_val, aggregated_losses_val, best_model_per_patient_val = evaluate_performance_llm(cv_df,
                                                                                                         columns_list)

                start_time = time.time()
                test = test.sort_values(by=['unique_id', 'time'])
                forecasts_test = pd.DataFrame()
                losses_test_test_list = []
                for iterations in np.arange(args.test_iterations):
                    test_samples = iterations * args.test_step_size + args.ts_length
                    df_to_predict = test.groupby("unique_id").apply(lambda x: x.iloc[iterations:test_samples])
                    df_real = test.groupby("unique_id").apply(
                        lambda x: x.iloc[test_samples:test_samples + args.prediction_horizon])
                    forecasts = nf.predict(df=df_to_predict, verbose=True)
                    forecasts = forecasts.sort_values(by=['unique_id', 'time'])
                    forecasts['cgm'] = df_real['cgm'].values
                    forecasts['unique_id'] = forecasts.index
                    forecasts_test = pd.concat([forecasts_test, forecasts], ignore_index=True)
                    losses_test, aggregated_losses_test = evaluate_performance_intrapatient(forecasts,
                                                                                            columns_list)
                    losses_test_test_list.append(losses_test)

                total_time = time.time() - start_time
                print(f'Tiempo total de prediccion: {total_time:.2f} sec')

                df_loses_test = pd.concat(losses_test_test_list)
                result_intra = df_loses_test.groupby(["unique_id", "model"])[['mse', 'mae', 'rmse']].agg(
                    ["mean", "std"]).reset_index()
                result_intra.columns = ['_'.join(col).strip() for col in result_intra.columns.values]
                result_intra.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                                 f'results_{args.model_name}_intra_{args.dataset_name}_{args.prediction_horizon}_{args.freq_sample}_{e}.csv'))
                losses_test, aggregated_losses_test, best_model_per_patient_test = evaluate_performance(forecasts_test,
                                                                                                        columns_list,
                                                                                                        best_model_per_patient_val)
                forecasts_test.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                                   f'forecasts_{args.dataset_name}_{args.model_name}_{args.prediction_horizon}_{args.freq_sample}_{e}.csv'))
                losses_test, aggregated_losses_test, best_model_per_patient_test = evaluate_performance_llm(forecasts_test,
                                                                                                            columns_list)

                aggregated_losses_test.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                                           f'aggregated_losses_test_{args.dataset_name}_{args.model_name}_{args.prediction_horizon}_{args.freq_sample}_{e}.csv'))
                losses_test.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                                f'losses_test_{args.dataset_name}_{args.model_name}_{args.prediction_horizon}_{args.freq_sample}_{e}.csv'))
                aggregated_losses_val.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                                          f'aggregated_losses_val_{args.dataset_name}_{args.model_name}_{args.prediction_horizon}_{args.freq_sample}_{e}.csv'))

                losses_val.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                               f'losses_val_{args.dataset_name}_{args.model_name}_{args.prediction_horizon}_{args.freq_sample}_{e}.csv'))



        else:

            llm_config, llm_model, llm_tokenizer = select_llm(name_llm=args.model_name)

            prompt_prefix = "{}_llm".format(args.model_name)
            start_time = time.time()
            if args.gpu_device > -1:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)

            timellm = TimeLLM(h=args.prediction_horizon,
                              patch_len = 64,
                              d_model = 64,
                              top_k = 10,
                              input_size=args.ts_length,
                              llm=llm_model,
                              llm_config=llm_config,
                              llm_tokenizer=llm_tokenizer,
                              prompt_prefix=prompt_prefix,
                              batch_size=args.batch_size,
                              windows_batch_size=args.windows_batch_size,
                              random_seed=args.seed,
                              # num_workers_loader=args.num_workers_loader,
                              max_steps=args.max_steps,
                              )

            nf = NeuralForecast(
                models=[timellm],
                freq=f'{args.freq_sample}min',
                local_scaler_type=args.scaler
            )

            cv_df = nf.cross_validation(
                df=train_all,
                id_col="unique_id",
                time_col="time",
                target_col="cgm",
                verbose=True,
                n_windows=args.n_windows,
                step_size=args.step_size
            )

            total_time = time.time() - start_time
            print(f'Tiempo total de entrenamiento: {total_time:.2f} sec')
            columns_list = ['TimeLLM', 'cgm', 'unique_id']
            losses_val, aggregated_losses_val, best_model_per_patient_val = evaluate_performance_llm(cv_df, columns_list)

            start_time = time.time()
            test = test_all.sort_values(by=['unique_id', 'time'])
            forecasts_test = pd.DataFrame()
            losses_test_test_list = []
            for iterations in np.arange(args.test_iterations):
                test_samples = iterations * args.test_step_size + args.ts_length
                df_to_predict = test.groupby("unique_id").apply(lambda x: x.iloc[iterations:test_samples])
                df_real = test.groupby("unique_id").apply(
                    lambda x: x.iloc[test_samples:test_samples + args.prediction_horizon])
                forecasts = nf.predict(df=df_to_predict, verbose=True)
                forecasts = forecasts.sort_values(by=['unique_id', 'time'])
                forecasts['cgm'] = df_real['cgm'].values
                forecasts['unique_id'] = forecasts.index
                forecasts_test = pd.concat([forecasts_test, forecasts], ignore_index=True)
                losses_test, aggregated_losses_test = evaluate_performance_intrapatient(forecasts,
                                                                                        columns_list)
                losses_test_test_list.append(losses_test)

            total_time = time.time() - start_time
            print(f'Tiempo total de prediccion: {total_time:.2f} sec')

            df_loses_test = pd.concat(losses_test_test_list)
            result_intra = df_loses_test.groupby(["unique_id", "model"])[['mse', 'mae', 'rmse']].agg(
                ["mean", "std"]).reset_index()
            result_intra.columns = ['_'.join(col).strip() for col in result_intra.columns.values]
            result_intra.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                             f'results_{args.model_name}_intra_{args.dataset_name}_{args.prediction_horizon}_{args.freq_sample}.csv'))
            losses_test, aggregated_losses_test, best_model_per_patient_test = evaluate_performance(forecasts_test,
                                                                                                    columns_list,
                                                                                                    best_model_per_patient_val)
            forecasts_test.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                    f'forecasts_{args.dataset_name}_{args.model_name}_{args.prediction_horizon}_{args.freq_sample}.csv'))
            losses_test, aggregated_losses_test, best_model_per_patient_test = evaluate_performance_llm(forecasts_test, columns_list)

            aggregated_losses_test.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                f'aggregated_losses_test_{args.dataset_name}_{args.model_name}_{args.prediction_horizon}_{args.freq_sample}.csv'))
            losses_test.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                f'losses_test_{args.dataset_name}_{args.model_name}_{args.prediction_horizon}_{args.freq_sample}.csv'))
            aggregated_losses_val.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                f'aggregated_losses_val_{args.dataset_name}_{args.model_name}_{args.prediction_horizon}_{args.freq_sample}.csv'))

            losses_val.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                f'losses_val_{args.dataset_name}_{args.model_name}_{args.prediction_horizon}_{args.freq_sample}.csv'))



