from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.auto import AutoNHITS, AutoLSTM, AutoTCN, AutoTiDE, AutoTSMixer, AutoPatchTST
from numpy import count_nonzero
from neuralforecast import NeuralForecast
from utils.extract_series_llm import extract_series_general
from utils.metrics import evaluate_performance
import argparse
import time
import numpy as np
import os
import utils.constants as cons
from utils.plotter import plot_results, plot_metric
import pandas as pd






def parse_arguments(parser):
    parser.add_argument('--dataset_name', type=str, default='vivli_mdi')
    parser.add_argument('--prediction_horizon', type=int, default=8)
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers_loader', type=int, default=18)
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--read_plot', type=bool, default=False)
    return parser.parse_args()


def check_sparsity(m_data):
    m_data = np.nan_to_num(m_data, 0)
    sparsity = 1.0 - (count_nonzero(m_data) / float(m_data.size))
    return sparsity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM forecasting')
    args = parse_arguments(parser)


    if args.read_plot:
        forecasts = pd.read_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                             f'forecasts_{args.dataset_name}_{args.prediction_horizon}.csv'))
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

        print(best_model_per_patient_test['model'].value_counts())

        columns_list = ['AutoTCN', 'AutoLSTM', 'AutoNHITS', 'AutoTiDE', 'AutoTSMixer', 'AutoPatchTST', 'cgm',
                        'unique_id']

        losses_test, aggregated_losses_test, best_model_per_patient_test = evaluate_performance(forecasts, columns_list,
                                                                                                best_model_per_patient_val)
        personalized_test = {
            'model': 'Personalized',
            'mse_mean': best_model_per_patient_test['mse_test'].mean(),
            'mse_std': best_model_per_patient_test['mse_test'].std(),
            'mae_mean': best_model_per_patient_test['mae_test'].mean(),
            'mae_std': best_model_per_patient_test['mae_test'].std(),
            'rmse_mean': best_model_per_patient_test['rmse_test'].mean(),
            'rmse_std': best_model_per_patient_test['rmse_test'].std()}

        aggregated_losses_test = aggregated_losses_test.append(personalized_test, ignore_index=True)

        aggregated_losses_test.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                    f'aggregated_losses_test_{args.dataset_name}_{args.prediction_horizon}.csv'))
        best_model_per_patient_test.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                    f'best_model_per_patient_test_{args.dataset_name}_{args.prediction_horizon}.csv'))
        losses_test.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                    f'losses_test_{args.dataset_name}_{args.prediction_horizon}.csv'))

        administation = args.dataset_name.split('_')[1]
        ph = args.prediction_horizon * 15
        plot_metric(aggregated_losses_test, administation, ph, 'mae')

    else:

        df, train, test = extract_series_general(dataset_name=args.dataset_name,
                                                 n_samples=args.n_samples,
                                                 prediction_horizon=args.prediction_horizon
                                                 )


        start_time = time.time()

        model_list = [
            AutoTCN(h=args.prediction_horizon, config=cons.config_tcn,
                num_samples=args.n_trials, search_alg=HyperOptSearch(random_state_seed=0), cpus=args.num_workers_loader,
                    gpus=1),
            AutoLSTM(h=args.prediction_horizon, config=cons.config_lstm,
                 num_samples=args.n_trials, search_alg=HyperOptSearch(random_state_seed=0), cpus=args.num_workers_loader,
                     gpus=1),

            AutoNHITS(h=args.prediction_horizon,config=cons.config_nhits,
                  num_samples=args.n_trials, search_alg=HyperOptSearch(random_state_seed=0), cpus=args.num_workers_loader,
                      gpus=1),
            AutoTiDE(h=args.prediction_horizon, config=cons.config_tide,
                 num_samples=args.n_trials, search_alg=HyperOptSearch(random_state_seed=0), cpus=args.num_workers_loader,
                     gpus=1),

            AutoTSMixer(h=args.prediction_horizon, n_series=1, config=cons.config_tsmixer,
                    num_samples=args.n_trials, search_alg=HyperOptSearch(random_state_seed=0), cpus=args.num_workers_loader,
                        gpus=1),
            AutoPatchTST(h=args.prediction_horizon,  config=cons.config_patchtst,
                    num_samples=args.n_trials, search_alg=HyperOptSearch(random_state_seed=0), cpus=args.num_workers_loader,
                         gpus=1),
        ]

            # TODO add local scaler param 'standard', 'robust', 'robust-iqr', 'minmax' or 'boxcox'
        nf = NeuralForecast(
            models = model_list,
            freq = '15min'
        )

        cv_df = nf.cross_validation(
            df=train,
            id_col="unique_id",
            time_col="time",
            target_col="cgm",
            verbose=True,
            n_windows=50,
            step_size=1)

        forecasts = nf.predict(futr_df=test, verbose=True)
        # columns_list=['AutoNHITS', 'AutoTiDE', 'AutoTSMixer', 'AutoPatchTST','cgm', 'unique_id']
        columns_list=['AutoTCN', 'AutoLSTM', 'AutoNHITS', 'AutoTiDE', 'AutoTSMixer', 'AutoPatchTST','cgm', 'unique_id']

        losses_val, aggregated_losses_val, best_model_per_patient_val = evaluate_performance(cv_df, columns_list)
        test = test.sort_values(by=['unique_id', 'time'])
        forecasts = forecasts.sort_values(by=['unique_id', 'time'])
        forecasts['cgm'] = test['cgm'].values
        losses_test, aggregated_losses_test, best_model_per_patient_test = evaluate_performance(forecasts, columns_list, best_model_per_patient_val)
        train = train.reset_index()
        test = test.reset_index()
        forecasts=forecasts.reset_index()
        personalized_val = {
            'model' : 'Personalized',
            'mse_mean' : best_model_per_patient_val['mse'].mean(),
            'mse_std': best_model_per_patient_val['mse'].std(),
            'mae_mean': best_model_per_patient_val['mae'].mean(),
            'mae_std': best_model_per_patient_val['mae'].std(),
            'rmse_mean' : best_model_per_patient_val['rmse'].mean(),
            'rmse_std': best_model_per_patient_val['rmse'].std()}

        personalized_test = {
            'model': 'Personalized',
            'mse_mean': best_model_per_patient_test['mse_test'].mean(),
            'mse_std': best_model_per_patient_test['mse_test'].std(),
            'mae_mean': best_model_per_patient_test['mae_test'].mean(),
            'mae_std': best_model_per_patient_test['mae_test'].std(),
            'rmse_mean': best_model_per_patient_test['rmse_test'].mean(),
            'rmse_std': best_model_per_patient_test['rmse_test'].std()}

        aggregated_losses_val = aggregated_losses_val.append(personalized_val, ignore_index=True)
        aggregated_losses_test = aggregated_losses_test.append(personalized_test, ignore_index=True)


        forecasts.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                    f'forecasts_{args.dataset_name}_{args.prediction_horizon}.csv'))
        aggregated_losses_test.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                    f'aggregated_losses_test_{args.dataset_name}_{args.prediction_horizon}.csv'))
        best_model_per_patient_test.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                    f'best_model_per_patient_test_{args.dataset_name}_{args.prediction_horizon}.csv'))
        losses_test.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                    f'losses_test_{args.dataset_name}_{args.prediction_horizon}.csv'))
        aggregated_losses_val.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                    f'aggregated_losses_val_{args.dataset_name}_{args.prediction_horizon}.csv'))
        best_model_per_patient_val.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                    f'best_model_per_patient_val_{args.dataset_name}_{args.prediction_horizon}.csv'))
        losses_val.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                    f'losses_val_{args.dataset_name}_{args.prediction_horizon}.csv'))





