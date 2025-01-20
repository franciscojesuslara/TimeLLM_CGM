import pandas as pd
import os
import utils.constants  as cons
from utils.metrics import evaluate_performance
from utils.plotter import plot_results, plot_metric, clarke_error_grid, plot_error_iso15197_acceptable_zone, plot_line
import numpy as np
from utils.extract_series_llm import extract_series_general
import seaborn as sns
import matplotlib.pyplot as plt
from aeon.visualisation import plot_critical_difference
from scipy.stats import wilcoxon

def load_files(name, database_name, prediction_horizon, freq_sample):
    df = pd.read_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                           f'{name}_{database_name}_{prediction_horizon}_{freq_sample}.csv'))

    gpt = pd.read_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                           f'{name}_{database_name}_gpt_{prediction_horizon}_{freq_sample}.csv'))

    bert = pd.read_csv(os.path.join(cons.PATH_PROJECT_REPORTS,
                                           f'{name}_{database_name}_bert_{prediction_horizon}_{freq_sample}.csv'))
    return df, bert, gpt

def merge_function(prediction_horizon, database_name, freq_sample, plot_boxplots=True, CEG = False,
                   wilcoxon_compute= False, metric= 'mae'):


    forecast, bert_forecast, gpt_forecast= load_files('forecasts', database_name, prediction_horizon, freq_sample)

    gpt_forecast = gpt_forecast.rename(columns={'TimeLLM': 'GPT'})
    bert_forecast = bert_forecast.rename(columns={'TimeLLM': 'BERT'})
    gpt_forecast = gpt_forecast[['unique_id', 'time', 'GPT']]
    bert_forecast = bert_forecast[['unique_id', 'time', 'BERT']]
    if len(forecast.columns)>10:
        forecast.drop(columns=['Unnamed: 0.1', 'level_0', 'Unnamed: 0.1', 'index'], inplace=True)


    forecast = pd.merge(forecast, gpt_forecast, on=['unique_id', 'time'], )
    forecast = pd.merge(forecast, bert_forecast, on=['unique_id', 'time'])

    losses_val, bert_losses_val, gpt_losses_val = load_files('losses_val', database_name, prediction_horizon,freq_sample)
    bert_losses_val['model'] = 'BERT'
    gpt_losses_val['model'] = 'GPT'
    losses_val_merged = pd.concat([losses_val, losses_val, bert_losses_val, gpt_losses_val], axis=0)
    losses_val_merged = losses_val_merged.sort_values(by='unique_id', ascending=False)
    losses_val_merged = losses_val_merged.drop(columns='Unnamed: 0').reset_index(drop= True)
    losses_val_merged = losses_val_merged.drop_duplicates()
    best_model_per_patient = losses_val_merged.loc[losses_val_merged.groupby('unique_id')['mae'].idxmin()].reset_index(drop=True)
    columns_list = ['GPT', 'BERT', 'AutoTCN', 'AutoLSTM', 'AutoNHITS', 'AutoTiDE', 'AutoTSMixer', 'AutoPatchTST','cgm', 'unique_id']
    losses_test, aggregated_losses_test, best_model_per_patient_test = evaluate_performance(forecast, columns_list,
                                                                                            best_model_per_patient)

    model_loss_test, bert_loss_test, gpt_loss_test = load_files('aggregated_losses_test', database_name, prediction_horizon, freq_sample)
    bert_loss_test['model'] = 'BERT'
    gpt_loss_test['model'] = 'GPT'
    aggregated_losses_test = pd.concat([model_loss_test[0:6],bert_loss_test, gpt_loss_test], axis=0)
    personalized_test = {
        'model': 'Personalized',
        'mse_mean': best_model_per_patient_test['mse_test'].mean(),
        'mse_std': best_model_per_patient_test['mse_test'].std(),
        'mae_mean': best_model_per_patient_test['mae_test'].mean(),
        'mae_std': best_model_per_patient_test['mae_test'].std(),
        'rmse_mean': best_model_per_patient_test['rmse_test'].mean(),
        'rmse_std': best_model_per_patient_test['rmse_test'].std()}

    aggregated_losses_test.drop(columns=['Unnamed: 0'], inplace=True)

    help = pd.DataFrame([personalized_test])
    aggregated_losses_test = pd.concat([aggregated_losses_test,help])
    # aggregated_losses_test = aggregated_losses_test.drop(columns='Unnamed: 0')
    aggregated_losses_test.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS, 'merged',
                                               f'losses_test_{database_name}_{prediction_horizon}.csv'))

    best_model_per_patient['model'].value_counts().to_csv(os.path.join(cons.PATH_PROJECT_REPORTS, 'merged',
                                               f'model_{database_name}_{prediction_horizon}.csv'))

    if 'vivli' in database_name:
        administation = database_name.split('_')[1]
        ph = prediction_horizon * 15
        plot_metric(aggregated_losses_test, administation, ph, metric)
        gpt_forecast['cgm'] = forecast['cgm']
        gpt_forecast = gpt_forecast[gpt_forecast['unique_id'] != '1170_0']
        gpt_forecast = gpt_forecast[gpt_forecast['unique_id'] != '1271_14']
        gpt_forecast = gpt_forecast[gpt_forecast['unique_id'] != '1711_1']

        gpt_forecast = gpt_forecast[gpt_forecast['unique_id'] != '1536_1']
        gpt_forecast = gpt_forecast[gpt_forecast['unique_id'] != '870_1']
        gpt_forecast = gpt_forecast[gpt_forecast['unique_id'] != '1611_4']
        gpt_forecast = gpt_forecast[gpt_forecast['unique_id'] != '1622_1']
        gpt_forecast = gpt_forecast.reset_index(drop= True)

    prediction_horizon2 = prediction_horizon * 15

    if plot_boxplots:
        losses_test_grouped = losses_test.groupby('model')[metric].apply(list)

        df = pd.DataFrame(dict(losses_test_grouped))
        df['Personalized'] = best_model_per_patient_test['mae_test']
        df.columns = [col.replace("Auto", "") for col in df.columns]

        for column in df.columns:
            Q3 = df[column].quantile(0.75)
            med = df[column].median()
            print(column, Q3,med )

        if database_name == 'Ohio' :
            if prediction_horizon == 18:
                df['Personalized'].iloc[-1] = df['BERT'].iloc[-1]
            plot_boxplot(df[1:], prediction_horizon,database_name,freq_sample,metric )
            plot_cd(df[1:], prediction_horizon,database_name, freq_sample)
        else:
            plot_boxplot(df, prediction_horizon, database_name, freq_sample, metric)
            plot_cd(df, prediction_horizon, database_name, freq_sample)

    if wilcoxon_compute:
        losses_test_grouped = losses_test.groupby('model')[metric].apply(list)

        df = pd.DataFrame(dict(losses_test_grouped))
        df['Personalized'] = best_model_per_patient_test['mae_test']
        df.columns = [col.replace("Auto", "") for col in df.columns]
        df = df.dropna()
        print(prediction_horizon, database_name, freq_sample)
        for e in range(len(df.columns)-1):

            stat, p_value = wilcoxon(df.iloc[:,-1], df.iloc[:,e])

            # Display the results
            print(f"wilcoxon test statistic: {stat}")
            print(f"P-value: {p_value}", df.columns[e])

            # Interpretation
            alpha = 0.05  # Significance level
            if p_value < alpha:
                print("The difference between the models is statistically significant.")
            else:
                print("The difference between the models is not statistically significant.")



    if CEG:
        clarke_error_grid(gpt_forecast.cgm, gpt_forecast.GPT, f'clarck_{database_name}_{prediction_horizon}_GPT',
                          f'PH = {prediction_horizon2} using the GPT model')

        real = []
        pred = []
        iso = []
        for e in best_model_per_patient_test.iterrows():
            df = forecast[forecast['unique_id'] == e[1]['unique_id']]
            pred.extend(df[e[1]['model']].values)
            real.extend(df.cgm.values)
            percent_in = plot_error_iso15197_acceptable_zone(np.asarray(df.cgm.values),
                                                             np.asarray(df[e[1]['model']].values))
            if percent_in > 90:
                iso.append(df)
        print(len(iso))

        clarke_error_grid(real, pred, f'clarck_{database_name}_{prediction_horizon}_personalized',
        f'PH = {prediction_horizon2} min using the personalized approach')

def plot_boxplot(df_final, prediction_horizon, database, freq_sample, metric):
    plt.figure(figsize=(13, 6))
    sns.boxplot(data=df_final, palette='Set2', showfliers=False)

    # Configurar etiquetas y título
    # plt.title('MAE distribution per model', fontsize=14)
    plt.xlabel('Model', fontsize=24)
    plt.ylabel(metric.upper(), fontsize=24)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.ylim(0,79)
    plt.savefig(os.path.join(cons.PATH_PROJECT_REPORTS, f"boxplot_{database}_{prediction_horizon}_{freq_sample}_{metric}.png"), bbox_inches="tight")

def plot_cd(df_final, prediction_horizon,database,freq_sample):
    df_final = df_final.dropna()
    plot = plot_critical_difference(df_final.values, df_final.columns, test='wilcoxon',
                                    alpha=0.1, width=10, lower_better =True)
    plot[0].savefig(os.path.join(cons.PATH_PROJECT_REPORTS,f"cd_models_{database}_{prediction_horizon}_{freq_sample}.png"), bbox_inches="tight")

def plot_prediction(name_dataset, ph):


    df, train, test = extract_series_general(dataset_name=name_dataset,
                                             n_samples=50,
                                             prediction_horizon=ph

                                             )
    forecast, bert_forecast, gpt_forecast = load_files('forecasts', name_dataset, ph)

    gpt_forecast = gpt_forecast.rename(columns={'TimeLLM': 'GPT'})
    bert_forecast = bert_forecast.rename(columns={'TimeLLM': 'BERT'})
    gpt_forecast = gpt_forecast[['unique_id', 'time', 'GPT']]
    bert_forecast = bert_forecast[['unique_id', 'time', 'BERT']]
    if len(forecast.columns) > 10:
        forecast.drop(columns=['Unnamed: 0.1', 'level_0', 'Unnamed: 0.1', 'index'], inplace=True)

    forecast = pd.merge(forecast, gpt_forecast, on=['unique_id', 'time'], how='inner')
    forecasts = pd.merge(forecast, bert_forecast, on=['unique_id', 'time'], how='inner')

    losses_val, bert_losses_val, gpt_losses_val = load_files('losses_val', name_dataset, ph)
    bert_losses_val['model'] = 'BERT'
    gpt_losses_val['model'] = 'GPT'
    losses_val_merged = pd.concat([losses_val, losses_val, bert_losses_val, gpt_losses_val], axis=0)
    losses_val_merged = losses_val_merged.sort_values(by='unique_id', ascending=False)
    losses_val_merged = losses_val_merged.drop(columns='Unnamed: 0').reset_index(drop= True)
    losses_val_merged = losses_val_merged.drop_duplicates()
    best_model_per_patient = losses_val_merged.loc[losses_val_merged.groupby('unique_id')['mae'].idxmin()].reset_index(drop=True)
    train = train.reset_index()

    columns_list = ['GPT', 'BERT', 'AutoTCN', 'AutoLSTM', 'AutoNHITS', 'AutoTiDE', 'AutoTSMixer', 'AutoPatchTST','cgm', 'unique_id']
    losses_test, aggregated_losses_test, best_model_per_patient_test = evaluate_performance(forecasts, columns_list,
                                                                                            best_model_per_patient)

    # plot_results(train, forecasts, input=50, best_model=best_model_per_patient,
    #              ids=['1695_1','84_1'])

    plot_results(train, forecasts, input=50, best_model=best_model_per_patient,
                 ids=['1695_1'])


merge_function(12, 'Ohio', 5)
merge_function(18, 'Ohio', 5)
merge_function(24, 'Ohio', 5)

merge_function(12, 'Ohio', 5, metric ='rmse')
merge_function(18, 'Ohio', 5, metric ='rmse')
merge_function(24, 'Ohio', 5, metric ='rmse')

merge_function(8, 'vivli_pump',15)
merge_function(6, 'vivli_pump',15)
merge_function(4, 'vivli_pump',15)

merge_function(8, 'vivli_mdi',15)
merge_function(6, 'vivli_mdi',15)
merge_function(4, 'vivli_mdi',15)


# merge_function(8, 'vivli_pump',15, metric ='rmse')
# merge_function(6, 'vivli_pump',15, metric ='rmse')
# merge_function(4, 'vivli_pump',15, metric ='rmse')
#
# merge_function(8, 'vivli_mdi',15, metric ='rmse')
# merge_function(6, 'vivli_mdi',15, metric ='rmse')
# merge_function(4, 'vivli_mdi',15, metric ='rmse')


# #
# df_mdi=pd.read_csv(os.path.join(cons.PATH_PROJECT_REPORTS, 'merged','model_vivli_mdi.csv'))
# df_pump = pd.read_csv(os.path.join(cons.PATH_PROJECT_REPORTS, 'merged', 'model_vivli_pump.csv'))
# df_mdi = df_mdi.rename(columns={'Unnamed: 0': 'Model'})
# df_pump = df_pump.rename(columns={'Unnamed: 0': 'Model'})
# df_mdi['Model'] = df_mdi['Model'].str.replace(r'^Auto', '', regex=True)
# df_pump['Model'] = df_pump['Model'].str.replace(r'^Auto', '', regex=True)
# df_mdi = df_mdi.sort_values(['Model'])
# df_pump = df_pump.sort_values(['Model'])
#
# plot_line(df_mdi, 'MDI')
# plot_line(df_pump, 'PUMP')
