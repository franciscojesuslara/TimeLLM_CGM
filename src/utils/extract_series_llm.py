import pandas as pd
import numpy as np
import utils.constants as cons
import os
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries
from sklearn.model_selection import train_test_split


# import psycopg2


# train_partitions=[]
#     test_partitions = []
#     scaler_partitions=[]

def load_data(number,dataset_name='vivli_mdi'):

    if dataset_name == 'vivli_mdi':
        cgm_values = np.load(os.path.join(cons.PATH_PROJECT_DATA, 'VIVLI', str(number) + '_values.pkl'),
                             allow_pickle=True)
        cgm_times = np.load(os.path.join(cons.PATH_PROJECT_DATA, 'VIVLI', str(number) + '_times.pkl'),
                            allow_pickle=True)

    elif dataset_name == 'vivli_pump':
        cgm_values = np.load(os.path.join(cons.PATH_PROJECT_DATA, 'VIVLI_pump', str(number)+ '_values.pkl'),
                             allow_pickle=True)
        cgm_times = np.load(os.path.join(cons.PATH_PROJECT_DATA, 'VIVLI_pump', str(number) + '_times.pkl'),
                            allow_pickle=True)
    return cgm_values, cgm_times


def extract_series_individual(cgm_values, cgm_times, patients_id, dataset_name='vivli'):
    dataframe_general = pd.DataFrame(columns=['unique_id','time','cgm'])
    for number, blocks in enumerate(cgm_values):
        if len(blocks) > 1:
            block_time = cgm_times[number]
            df = pd.DataFrame(np.asarray(blocks), columns=['cgm'])
            df.index = pd.DatetimeIndex(block_time)
            if dataset_name == 'palmas':
                df2 = df['cgm'].resample("15min", offset='1T').mean().interpolate().to_frame()
            else:
                df2 = df['cgm'].resample("15min", offset='1T').mean().interpolate().to_frame()
            df2['unique_id'] = '{}_{}'.format(patients_id, number)
            df2['time'] = df2.index
            dataframe_general = pd.concat([dataframe_general, df2])
    return dataframe_general


def extract_series_general(dataset_name='vivli_mdi', n_samples=None, prediction_horizon=4, ts_length=96):
    dataframe_general = pd.DataFrame(columns=['unique_id', 'time', 'cgm'])
    train = pd.DataFrame(columns=['unique_id', 'time', 'cgm'])
    test = pd.DataFrame(columns=['unique_id', 'time', 'cgm'])
    if dataset_name == 'vivli_mdi' or dataset_name == 'vivli_pump':
        if dataset_name == 'vivli_mdi':
            patients_id = np.load(os.path.join(cons.PATH_PROJECT_DATA, 'patients_id_mdi.npy'))
        else:
            patients_id = np.load(os.path.join(cons.PATH_PROJECT_DATA, 'patients_id_pump.npy'))
        for i in patients_id:
            cgm_values, cgm_times = load_data(i, dataset_name)
            df_individual = extract_series_individual(cgm_values, cgm_times, i, dataset_name)
            largest_window = df_individual['unique_id'].mode()[0]
            df_individual = df_individual[df_individual['unique_id'] == largest_window]
            dataframe_general = pd.concat([dataframe_general, df_individual])

    if n_samples is not None:
        largest_samples = dataframe_general['unique_id'].value_counts().nlargest(n_samples).index
        dataframe_general = dataframe_general[dataframe_general['unique_id'].isin(largest_samples)]

        for valor in largest_samples:
            df_valor = dataframe_general[dataframe_general['unique_id'] == valor].tail(prediction_horizon)
            test = pd.concat([test, df_valor])
            df_valor = dataframe_general[dataframe_general['unique_id'] == valor][:-prediction_horizon]
            train = pd.concat([train, df_valor])
    else:
        for valor in dataframe_general['unique_id'].unique():
            df_valor = dataframe_general[dataframe_general['unique_id'] == valor].tail(prediction_horizon)
            test = pd.concat([test, df_valor])
            df_valor = dataframe_general[dataframe_general['unique_id'] == valor][:-prediction_horizon]
            train = pd.concat([train, df_valor])

    return dataframe_general, train.reset_index(drop=True), test.reset_index(drop=True)
