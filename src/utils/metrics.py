import matplotlib.pyplot as plt
import numpy as np
from darts import TimeSeries
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy.spatial import distance
import utils.constants as cons
import os
import pandas as pd
from datasetsforecast.losses import mse, mae, rmse, rmae

def JENSENSHANNON(serie1, serie2, serie3, serie4, serie5, serie6):
    plt.figure(figsize=(10, 6))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # Crea una lista para almacenar las distancias de Jensen-Shannon para cada serie
    distancias_serie2 = []
    distancias_serie3 = []
    distancias_serie4 = []
    distancias_serie5 = []
    distancias_serie6 = []

    # Calcula la longitud de la serie temporal
    longitud_serie = len(serie1)

    # Calcula el número de días
    num_dias = longitud_serie // 288


    # Calcula la distancia de Jensen-Shannon para cada día y cada serie
    for i in range(1, num_dias + 1):
        # Obtiene los subconjuntos de registros para cada serie
        subserie1 = serie1[:i * 288]
        subserie2 = serie2[:i * 288]
        subserie3 = serie3[:i * 288]
        subserie4 = serie4[:i * 288]
        subserie5 = serie5[:i * 288]
        subserie6 = serie6[:i * 288]

        # Calcula las distribuciones de probabilidad para cada subserie
        distribucion1 = np.array(subserie1) / np.sum(subserie1)
        distribucion2 = np.array(subserie2) / np.sum(subserie2)
        distribucion3 = np.array(subserie3) / np.sum(subserie3)
        distribucion4 = np.array(subserie4) / np.sum(subserie4)
        distribucion5 = np.array(subserie5) / np.sum(subserie5)
        distribucion6 = np.array(subserie6) / np.sum(subserie6)

        # Calcula la distancia de Jensen-Shannon para cada serie
        distancia2 = distance.jensenshannon(distribucion1, distribucion2)
        distancia3 = distance.jensenshannon(distribucion1, distribucion3)
        distancia4 = distance.jensenshannon(distribucion1, distribucion4)
        distancia5 = distance.jensenshannon(distribucion1, distribucion5)
        distancia6 = distance.jensenshannon(distribucion1, distribucion6)

        # Agrega las distancias a las listas correspondientes
        distancias_serie2.append(distancia2)
        distancias_serie3.append(distancia3)
        distancias_serie4.append(distancia4)
        distancias_serie5.append(distancia5)
        distancias_serie6.append(distancia6)

    # Crea una lista de días
    dias = range(1, num_dias + 1)

    # Grafica las distancias de Jensen-Shannon para todas las series
    plt.plot(dias, distancias_serie2, label='Paciente sintético 1')
    plt.plot(dias, distancias_serie3, label='Paciente sintético 2')
    plt.plot(dias, distancias_serie4, label='Paciente sintético 3')
    plt.plot(dias, distancias_serie5, label='Paciente sintético 4')
    plt.plot(dias, distancias_serie6, label='Paciente sintético 5')

    plt.scatter(dias, distancias_serie2, marker='x', color='blue')
    plt.scatter(dias, distancias_serie3, marker='x', color='orange')
    plt.scatter(dias, distancias_serie4, marker='x', color='green')
    plt.scatter(dias, distancias_serie5, marker='x', color='red')
    plt.scatter(dias, distancias_serie6, marker='x', color='purple')

    plt.xlabel('Día', fontsize=20)
    plt.ylim(0, 0.2)

    plt.ylabel('Divergencia de Jensen-Shannon',fontsize=20)
    plt.savefig("JENSENSHANNON_DGAN.pdf")

    plt.legend(fontsize=15)
    plt.savefig("JENSENSHANNON_DGAN.pdf")
    plt.show()


def compute_mrae(y_real, y_pred):
    epison = 1e-6
    difference = np.abs(y_real - y_pred) / (y_real + epison)
    return np.mean(difference)


def bgISOAcceptableZone(totalLabel, totalPred):
    # Compute the measurement vector error
    # 95% of values must be +-15 mg/dL when < 100 mg/dL and +-15% when >=100 mg/dL
    diffVector = totalLabel - totalPred

    # Plot difference chart
    # Establish the limits
    # Region UP
    region_x = np.array([0, 100, 500])
    regionUp_y = np.array([15, 15, 75])
    # Region DOWN
    regionDown_y = np.array([-15, -15, -75])
    # Plot boundaries in the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(True)
    ax.set_xlim([0, 550])
    ax.set_ylim([-90, 90])
    ax.plot(region_x, regionUp_y, '--r')
    ax.plot(region_x, regionDown_y, '--r')
    ax.set_xlabel('Concentración glucosa (mg/dl)', fontsize=13)
    ax.set_ylabel('Diferencia entre valores reales y estimados', fontsize=13)
    # Plot error points and label
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.plot(totalLabel, diffVector, 'b.')

    # Compute the percentage of samples out of limits
    # Compute the total number of samples
    totalSamples = totalLabel.shape[0]
    # Compute the percentage of differences higher than +-15 mg/dL
    firstRange = totalLabel < 100
    # Convert to absolute values
    a = np.abs(diffVector[firstRange])
    # Identify the samples out of the limits
    b = a > 15
    # Compute the percentage of samples out of the limits
    firstPercentOut = np.sum(b) / totalSamples * 100

    # Compute the percentage of differences higher than +-15%
    secondRange = totalLabel >= 100
    # Convert to absolute values
    a = np.abs(diffVector[secondRange])
    # Extract the reference measurement numbers >100 mg/dL
    labelNumbers = totalLabel[secondRange]
    # Compute the percentage limit 15%
    labelPercent = 0.15 * labelNumbers
    # Identify higher values than 15%
    b = a > labelPercent
    # Compute the total percentage of samples out of range in the second range
    secondPercentOut = np.sum(b) / totalSamples * 100

    # Compute the total percentage of samples out of range
    percentOut = firstPercentOut + secondPercentOut
    # Compute the total percentage of samples within the range
    percentIn = 100 - percentOut
    ax.set_title('% valores de glucosa en zona recomendada= {:.2f}'.format(percentIn))
    plt.show()

    return percentIn

def compute_losses(group):
    actual = group['cgm']
    predicted = group['predicted']
    mse_loss = mse(actual, predicted)
    mae_loss = mae(actual, predicted)
    rmse_loss = rmse(actual, predicted)
    return pd.Series({
        'mse': mse_loss,
        'mae': mae_loss,
        'rmse': rmse_loss,
    })
def evaluate_performance(df, columnas, df_val=None):
    df = df.reset_index()[columnas]

    melted_df = df.melt(id_vars=['unique_id', 'cgm'], var_name='model', value_name='predicted')
    losses = melted_df.groupby(['unique_id', 'model']).apply(compute_losses).reset_index()
    aggregated_losses = losses.groupby('model').agg(
        mse_mean=('mse', 'mean'),
        mse_std=('mse', 'std'),
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'),
        rmse_std=('rmse', 'std'),
    ).reset_index()
    if df_val is not None:

        df_val = df_val[df_val['unique_id'] != '1170_0']
        df_val = df_val[df_val['unique_id'] != '1271_14']
        df_val = df_val[df_val['unique_id'] != '1711_1']

        df_val = df_val[df_val['unique_id'] != '1536_1']
        df_val = df_val[df_val['unique_id'] != '870_1']
        df_val = df_val[df_val['unique_id'] != '1611_4']
        df_val = df_val[df_val['unique_id'] != '1622_1']
        best_model_per_patient = pd.merge(losses, df_val, on=['unique_id', 'model'],suffixes=('_test', '_val'),
                                          how='inner')
    else:
        best_model_per_patient = losses.loc[losses.groupby('unique_id')['mae'].idxmin()].reset_index(drop=True)
    print(best_model_per_patient['model'].value_counts())
    return losses, aggregated_losses, best_model_per_patient


def evaluate_performance_llm(df, columnas):
    df = df.reset_index()[columnas]
    df = df[df['unique_id'] != '1536_1']
    df = df[df['unique_id'] != '1170_0']
    df = df[df['unique_id'] != '1271_14']

    df = df[df['unique_id'] != '1711_1']
    df = df[df['unique_id'] != '870_1']
    df = df[df['unique_id'] != '1611_4']
    df = df[df['unique_id'] != '1622_1']

    melted_df = df.melt(id_vars=['unique_id', 'cgm'], var_name='model', value_name='predicted')
    losses = melted_df.groupby(['unique_id', 'model']).apply(compute_losses).reset_index()
    aggregated_losses = losses.groupby('model').agg(
        mse_mean=('mse', 'mean'),
        mse_std=('mse', 'std'),
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'),
        rmse_std=('rmse', 'std'),
    ).reset_index()
    best_model_per_patient = losses.loc[losses.groupby('unique_id')['mae'].idxmin()].reset_index(drop=True)
    print(best_model_per_patient['model'].value_counts())
    return losses, aggregated_losses, best_model_per_patient
def check_model_performance_llm(test, forecast):
    list_mae_values = []
    list_mse_values = []
    list_mrae_values = []
    list_mape_values = []

    print('xxxxx')
    print(forecast)
    print('xxxxx')

    for patient_id in test['unique_id'].unique():
        test_id = test[test['unique_id'] == patient_id]['cgm'].values
        forecast_id = forecast[forecast['unique_id'] == patient_id]['NBEATS'].values
        mae_val = mean_absolute_error(test_id, forecast_id)
        mse_val = mean_squared_error(test_id, forecast_id)
        mrae_val = compute_mrae(test_id, forecast_id)
        mape_val = mean_absolute_percentage_error(test_id, forecast_id)
        list_mae_values.append(mae_val)
        list_mse_values.append(mse_val)
        list_mrae_values.append(mrae_val)
        list_mape_values.append(mape_val)

    average_mae, std_mae = np.mean(list_mae_values), np.std(list_mae_values)
    average_mse, std_mse = np.mean(list_mse_values), np.std(list_mse_values)
    average_mrae, std_mrae = np.mean(list_mrae_values), np.std(list_mrae_values)
    average_mape, std_mape = np.mean(list_mape_values), np.std(list_mape_values)

    print('MAE: {} ({})'.format(average_mae, std_mae))
    print('MSE: {} ({})'.format(average_mse, std_mse))
    print('MRAE: {} ({})'.format(average_mrae, std_mrae))
    print('MAPE: {} ({})'.format(average_mape, std_mape))

    # print('MAE: ', average_mae, 'MSE:', average_mse, 'MRAE: ', average_mrae, 'MAPE: ', average_mape)

    return average_mae, average_mse, average_mrae, average_mape


def check_model_performance(model, series_list, predict_window, scaler):
    MAE_list=[]
    MSE_list = []
    MRAE_list = []
    MAPE_list = []
    list_train=[]
    list_test=[]
    for e in series_list:
        list_train.append(TimeSeries.from_series(e[:-predict_window]))
        test = scaler.inverse_transform(np.asarray(e[-predict_window:].values).reshape(-1, 1))
        list_test.append(test)
    preds = model.predict(series=list_train, n=predict_window, n_jobs=5)
    for j,e in enumerate(preds):
        preds=scaler.inverse_transform(np.asarray(e.data_array()).reshape(-1, 1))
        mae_val = mean_absolute_error(list_test[j], preds)
        mse_val = mean_squared_error(list_test[j], preds)
        mrae_val = compute_mrae(list_test[j], preds)
        mape_val = mean_absolute_percentage_error(list_test[j], preds)
        MAE_list.append(mae_val)
        MSE_list.append(mse_val)
        MRAE_list.append(mrae_val)
        MAPE_list.append(mape_val)
    MAE = np.mean(MAE_list)
    MSE = np.mean(MSE_list)
    MRAE = np.mean(MRAE_list)
    MAPE = np.mean(MAPE_list)
    print('MAE:', MAE,'MSE:',MSE ,'MRAE:',MRAE ,'MAPE:', MAPE)
    return MAE, MSE, MRAE, MAPE


def save_results_forecasting(database_name: str,
                             days: int,
                             model_name: str,
                             partition: int,
                             mae_value: float,
                             mse_value: float,
                             mrae_value: float,
                             mape_value: float
                             ):

    if os.path.isfile(os.path.join(cons.PATH_PROJECT_REPORTS, 'results_' + database_name + '.csv')):
        df = pd.read_csv(os.path.join(cons.PATH_PROJECT_REPORTS, 'results_' + database_name + '.csv'))
        results = pd.DataFrame([days, model_name, partition, mae_value, mse_value, mrae_value, mape_value]).T
        results.columns = ['days', 'model_name', 'partition', 'MAE', 'MSE', 'MRAE', 'MAPE']
        df_results = pd.concat([df, results], ignore_index=True, axis=0)
        df_results.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS, 'results_' + database_name + '.csv'), index=False)
    else:
        results = pd.DataFrame([days, model_name, partition, mae_value, mse_value, mrae_value, mape_value]).T
        results.columns = ['days', 'model_name', 'partition', 'MAE', 'MSE', 'MRAE', 'MAPE']
        results.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS, 'results_' + database_name + '.csv'), index=False)

