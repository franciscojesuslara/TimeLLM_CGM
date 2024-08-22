import matplotlib.pyplot as plt
import utils.constants as cons
import os
import numpy as np


def plot_line(df, name):
    plt.figure(figsize=(10,5))
    plt.plot(df['Model'], df['PH=60'], color='blue',  marker='o', linewidth=2, label='PH=60')
    plt.plot(df['Model'], df['PH=90'], color='red',marker='o', linewidth=2, label='PH=90')
    plt.plot(df['Model'], df['PH=120'], color='green', marker='o', linewidth=2, label='PH=120')

    # Añadir títulos y etiquetas
    plt.xlabel('Model', fontsize=20)
    plt.ylabel('Number of Participants', fontsize=20)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(os.path.join(cons.PATH_PROJECT_REPORTS, 'merged', name + '.pdf'), bbox_inches='tight')


def plot_error_iso15197_acceptable_zone(y_true, y_pred):
    # Compute the measurement vector error
    # 95% of values must be +-15 mg/dL when < 100 mg/dL and +-15% when >=100 mg/dL
    diff_vector = y_true - y_pred

    # Plot difference chart
    # Establish the limits
    # Region UP
    region_x = np.array([0, 100, 500])
    region_up_y = np.array([15, 15, 75])
    # Region DOWN
    region_down_y = np.array([-15, -15, -75])
    # Plot boundaries in the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(True)
    ax.set_xlim([0, 550])
    ax.set_ylim([-90, 90])
    ax.plot(region_x, region_up_y, '--r')
    ax.plot(region_x, region_down_y, '--r')
    ax.set_xlabel('Glucose concentration (mg/dL)', fontsize=13)
    ax.set_ylabel('Difference between real and predicted glucose values', fontsize=13)

    # Plot error points and label
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.plot(y_true, diff_vector, 'b.')

    # Compute the percentage of samples out of limits
    total_samples = y_true.shape[0]
    # Compute the percentage of differences higher than +-15 mg/dL
    first_range = y_true < 100
    # Convert to absolute values
    a = np.abs(diff_vector[first_range])
    # Identify the samples out of the limits
    b = a > 15
    # Compute the percentage of samples out of the limits
    first_percent_out = np.sum(b) / total_samples * 100

    # Compute the percentage of differences higher than +-15%
    second_range = y_true >= 100
    # Convert to absolute values
    a = np.abs(diff_vector[second_range])
    # Extract the reference measurement numbers >100 mg/dL
    label_numbers = y_true[second_range]
    # Compute the percentage limit 15%
    label_percent = 0.15 * label_numbers
    # Identify higher values than 15%
    b = a > label_percent
    # Compute the total percentage of samples out of range in the second range
    second_percent_out = np.sum(b) / total_samples * 100

    # Compute the total percentage of samples out of range
    percent_out = first_percent_out + second_percent_out
    # Compute the total percentage of samples within the range
    percent_in = 100 - percent_out
    ax.set_title('% glucose values in recommended zone = {:.2f}'.format(percent_in))
    # plt.show()

    return percent_in


def grafica_total(train,test,pred_nbeats,contador):
    train.plot(label='Train')
    test.plot(label='Test')
    plt.ylabel('Glucose Value')
    pred_nbeats.plot(label=f'Predicted {contador}h')
    plt.show()


def plot_metric(df, dataset_name, prediction_horizon, metric):

    categories = df['model'].str.replace('^Auto', '', regex=True)
    values = df[f'{metric}_mean']
    errors = df[f'{metric}_std']
    plt.figure(figsize=(8, 6))
    plt.bar(categories, values, yerr=errors, capsize=5, color='cornflowerblue')

    # Añadir etiquetas y título
    plt.xlabel('Models', fontsize=14)
    plt.ylabel(metric.upper(), fontsize=14)
    plt.title(f'Participants using {dataset_name} , PH : {prediction_horizon} minutes', fontsize=18)
    # Aumentar el tamaño de las etiquetas de los ejes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=12)
    plt.ylim(0,50)

    plt.savefig(os.path.join(cons.PATH_PROJECT_REPORTS, f'{dataset_name}_{prediction_horizon}_{metric}.png'),
                dpi=300, bbox_inches='tight')


def plot_results(train, forecast,input, best_model, ids):

    if len(ids) > 1:
        fig, axes = plt.subplots(len(ids), 1, figsize=(13, 8 * len(ids)))
        plt.rcParams.update({
            'font.size': 20,  # Tamaño general de la fuente
            'axes.titlesize': 22,  # Tamaño del título de los ejes
            'axes.labelsize': 22,  # Tamaño de las etiquetas de los ejes
            'xtick.labelsize': 20,  # Tamaño de las etiquetas de las marcas en el eje x
            'ytick.labelsize': 20,  # Tamaño de las etiquetas de las marcas en el eje y
            'legend.fontsize': 20,  # Tamaño de la fuente de la leyenda
            'figure.titlesize': 20  # Tamaño del título de la figura
        })
        for i, unique_id in enumerate(ids):
            best_model_name = best_model[best_model['unique_id'] == unique_id]['model']
            train_values = train[train['unique_id'] == unique_id].tail(input)

            # Seleccionar los 4 valores de test correspondientes
            forecast_values = forecast[forecast['unique_id'] == unique_id]
            forecast_predict = forecast_values[best_model_name]
            forecast_real = forecast_values['cgm']

            # Plotear los valores de train
            axes[i].plot(train_values['cgm'].values, marker='o', color='blue',)

            # Plotear los valores de test en otro color
            axes[i].plot(range(len(train_values), len(train_values) + len(forecast_values)),
                         forecast_predict.values, marker='o', color='red', label=f'Test ({best_model_name.values[0]})')

            axes[i].plot(range(len(train_values), len(train_values) + len(forecast_values)),
                        forecast_real.values, marker='o', color='blue', label='original')

            # Añadir título y leyenda
            axes[i].set_title(f'Forecasting unique_id: {unique_id}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('CGM')
            axes[i].axvline(x=input, color='r', linestyle='--', label='Start prediction')
            axes[i].legend()
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        unique_id = ids[0]
        best_model_name = best_model[best_model['unique_id'] == unique_id]['model']
        train_values = train[train['unique_id'] == unique_id].tail(input)

        # Seleccionar los 4 valores de test correspondientes
        forecast_values = forecast[forecast['unique_id'] == unique_id]
        forecast_predict = forecast_values[best_model_name]
        forecast_real = forecast_values['cgm']

        # Plotear los valores de train
        plt.plot(train_values['cgm'].values, marker='o', color='blue', )

        # Plotear los valores de test en otro color
        plt.plot(range(len(train_values), len(train_values) + len(forecast_values)),
                     forecast_predict.values, marker='o', color='red', label=f'Predicted  ({best_model_name.values[0]})')

        real_values = forecast_real.values
        real_values = np.insert(real_values, 0, train_values.cgm.values[-1])

        plt.plot(range(len(train_values)-1, len(train_values) + len(forecast_values)),
                     real_values, marker='o', color='blue', label='Original')

        # Añadir título y leyenda
        # plt.title(f'Forecasting unique_id: {unique_id}')
        plt.xlabel('Time',fontsize=16)
        plt.ylabel('CGM',fontsize=16)
        # plt.axvline(x=input, color='r', linestyle='--', label='Start prediction')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12)

    plt.tight_layout()

    # Guardar la figura
    fig.savefig(os.path.join(cons.PATH_PROJECT_REPORTS,f'cgm_plot_{unique_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_test_estimated(test_fh, pred_fh, name, flag_save_figure):

    plt.style.use('tableau-colorblind10')

    # test_fh.pd_dataframe()['Glucose'].plot(label='test', marker='o', grid=True)
    # pred_fh.pd_dataframe()['Glucose'].plot(label='predicciones', marker='o', grid=True)
    # ax.set_xlabel('Fecha')
    # ax.set_ylabel('Nivel de glucosa')
    # plt.xticks(fontsize=14, orientation=90)
    # plt.yticks(fontsize=14)
    # plt.legend(fontsize=12)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(test_fh.pd_dataframe().index, test_fh.pd_dataframe()['glucose'],
            label='test', color='blue',
            marker='o'
            )
    ax.plot(pred_fh.pd_dataframe().index, pred_fh.pd_dataframe()['glucose'],
            label=f'predicción {contador}h', color='green',
            marker='o'
            )
    fig.autofmt_xdate()
    plt.legend()
    plt.tight_layout()

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_REPORTS, 'test_pred_{}.pdf'.format(name))))
        plt.close()
    else:
        plt.show()


def plot_scatter_real_pred(y_real, y_pred, title_figure, name, seed_value, flag_save_figure):
    x = np.arange(0, 1.05, 0.05)
    y = x
    fig, ax = plt.subplots()
    plt.scatter(y_real, y_pred, s=14)
    plt.plot(x, y, color='r', linestyle='solid')
    plt.xlabel('CVD risk by ST1RE')
    plt.ylabel('Predicted CVD risk')
    plt.title(title_figure)
    plt.grid(alpha=0.5, linestyle='--')

    # plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('equal')
    plt.tight_layout()

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if flag_save_figure:
        fig.savefig(str(Path.joinpath(consts.PATH_PROJECT_FIGURES, '{}_scatter_{}.pdf'.format(name, seed_value))))
        plt.close()
    else:
        plt.show()


def clarke_error_grid(ref_values, pred_values, name, title_string):
    assert len(ref_values) == len(pred_values), "Unequal number of values (reference: {}) (prediction: {}).".format(
        len(ref_values), len(pred_values)
    )
    if max(ref_values) > 400 or max(pred_values) > 400:
        print(
            "Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(
                max(ref_values), max(pred_values)
            )
        )
    if min(ref_values) < 0 or min(pred_values) < 0:
        print(
            "Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(
                min(ref_values), min(pred_values)
            )
        )
    fig, ax = plt.subplots(figsize=(8, 4))
    zones = [0] * 5
    colors = np.zeros(len(ref_values), dtype=int)

    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (
            pred_values[i] <= 1.2 * ref_values[i] and pred_values[i] >= 0.8 * ref_values[i]
        ):
            zones[0] += 1  # Zone A
            colors[i] = 0  # Assign color for Zone A
        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (
            ref_values[i] <= 70 and pred_values[i] >= 180
        ):
            zones[4] += 1  # Zone E
            colors[i] = 4  # Assign color for Zone E
        elif (
            (ref_values[i] >= 70 and ref_values[i] <= 290)
            and pred_values[i] >= ref_values[i] + 110
        ) or (
            (ref_values[i] >= 130 and ref_values[i] <= 180)
            and (pred_values[i] <= (7 / 5) * ref_values[i] - 182)
        ):
            zones[2] += 1  # Zone C
            colors[i] = 2  # Assign color for Zone C
        elif (
            ref_values[i] >= 240
            and (pred_values[i] >= 70 and pred_values[i] <= 180)
        ) or (
            ref_values[i] <= 175 / 3
            and pred_values[i] <= 180
            and pred_values[i] >= 70
        ) or (
            (ref_values[i] >= 175 / 3 and ref_values[i] <= 70)
            and pred_values[i] >= (6 / 5) * ref_values[i]
        ):
            zones[3] += 1  # Zone D
            colors[i] = 3  # Assign color for Zone D
        else:
            zones[1] += 1  # Zone B
            colors[i] = 1  # Assign color for Zone B

    ax.scatter(
        ref_values,
        pred_values,
        marker="o",
        c=colors,
        cmap="viridis",  # You can choose a different colormap if desired
        s=8
    )
    # ax.set_title(title_string)
    ax.set_xlabel("Real (mg/dL)")
    ax.set_ylabel("Prediction (mg/DL)")
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax.set_facecolor("white")
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 400])
    ax.set_aspect((400) / (400))
    ax.plot([0, 400], [0, 400], ":", c="black")
    ax.plot([0, 175 / 3], [70, 70], "-", c="black")
    ax.plot([175 / 3, 400 / 1.2], [70, 400], "-", c="black")
    ax.plot([70, 70], [84, 400], "-", c="black")
    ax.plot([0, 70], [180, 180], "-", c="black")
    ax.plot([70, 290], [180, 400], "-", c="black")
    ax.plot([70, 70], [0, 56], "-", c="black")
    ax.plot([70, 400], [56, 320], "-", c="black")
    ax.plot([180, 180], [0, 70], "-", c="black")
    ax.plot([180, 400], [70, 70], "-", c="black")
    ax.plot([240, 240], [70, 180], "-", c="black")
    ax.plot([240, 400], [180, 180], "-", c="black")
    ax.plot([130, 180], [0, 70], "-", c="black")
    ax.text(30, 15, "A", fontsize=15)
    ax.text(370, 260, "B", fontsize=15)
    ax.text(280, 370, "B", fontsize=15)
    ax.text(160, 370, "C", fontsize=15)
    ax.text(160, 15, "C", fontsize=15)
    ax.text(30, 140, "D", fontsize=15)
    ax.text(370, 120, "D", fontsize=15)
    ax.text(30, 370, "E", fontsize=15)
    ax.text(370, 15, "E", fontsize=15)

    plt.savefig(os.path.join(cons.PATH_PROJECT_REPORTS, name+'.pdf'), bbox_inches="tight")

    print(
        "There are: \n {0} values in zone A ({1}%), \n {2} values in zone B ({3}%), \n {4} values in zone C ({5}%), \n {6} values in zone D ({7}%), \n {8} values in zone E ({9}%)".format(
            zones[0],
            zones[0] / sum(zones),
            zones[1],
            zones[1] / sum(zones),
            zones[2],
            zones[2] / sum(zones),
            zones[3],
            zones[3] / sum(zones),
            zones[4],
            zones[4] / sum(zones),
        )
    )

    return fig, zones


def plot_train_test(train, test, preds_nbeats, uid, flag_save_figure):
    fig, ax = plt.subplots(figsize=(8, 4))
    train_nonorm = train
    test_nonorm = test
    preds_nonorm = preds_nbeats
    train_nonorm.pd_dataframe()['glucose'].plot(ax=ax, label='train')
    test_nonorm.pd_dataframe()['glucose'].plot(ax=ax, label='test')
    preds_nonorm.pd_dataframe()['glucose'].plot(ax=ax,label=f'predicción {contador}h')
    # plt.savefig('train_test_pred' + str(i) + '.pdf')
    plt.xlabel('Fecha',fontsize=13)
    plt.ylabel('Nivel de glucosa',fontsize=13)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)

    if flag_save_figure:
        plt.savefig(str(Path.joinpath(consts.PATH_PROJECT_REPORTS, 'train_test_pred_{}.pdf'.format(uid))))
        plt.close()
    else:
        plt.show()


def grafica(test, pred_TCN, contador):
    test.plot(label='Test (real)')
    plt.ylim(0, 200)
    pred_TCN.plot(label=f'Test (predicción) {contador}h')
    plt.show()
