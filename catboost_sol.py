import pandas as pd
import numpy as np
import seaborn as sns
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
#import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path



def lag(dataset, code_column, lag):
    df = np.zeros_like(dataset[code_column].values)
    df[lag:] = dataset[code_column].values[:-lag]

    return df


def train_model(data,RANDOM_SEED = 10, EARLY_STOPPING_ROUND = 10, SAMPLE_RATE = 0.4):
    X, y, datetime, reference = data_preprocessing(data)


    X_train, X_valid, y_train, y_valid = train_test_split(X, y.iloc[:, 1:], test_size=0.1, random_state=RANDOM_SEED,
                                                          stratify=X['Прогноз_вперед'])  # shuffle = False)
    optimized_regressor = CatBoostRegressor(learning_rate=0.1,
                                            depth=15,  # study.best_params['depth'],
                                            l2_leaf_reg=5,  # study.best_params['l2_leaf_reg'],
                                            min_child_samples=3,  # study.best_params['min_child_samples'],
                                            grow_policy='Depthwise',
                                            iterations=500,
                                            use_best_model=True,
                                            eval_metric='RMSE',
                                            od_type='iter',
                                            # od_wait=20,

                                            random_state=RANDOM_SEED,
                                            #  logging_level='Silent',
                                            #  task_type="GPU",
                                            verbose=True
                                            )
    optimized_regressor.fit(X_train.copy(), y_train.copy(),
                            eval_set=[(X_valid.copy(), y_valid.copy())],
                            early_stopping_rounds=EARLY_STOPPING_ROUND)
    return optimized_regressor


def data_preprocessing(well_n):
    df = pd.DataFrame()
    datasets = []
    for i in range(1, 91):
        well_n1 = well_n.iloc[1:, :].copy()

        well_n1['Дебит нефти_t'] = well_n1['Дебит нефти'].shift(i)
        # (well_n1['Дебит нефти'].shift(i))/well_n['Дебит нефти'].iloc[0]
        # well_n['Дебит нефти'].iloc[1:].diff(-i)
        # (well_n1['Дебит нефти'].shift(i))/well_n['Дебит нефти'].iloc[0]
        well_n1['Прогноз_вперед'] = i

        df = pd.concat([df, well_n1])
    df = df.dropna(subset=['Дебит нефти_t'])
    y = df[['datetime', 'Дебит нефти_t']]
    datetime = df['datetime']
    X = df[['Номер скважины', 'Давление забойное от Hд', 'Давление линейное (ТМ)', 'Дебит нефти', 'Прогноз_вперед',
            'Давление забойное', 'Дебит жидкости (ТМ)', 'Газовый фактор рабочий (ТМ)',
            'Дебит газа (ТМ)', 'Давление забойное от Pпр', 'Давление на входе ЭЦН (ТМ)']]
    X = X.interpolate(method='linear', axis=1).ffill().bfill()
    windows = [7, 14, 30, 90]  # ,180,360]
    i = 0
    cols = X.columns
    cols = cols[cols != 'Прогноз_вперед']
    statistic = ['mean', 'std']  # 'min','max', 'var']
    for win in windows:
        for stat in statistic:
            for f in cols:
                if f == 'Прогноз_вперед' or f == 'Номер скважины':
                    continue
                else:
                    X[f'{f}_window_{win}_stat_{stat}'] = X.rolling(window=win, min_periods=1)[f].agg(
                        stat).ffill().bfill()

    lag_l = ['Дебит нефти', 'Давление забойное', 'Дебит жидкости (ТМ)', 'Давление забойное от Hд',
             'Давление линейное (ТМ)', 'Дебит газа (ТМ)']
    lag1 = pd.DataFrame(lag(X, lag_l, 5), columns=['Lag51', 'Lag52', 'Lag53', 'Lag54', 'Lag55', 'Lag56'])
    lag2 = pd.DataFrame(lag(X, lag_l, 4), columns=['Lag41', 'Lag42', 'Lag43', 'Lag44', 'Lag45', 'Lag46'])
    lag3 = pd.DataFrame(lag(X, lag_l, 3), columns=['Lag31', 'Lag32', 'Lag33', 'Lag34', 'Lag35', 'Lag36'])
    lag4 = pd.DataFrame(lag(X, lag_l, 2), columns=['Lag21', 'Lag22', 'Lag23', 'Lag24', 'Lag25', 'Lag26'])
    lag5 = pd.DataFrame(lag(X, lag_l, 6), columns=['Lag61', 'Lag62', 'Lag63', 'Lag64', 'Lag65', 'Lag66'])
    lag6 = pd.DataFrame(lag(X, lag_l, 7), columns=['Lag71', 'Lag72', 'Lag73', 'Lag74', 'Lag75', 'Lag76'])
    lag7 = pd.DataFrame(lag(X, lag_l, 8), columns=['Lag81', 'Lag82', 'Lag83', 'Lag84', 'Lag85', 'Lag86'])
    lags = pd.concat([lag1, lag2, lag3, lag4, lag5, lag6, lag7], axis=1)
    X = pd.concat([X.reset_index(drop=True), lags], axis=1)
    return X, y, datetime, well_n['Дебит нефти'].iloc[0]


def main(data, forecast_horizon=90):
    wells = data['Номер скважины'].unique()
    train_path = Path().cwd().parent / 'data' / 'train.csv'
    data = pd.read_csv(train_path)
    X, y, datetime, reference = data_preprocessing(data)
    optimized_regressor = train_model(X,y)
    date_range = pd.date_range(start='1992-04-11', freq='1D', periods=forecast_horizon)
    all_forecasts = []

    for well in wells:
        print(f'Processing well № {well}')
        well_n = data.query(f'`Номер скважины` == {well}')


        # optimized_regressor = train_model(X,y)
        X['datetime'] = datetime.values
        last_date = well_n[well_n.datetime == well_n.datetime.max()]
        last_date_pred = X.loc[(X.datetime == last_date.datetime.values[0]) & (X['Номер скважины'] == well)]
        last_date_pred_f = last_date_pred.drop(columns='datetime').iloc[0]

        # make prediction
        predictions = []
        for i in range(0, forecast_horizon):
            last_date_pred_f['Прогноз_вперед'] = i
            pred = (optimized_regressor.predict(last_date_pred_f))
            predictions.append(pred)
            plt.plot(predictions)
            plt.show()

        forecast_df = pd.DataFrame({'datetime': date_range, 'forecast': predictions})
        forecast_df['Номер скважины'] = [well] * len(predictions)
        all_forecasts.append(forecast_df)

    all_forecasts = pd.concat(all_forecasts)
    print(f'Completed data processing. Forecast shape: {all_forecasts.shape}')
    print(f'Number of unique wells: {len(all_forecasts["Номер скважины"].unique())}')

    all_forecasts.to_csv('catbst_forecast.csv', index=False, encoding="utf-8")
    print('Saved forecast to "catbst_forecast.csv"')


def predict(data, well, optimized_regressor, max_date='1992-01-11'):
    preds = []
    preds_constant = []
    rmse_cum = 0
    rmse_cum_const = 0

    wells = data['Номер скважины'].unique()
    well_n = data.query(f'`Номер скважины` == {wells[well]}')

    last_date_pred = X.loc[(X.datetime == max_date) & (X['Номер скважины'] == well)]
    last_date_pred_f = last_date_pred.drop(columns='datetime').iloc[0]
    for i in range(1, 91):
        last_date_pred_f['Прогноз_вперед'] = i
        pred = (optimized_regressor.predict(last_date_pred_f))
        pred2 = last_date_pred_f['Дебит нефти']
        preds_constant.append(pred2)
        preds.append(pred)

    predictions2 = pd.DataFrame(data=preds, index=well_n[well_n.datetime > max_date].iloc[:90].index)
    predictions_const = pd.DataFrame(data=preds_constant, index=well_n[well_n.datetime > max_date].iloc[:90].index)

    rmse = mean_squared_error(well_n[well_n.datetime > max_date].iloc[:90]['Дебит нефти'], predictions2.values) / len(
        predictions_const)

    rmse_const = mean_squared_error(well_n[well_n.datetime > max_date].iloc[:90]['Дебит нефти'],
                                    predictions_const.values) / len(predictions_const)

    rmse_cum += rmse
    rmse_cum_const += rmse_const

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(well_n[well_n.datetime < max_date]['Дебит нефти'])
    ax.plot(well_n[well_n.datetime > max_date].iloc[:90]['Дебит нефти'], label='test')

    ax.plot(predictions2, label='model')
    ax.plot(predictions_const, label='constant')
    print('Cumulative RMSE for catboost:', rmse_cum)
    print('Cumulative RMSE for constant predictions:', rmse_cum_const)
    plt.legend()


