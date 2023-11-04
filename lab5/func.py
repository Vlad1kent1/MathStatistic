import random
import numpy as np
import pandas as pd
import statsmodels.api as sm

def calc_data(a, i):
    N = random.uniform(0, 1)
    data = i + a * (N - 1/2)
    return list(data)

def depend_analysis(data):
    correlation_matrix = data.corr()
    print("\nКореляційна матриця:")
    print(correlation_matrix)

def build_regresive_model(x1, x2, y):
    X = sm.add_constant(pd.DataFrame({'x1': x1, 'x2': x2}))
    model = sm.OLS(y, X).fit()
    print("Регресійна модель:")
    print(model.summary())
    return model

def check_Fisher(model):
    F_statistic = model.fvalue
    F_p_value = model.f_pvalue
    alpha = 0.05
    if F_p_value < alpha:
        print("\nМодель є адекватною за F-критерієм Фішера")
    else:
        print("\nМодель не є адекватною за F-критерієм Фішера")

def estimation_conf_intervals(model):
    b0, b1, b2 = model.params['const'], model.params['x1'], model.params['x2']
    print("\nParameters:")
    print("b0:", b0)
    print("b1:", b1)
    print("b2:", b2)
    conf_int_b1 = model.conf_int().loc['x1']
    conf_int_b2 = model.conf_int().loc['x2']
    print("\nConfidence Intervals:")
    print(conf_int_b1)
    print(conf_int_b2)

def conf_interval(model):
    new_x1 = float(input("\nВведіть x1: "))
    new_x2 = float(input("Введіть x2: "))
    b0, b1, b2 = model.params['const'], model.params['x1'], model.params['x2']
    x_pred = np.array([1, new_x1, new_x2])  # Створення масиву зі значеннями, включаючи константу
    y_pred = np.dot(x_pred, [b0, b1, b2])
    # Обчислення довірчого інтервалу
    x_pred = x_pred.reshape(1, -1)  # Зміна форми для правильного обчислення інтервалу
    y_pred_ci = model.get_prediction(x_pred).conf_int()
    return y_pred_ci

def interpretation_params(model, y_pred_ci):
    print("\nParameter Interpretation:")
    print("b0 (Intercept):", model.params['const'])
    print("b1 (x1):", model.params['x1'])
    print("b2 (x2):", model.params['x2'])
    print("Довірчий інтервал для b1:", model.conf_int().loc['x1'])
    print("Довірчий інтервал для b2:", model.conf_int().loc['x2'])
    print("Довірчий інтервал для прогнозованого значення y:", y_pred_ci)
