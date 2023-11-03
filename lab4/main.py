from func import *
import pandas as pd
import statsmodels.api as sm

# Завантажимо дані з таблиці
data = pd.read_csv('lab4/data.csv')

Ni = data['Ni']
Pi = data['Pi']
Qi = data['Qi']

# Опишемо вашу однофакторну регресійну модель
x = calc_data(30, Ni, Qi)  # Ознака x
print(x)
X = sm.add_constant(x)  # Додамо стовпчик константи для b0
Y = calc_data(20, Ni, Pi)  # Ознака y
print(Y)

# Побудуємо регресійну модель за допомогою методу найменших квадратів
model = build_regresive_model(X, Y)

# Розвязок по формулі
xt = find_chtryh(x)
yt = find_chtryh(Y)

formula(x, Y, xt, yt)
# # Проведемо оцінку значимості параметрів рівняння регресії
# regression_eval(model, X)

# # Нанесемо на координатну площину кореляційне поле і теоретичну лінію парної регресії
# build_graph(model, x, X, Y)

# # Розрахуємо та оцінимо прогнозні значення
# check_pred(model)
