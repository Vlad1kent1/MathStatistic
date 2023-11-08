from func import *
import pandas as pd

data = pd.read_csv('lab5/data.csv')

y = calc_data(1.5, data['y'])
x1 = calc_data(2.5, data['x1'])
x2 = calc_data(10, data['x2'])
visualization(y, x1, 'x1')
visualization(y, x2, 'x2')
print("y:", y, "\nx1:", x1, "\nx2:", x2)

X = np.column_stack((np.ones(len(x1)), x1, x2))
B = find_params(y, X)
U = find_param_u(y, X, B)

# 1. Аналіз залежності між факторами
depend_analysis(data)

# 2. Побудувати регресійну модель
model = build_regresive_model(x1, x2, y)

# 3. Перевірити модель на адекватність за F-критерієм Фішера
check_Fisher(calculate_F_value(X, y, B, U))

# 4. Оцінка значимості параметрів та інтервали довіри
evaluate_significance(y, X, B, U)

# 5. Довірчий інтервал для окремого значення залежної змінної
x_new = np.array([1, 50, 30])
print('Довірчий інтервал для прогнозованого значення y: ', predict_confidence_interval(x_new, X, y, B, U))
