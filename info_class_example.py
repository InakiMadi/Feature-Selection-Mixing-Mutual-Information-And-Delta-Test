import numpy as np
import info

# Fijamos el seed
seed = 13

# Leemos los datos
data = []
with open("datos/winequality-red.csv", "r") as filestream:
    for line in filestream:
        chunks = line.split(";")
        data.append(np.array(chunks))
# We pass from string to float, and we drop the line naming the features.
data = np.array(data[1:], np.float64)
# To simplify, we use int (rounding the numbers).
data = np.around(data)
data = data.astype(int)

info_object = info.Informacion(X=data,last_feature_is_Y=True)

print()
print("INDEXES OF FEATURES SELECTED:")
mi_f = info_object.filter_MI()
print("MI:",mi_f)
delta_f = info_object.filter_Delta()
print("Delta:",delta_f)
mix_f = info_object.filter_MI_Delta()
print("Mix:",mix_f)

print()
print("LOGISTIC REGRESSION. Porcentaje bien clasificado:")
orig = info_object.logistic_regression_lerner(seed=seed)[0]
print("Original:",orig)
mi_l = info_object.logistic_regression_lerner(cols_pos= mi_f[0] ,seed=seed)[0]
print("MI:",mi_l)
delta_l = info_object.logistic_regression_lerner(cols_pos= delta_f[0] ,seed=seed)[0]
print("Delta:",delta_l)
mix_l = info_object.logistic_regression_lerner(cols_pos= mix_f[0] ,seed=seed)[0]
print("Mix:",mix_l)
