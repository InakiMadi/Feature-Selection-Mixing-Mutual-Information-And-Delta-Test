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

orig = info_object.logistic_regression_lerner(seed=seed)[0]
im = info_object.logistic_regression_lerner(cols_pos= info_object.filter_MI() ,seed=seed)[0]
delta = info_object.logistic_regression_lerner(cols_pos= info_object.filter_Delta() ,seed=seed)[0]

print("LOGISTIC REGRESSION. Porcentaje bien clasificado:")
print("Original:",orig)
print("IM:",im)
print("Delta Test:",delta)
