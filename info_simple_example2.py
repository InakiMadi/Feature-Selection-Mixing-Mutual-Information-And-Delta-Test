import numpy as np
import info
import numpy as np
"""
    X has 6 features (columns).
      x1, x2, x3, x4, x5, x6 have random numbers from 0 to 100.
    Y(x1,x2,x3,x4) = 10 sin(pi x1 x2) + 20(x3 - 0.5)^2 + 10x4
"""
# Fijamos el seed
seed = 13
np.random.seed(seed)

data = np.around(np.random.rand(6000,6)*100)
# Y function is number (20) in page 120 of https://sci-hub.tw/10.1016/S0893-6080(03)00169-2
# Y(X) = 10 sin(pi x1 x2) + 20(x3 - 0.5)^2 + 10x4
Y = []
for row in data:
    Y.append(2*row[0] + ((row[1] + row[2])**2)/2 - 3*row[3] )
    #Y.append( 10*np.sin(np.pi * row[0] * row[1]) + 20*(row[2] - 0.5)**2 + 10*row[3] )
Y = np.array(Y, np.float64)
data = np.array(data, np.int64)

info = info.Informacion(X=data,Y=Y)

print(data)
print(Y)

print()
print("MI:", info.mutual_information(log_base=2))
print("Delta:", info.TestDelta())
print()
print("MI without extra features:", info.mutual_information(data=data[...,:-2],log_base=2))
print("Delta without extra features:", info.TestDelta(data=data[...,:-2]))

# info_object = info.Informacion(X=data,last_feature_is_Y=True)
#
# print()
# print("INDEXES OF FEATURES SELECTED:")
# mi_f = info_object.filter_MI()
# print("MI:",mi_f)
# delta_f = info_object.filter_Delta()
# print("Delta:",delta_f)
# mix_f = info_object.filter_MI_Delta()
# print("Mix:",mix_f)
#
# print()
# print("LOGISTIC REGRESSION. Porcentaje bien clasificado:")
# orig = info_object.logistic_regression_lerner(seed=seed)[0]
# print("Original:",orig)
# mi_l = info_object.logistic_regression_lerner(cols_pos= mi_f[0] ,seed=seed)[0]
# print("MI:",mi_l)
# delta_l = info_object.logistic_regression_lerner(cols_pos= delta_f[0] ,seed=seed)[0]
# print("Delta:",delta_l)
# mix_l = info_object.logistic_regression_lerner(cols_pos= mix_f[0] ,seed=seed)[0]
# print("Mix:",mix_l)
