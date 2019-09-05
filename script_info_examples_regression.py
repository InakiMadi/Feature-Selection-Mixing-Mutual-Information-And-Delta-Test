import numpy as np
import info
import csv
from sklearn.preprocessing import StandardScaler

# Fijamos el seed
seed = 13
nums = [0.5,1,1.5,2,3,4,5,10]

# Leemos los datos
data = []
with open("datos/winequality-red.csv", "r") as filestream:
#with open("datos/winequality-white.csv", "r") as filestream:
    for line in filestream:
        chunks = line.split(";")
        data.append(np.array(chunks))
# We pass from string to float, and we drop the line naming the features.
data = np.array(data[1:], np.float64)
# Preprocesamiento
# Restamos por la media y dividimos por la desviación típica
data = StandardScaler(with_mean=True, with_std=True).fit_transform(data)
info_o = info.Informacion(X=data,last_feature_is_Y=True)


fmi = info_o.brute_force_MI()
fd = info_o.brute_force_Delta()
md = info_o.brute_force_MI_Delta()
md_2 = info_o.MI_Delta(arr_mi=fmi[2],arr_delta=fd[2])

print()
print("ALGORITMOS:")
print("  Brute Force MI:")
print(fmi[0],fmi[1])
print()
print("  Brute Force Delta:")
print(fd[0],fd[1])
print()
print("Brute Force MI-Delta:")
print(md[0],md[1])

print()
print()
print("LINEAR REGRESSION. E_out:")
orig = info_o.linear_regression_lerner(seed=seed)[0]
print("Original:",orig)
mi_l = info_o.linear_regression_lerner(cols_pos= fmi[0] ,seed=seed)[0]
print("Brute Force MI:",mi_l)
delta_l = info_o.linear_regression_lerner(cols_pos= fd[0] ,seed=seed)[0]
print("Brute Force Delta:",delta_l)
mix_l = info_o.linear_regression_lerner(cols_pos= md[0] ,seed=seed)[0]
print("Brute Force MI-Delta:",mix_l)

with open('datos/REGRESSION_RED.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow([fmi[0],fmi[1],"",fd[0],fd[1],"",md[0],md[1],"",orig,mi_l,delta_l,mix_l])
    for m in nums:
        for n in nums:
            md = info_o.MI_Delta(arr_mi=fmi[2],arr_delta=fd[2],alpha=m,beta=n)
            mix_l = info_o.linear_regression_lerner(cols_pos= md[0] ,seed=seed)[0]
            writer.writerow([m,n,mix_l])
            print(m,n,mix_l)





print()
print()
print()
print()

# Leemos los datos
data = []
with open("datos/winequality-white.csv", "r") as filestream:
#with open("datos/winequality-white.csv", "r") as filestream:
    for line in filestream:
        chunks = line.split(";")
        data.append(np.array(chunks))
# We pass from string to float, and we drop the line naming the features.
data = np.array(data[1:], np.float64)
# Preprocesamiento
# Restamos por la media y dividimos por la desviación típica
data = StandardScaler(with_mean=True, with_std=True).fit_transform(data)
info_o = info.Informacion(X=data,last_feature_is_Y=True)


fmi = info_o.brute_force_MI()
fd = info_o.brute_force_Delta()
md = info_o.brute_force_MI_Delta()
md_2 = info_o.MI_Delta(arr_mi=fmi[2],arr_delta=fd[2])

print()
print("ALGORITMOS:")
print("  Brute Force MI:")
print(fmi[0],fmi[1])
print()
print("  Brute Force Delta:")
print(fd[0],fd[1])
print()
print("Brute Force MI-Delta:")
print(md[0],md[1])

print()
print()
print("LINEAR REGRESSION. E_out:")
orig = info_o.linear_regression_lerner(seed=seed)[0]
print("Original:",orig)
mi_l = info_o.linear_regression_lerner(cols_pos= fmi[0] ,seed=seed)[0]
print("Brute Force MI:",mi_l)
delta_l = info_o.linear_regression_lerner(cols_pos= fd[0] ,seed=seed)[0]
print("Brute Force Delta:",delta_l)
mix_l = info_o.linear_regression_lerner(cols_pos= md[0] ,seed=seed)[0]
print("Brute Force MI-Delta:",mix_l)

with open('datos/REGRESSION_WHITE.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow([fmi[0],fmi[1],"",fd[0],fd[1],"",md[0],md[1],"",orig,mi_l,delta_l,mix_l])
    for m in nums:
        for n in nums:
            md = info_o.MI_Delta(arr_mi=fmi[2],arr_delta=fd[2],alpha=m,beta=n)
            mix_l = info_o.linear_regression_lerner(cols_pos= md[0] ,seed=seed)[0]
            writer.writerow([m,n,mix_l])
            print(m,n,mix_l)
