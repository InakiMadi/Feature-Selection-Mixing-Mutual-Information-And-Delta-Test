import numpy as np
import info_examples
import info
import csv



seed = 13
np.random.seed(seed)
n_rows = 1000
# X_orig,Y_orig = info_examples.genera_XY(n_rows,1,100, (lambda x: 4*x**2 + 3) ,[0],seed=seed)

with open('datos/exampleaaa.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';')

    writer.writerow(["Variables en X:",1, "4x^2 + 3","Sin ruido"])
    for i in range(1,4+1):
        # Y(x) = 4x^2 + 3
        X,Y = info_examples.genera_XY(n_rows,i,100, (lambda x: 4*x**2 + 3) ,[0], seed=seed)
        info_o = info.Informacion(X=X,Y=Y)
        mi = info_o.mutual_information()
        fmi = info_o.brute_force_MI()
        delta = info_o.TestDelta()
        fd = info_o.brute_force_Delta()
        print(i,"variables en X:")
        print("   ",mi, "  -  ",fmi)
        print("   ",delta, "  -  ",fd)

        writer.writerow([mi,fmi[1],fmi[1]-mi,fmi[0], "", delta,fd[1],delta-fd[1],fd[0]])

    print()
    writer.writerow([])
    #
    #
    writer.writerow(["Variables en X:",2, "sqrt(x1^2 + x2^2) * (x1^2+x2^2)", "Sin ruido"])
    for i in range(2,4+1):
        # Y(x1,x2) = sqrt(x1^2 + x2^2) * (x1^2+x2^2)
        X,Y = info_examples.genera_XY(n_rows,i,100,
                (lambda x,y: np.sqrt(x**2 + y**2)*(x**2 + y**2) ),
                [0,1], seed=seed)
        info_o = info.Informacion(X=X,Y=Y)
        mi = info_o.mutual_information()
        fmi = info_o.brute_force_MI()
        delta = info_o.TestDelta()
        fd = info_o.brute_force_Delta()
        print(i,"variables en X:")
        print("   ",mi, "  -  ",fmi)
        print("   ",delta, "  -  ",fd)
