import numpy as np
import info
from sklearn.preprocessing import StandardScaler
import csv
import time

"""
    X has 6 features (columns). Features 5 and 6 are not independent from the rest.
      x1, x2, x3, x4 have random numbers from 0 to 100.
      x5 = x2 + 0.8*r , with r a random number from 0 to 100.
      x6 = x1 + 0.3*x3.
    Y(x1,x2,x3,x4) = 10 sin(pi x1 x2) + 20(x3 - 0.5)^2 + 10x4
"""

def genera_X(n_rows,n_cols,max_x=1):
    X = np.random.rand(n_rows,n_cols)

    #Restamos por la media y dividimos por la desviación típica
    #X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    # Normalizamos
    #X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    # Discretizamos pasándolo al intervalo [0,100]
    #X = np.around(X*max_x)
    #X = np.array(X, np.int64)
    X = X*max_x
    X = np.array(X, np.float64)
    return X

def genera_Y(X,Y_function,Y_indexes,ruido=0,porc_ruido=0):
    Y = []
    for row in X:
        if len(Y_indexes) == 1:
            #Y.append( round( Y_function(row[Y_indexes[0]]) + ruido*porc_ruido ) )
            Y.append( Y_function(row[Y_indexes[0]]) + ruido*porc_ruido )
        elif len(Y_indexes) == 2:
            #Y.append( round( Y_function(row[Y_indexes[0]], row[Y_indexes[1]]) + ruido*porc_ruido ) )
            Y.append( Y_function(row[Y_indexes[0]], row[Y_indexes[1]]) + ruido*porc_ruido )
        elif len(Y_indexes) == 3:
            Y.append( Y_function(row[Y_indexes[0]], row[Y_indexes[1]], row[Y_indexes[2]]) + ruido*porc_ruido )
        elif len(Y_indexes) == 4:
            Y.append( Y_function(row[Y_indexes[0]], row[Y_indexes[1]], row[Y_indexes[2]], row[Y_indexes[3]]) + ruido*porc_ruido )
        elif len(Y_indexes) == 5:
            # Y(X) = 10 sin(pi x1 x2) + 20(x3 - 0.5)^2 + 10x4 +5x5
            Y.append( Y_function(row[Y_indexes[0]], row[Y_indexes[1]], row[Y_indexes[2]], row[Y_indexes[3]], row[Y_indexes[4]]) + ruido*porc_ruido )
    Y = np.array(Y, np.float64)
    return Y




# X_orig,Y_orig = info_examples.genera_XY(n_rows,1,100, (lambda x: 4*x**2 + 3) ,[0],seed=seed)

def ejemplo(name_file,n_rows,start_cols,finish_cols,Y_function,Y_indexes,max_x=1,ruido=0,porc_ruido=0,
            alpha=1,beta=1,
            seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = genera_X(n_rows,start_cols,max_x)
    Y = genera_Y(X, Y_function,Y_indexes,ruido,porc_ruido)
    info_o = info.Informacion(X=X,Y=Y)
    with open('datos/'+name_file+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')

        for i in range(start_cols,finish_cols+1):
            if i > start_cols:
                X_nuevo = genera_X(n_rows,1,max_x).reshape(-1)
                info_o.add_column_to_X( X_nuevo )

            t_s = time.time()
            mi = info_o.mutual_information()
            t_mi = time.time()-t_s
            t_s = time.time()
            fmi = info_o.brute_force_MI()
            t_fmi = time.time()-t_s
            t_s = time.time()
            delta = info_o.TestDelta()
            t_delta = time.time()-t_s
            t_s = time.time()
            fd = info_o.brute_force_Delta()
            t_fd = time.time()-t_s
            t_s = time.time()
            md = info_o.brute_force_MI_Delta(alpha=alpha,beta=beta)
            t_md = time.time()-t_s
            # print(i,"variables en X:")
            # print("   MI:",fmi[0], "  -  ",fmi[1], " - ",mi)
            # print("   Delta:",fd[0], "  -  ",fd[1], " - ",delta)
            # print("   Mix:",md[0], " - ", md[1])
            # print(t_mi,t_fmi,t_delta,t_fd,t_md)
            fmi_t = 0
            fd_t = 0
            md_t = 0
            aux = [i for i in fmi[0] if i in Y_indexes]
            fmi_t = (len(aux) / len(Y_indexes)) * 100
            aux = [i for i in fd[0] if i in Y_indexes]
            fd_t = (len(aux) / len(Y_indexes)) * 100
            aux = [i for i in md[0] if i in Y_indexes]
            md_t = (len(aux) / len(Y_indexes)) * 100
            # 0-6 , 7-14, 15-20, 21-24, 25
            writer.writerow([fmi[0],fd[0],md[0],"",mi,delta,"",
                            fmi[1],mi-fmi[1], "", fd[1],delta-fd[1],"",md[1],"",
                            t_mi,t_fmi,t_delta,t_fd,t_md,"",
                            fmi_t,fd_t,md_t,"",
                            Y_indexes])

def ejemplo_todos_algoritmos(name_file,n_rows,start_cols,finish_cols,Y_function,Y_indexes,max_x=1,ruido=0,porc_ruido=0,
            alpha=1,beta=1,
            seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = genera_X(n_rows,start_cols,max_x)
    Y = genera_Y(X, Y_function,Y_indexes,ruido,porc_ruido)
    info_o = info.Informacion(X=X,Y=Y)
    with open('datos/'+name_file+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')

        for i in range(start_cols,finish_cols+1):
            if i > start_cols:
                X_nuevo = genera_X(n_rows,1,max_x).reshape(-1)
                info_o.add_column_to_X( X_nuevo )

            t_s = time.time()
            mi = info_o.mutual_information()
            t_mi = time.time()-t_s
            t_s = time.time()
            fmi = info_o.brute_force_MI()
            t_fmi = time.time()-t_s
            t_s = time.time()
            delta = info_o.TestDelta()
            t_delta = time.time()-t_s
            t_s = time.time()
            fd = info_o.brute_force_Delta()
            t_fd = time.time()-t_s
            t_s = time.time()
            md = info_o.brute_force_MI_Delta(alpha=alpha,beta=beta)
            t_md = time.time()-t_s
            t_s = time.time()
            filt_mi = info_o.filter_MI()
            t_filt_mi = time.time()-t_s
            t_s = time.time()
            filt_d = info_o.filter_Delta()
            t_filt_d = time.time()-t_s
            t_s = time.time()
            filt_md = info_o.filter_MI_Delta()
            t_filt_md = time.time()-t_s
            t_s = time.time()
            mrmr_mi = info_o.mRMR_MI()
            t_mrmr_mi = time.time()-t_s
            # print(i,"variables en X:")
            # print("   MI:",fmi[0], "  -  ",fmi[1], " - ",mi)
            # print("   Delta:",fd[0], "  -  ",fd[1], " - ",delta)
            # print("   Mix:",md[0], " - ", md[1])
            # print(t_mi,t_fmi,t_delta,t_fd,t_md)
            fmi_t = 0
            fd_t = 0
            md_t = 0
            filt_mi_t = 0
            filt_d_t = 0
            filt_md_t = 0
            mrmr_mi_t = 0

            aux = [i for i in fmi[0] if i in Y_indexes]
            fmi_t = (len(aux) / len(Y_indexes)) * 100
            aux = [i for i in fd[0] if i in Y_indexes]
            fd_t = (len(aux) / len(Y_indexes)) * 100
            aux = [i for i in md[0] if i in Y_indexes]
            md_t = (len(aux) / len(Y_indexes)) * 100
            aux = [i for i in filt_mi[0] if i in Y_indexes]
            filt_mi_t = (len(aux) / len(Y_indexes)) * 100
            aux = [i for i in filt_d[0] if i in Y_indexes]
            filt_d_t = (len(aux) / len(Y_indexes)) * 100
            aux = [i for i in filt_md if i in Y_indexes]
            filt_md_t = (len(aux) / len(Y_indexes)) * 100
            aux = [i for i in mrmr_mi if i in Y_indexes]
            mrmr_mi_t = (len(aux) / len(Y_indexes)) * 100

            # 0-4 5-8 9 , 10-16 17 , 18
            writer.writerow([t_mi,t_fmi,t_delta,t_fd,t_md, t_filt_mi, t_filt_d, t_filt_md, t_mrmr_mi,"",
                            fmi_t, fd_t, md_t, filt_mi_t, filt_d_t, filt_md_t, mrmr_mi_t,"",
                            Y_indexes])


def ejemplo_MIDT(name_file,n_rows,start_cols,finish_cols,Y_function,Y_indexes,max_x=1,ruido=0,porc_ruido=0,
            seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = genera_X(n_rows,start_cols,max_x)
    Y = genera_Y(X, Y_function,Y_indexes,ruido,porc_ruido)
    nums = [0.5,1,1.5,2,3,4,5,10]

    with open('datos/'+name_file+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        md_t = 0
        info_o = info.Informacion(X=X,Y=Y)
        for i in range(start_cols,finish_cols+1):
            if i > start_cols:
                X_nuevo = genera_X(n_rows,1,max_x).reshape(-1)
                info_o.add_column_to_X( X_nuevo )
            fmi = info_o.brute_force_MI()
            fd = info_o.brute_force_Delta()

            for m in nums:
                for n in nums:
                    md = info_o.MI_Delta(arr_mi=fmi[2],arr_delta=fd[2],alpha=m,beta=n)
                    # print(i,"variables en X:")
                    # print("   MI:",fmi[0], "  -  ",fmi[1], " - ",mi)
                    # print("   Delta:",fd[0], "  -  ",fd[1], " - ",delta)
                    # print("   Mix:",md[0], " - ", md[1])
                    # print(t_mi,t_fmi,t_delta,t_fd,t_md)
                    aux = [i for i in md[0] if i in Y_indexes]
                    md_t = (len(aux) / len(Y_indexes)) * 100

                    writer.writerow([m,n,md_t])
