import matplotlib.pyplot as plt
import numpy as np

def plot_7(file_name,pic_name,y_label,title,start_cols,num_col1,label1,num_col2,label2,num_col3,label3,num_col4,label4,num_col5,label5,num_col6,label6,num_col7,label7):
    data = np.genfromtxt('datos/'+file_name+'.csv',delimiter=';', dtype = float)
    a = [ row[num_col1] for row in data ]
    b = [ row[num_col2] for row in data ]
    c = [ row[num_col3] for row in data ]
    d = [ row[num_col4] for row in data ]
    e = [ row[num_col5] for row in data ]
    f = [ row[num_col6] for row in data ]
    g = [ row[num_col7] for row in data ]

    x = list(range(start_cols, 11+1))

    plt.plot(x, a,'-x', label=label1)
    plt.plot(x, b,'-x',label=label2)
    plt.plot(x, c,'-x',label=label3)
    plt.plot(x, d,'-x', label=label4)
    plt.plot(x, e,'-x',label=label5)
    plt.plot(x, f,'-x',label=label6)
    plt.plot(x, g,'-x',label=label7)

    min = np.amin(a+b+c+d+e+f+g)
    max = np.amax(a+b+c+d+e+f+g)
    plt.axis([start_cols,11, -5,105])

    plt.legend()
    plt.xlabel("Número de variables en X")
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig('plots/'+pic_name+'.png')

    plt.close()

def plot_5(file_name,pic_name,y_label,title,start_cols,num_col1,label1,num_col2,label2,num_col3,label3,num_col4,label4,num_col5,label5):
    data = np.genfromtxt('datos/'+file_name+'.csv',delimiter=';', dtype = float)
    a = [ row[num_col1] for row in data ]
    b = [ row[num_col2] for row in data ]
    c = [ row[num_col3] for row in data ]
    d = [ row[num_col4] for row in data ]
    e = [ row[num_col5] for row in data ]

    x = list(range(start_cols, 11+1))

    plt.plot(x, a, '-^', color='k', label=label1)
    plt.plot(x, b, '-v', color='b', label=label2)
    plt.plot(x, c, '-x', color='r', label=label3)
    plt.plot(x, d, '-.', color='g', label=label4)
    plt.plot(x, e, '-+', color='m', label=label5)

    min = np.amin(a+b+c+d+e)
    max = np.amax(a+b+c+d+e)
    plt.axis([start_cols,11, min-0.2,max+0.2])

    plt.legend()
    plt.xlabel("Número de variables en X")
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig('plots/'+pic_name+'.png')

    plt.close()

def plot_4(file_name,pic_name,y_label,title,start_cols,num_col1,label1,num_col2,label2,num_col3,label3,num_col4,label4):
    data = np.genfromtxt('datos/'+file_name+'.csv',delimiter=';', dtype = float)
    a = [ row[num_col1] for row in data ]
    b = [ row[num_col2] for row in data ]
    c = [ row[num_col3] for row in data ]
    d = [ row[num_col4] for row in data ]

    x = list(range(start_cols, 11+1))

    plt.plot(x, a, '-^', color='k', label=label1)
    plt.plot(x, b, '-v', color='b', label=label2)
    plt.plot(x, c, '-x', color='r', label=label3)
    plt.plot(x, d, '-.', color='g', label=label4)

    min = np.amin(a+b+c+d)
    max = np.amax(a+b+c+d)
    plt.axis([start_cols,11, min-0.2,max+0.2])

    plt.legend()
    plt.xlabel("Número de variables en X")
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig('plots/'+pic_name+'.png')

    plt.close()

def plot_3(file_name,pic_name,y_label,title,start_cols,num_col1,label1,num_col2,label2,num_col3,label3):
    data = np.genfromtxt('datos/'+file_name+'.csv',delimiter=';', dtype = float)
    a = [ row[num_col1] for row in data ]
    b = [ row[num_col2] for row in data ]
    c = [ row[num_col3] for row in data ]

    x = list(range(start_cols, 11+1))

    plt.plot(x, a, '-^', color='k', label=label1)
    plt.plot(x, b, '-v', color='b', label=label2)
    plt.plot(x, c, '-x', color='r', label=label3)

    min = np.amin(a+b+c)
    max = np.amax(a+b+c)
    plt.axis([start_cols,11, min-0.2,max+0.2])

    plt.legend()
    plt.xlabel("Number of variables of X")
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig('plots/'+pic_name+'.png')

    plt.close()




# nombres = ['ejemploA',
#             'ejemploB',
#             'ejemploB_2_3',
#             'ejemploC',
#             'ejemploC_Uruido01',
#             'ejemploC_Uruido05',
#             'ejemploC_Nruido01',
#             'ejemploC_Nruido05',
#             'ejemploE',
#             'ejemploE_Uruido01',
#             'ejemploE_Uruido05',
#             'ejemploE_Nruido01',
#             'ejemploE_Nruido05',
#             'ejemploD',
#             ]
# titles = ['Y(x_0) = 4*x_0^2 + 3',
#             'Y(x1,x2) = sqrt(x1^2 + x2^2)*(x1^2+x2^2)',
#             'Y(x3,x4) = sqrt(x3^2 + x4^2)*(x3^2+x4^2)',
#             'Y(x1,x2) = sin(sqrt(x1^2 + x2^2) / x1^2 + x2^2)',
#             'Y(x1,x2) = sin(sqrt(x1^2 + x2^2) / x1^2 + x2^2)',
#             'Y(x1,x2) = sin(sqrt(x1^2 + x2^2) / x1^2 + x2^2)',
#             'Y(x1,x2) = sin(sqrt(x1^2 + x2^2) / x1^2 + x2^2)',
#             'Y(x1,x2) = sin(sqrt(x1^2 + x2^2) / x1^2 + x2^2)',
#             'Y(x1,x2,x3) = sin(sqrt(x^2 + y^2) / x^2 + y^2) + 20(x3-0.5)^2',
#             'Y(x1,x2,x3) = sin(sqrt(x^2 + y^2) / x^2 + y^2) + 20(x3-0.5)^2',
#             'Y(x1,x2,x3) = sin(sqrt(x^2 + y^2) / x^2 + y^2) + 20(x3-0.5)^2',
#             'Y(x1,x2,x3) = sin(sqrt(x^2 + y^2) / x^2 + y^2) + 20(x3-0.5)^2',
#             'Y(x1,x2,x3) = sin(sqrt(x^2 + y^2) / x^2 + y^2) + 20(x3-0.5)^2',
#             'Y(x1,...,x5) = 10sin(pi x1 x2) + 20(x3 - 0.5)^2 + 10x4 + 5x5'
#             ]
# s=2
# for i in range(len(nombres)):
    # plot_5(nombres[i], nombres[i]+'-all',
    #       '',titles[i],s,
    #       4,'MI of X',
    #       5,'Delta Test of X',
    #       7,'Brute force MI',
    #       10,'Brute force Delta Test',
    #       13,'Brute force MI-Delta'
    #       )
    # plot_5(nombres[i], nombres[i]+'-times',
    #       'Seconds',titles[i],s,
    #       15,'Time for MI',
    #       16,'Time for Brute Force MI',
    #       17,'Time for Delta Test',
    #       18,'Time for Brute Force DT'
    #       #19,'Time for Brute Force MI-DT'
    #       )
    # plot_3(nombres[i], nombres[i]+'-correct',
    #       'Percentage',titles[i],s,
    #       21,'BF MI',
    #       22,'BF DT',
    #       23,'BF MI-DT'
    #       )

plot_4('ejemploE_todos_algoritmos', 'ejemploE_todos_algoritmos'+'-times_closer',
      'Seconds','Y(x1,x2,x3) = sin(sqrt(x^2 + y^2) / x^2 + y^2) + 20(x3-0.5)^2',3,
      5,'Filter MI',
      6,'Filter DT',
      7,'Filter MI-DT',
      8,'mRMR MI'
      )
plot_7('ejemploE_todos_algoritmos', 'ejemploE_todos_algoritmos'+'-correct',
      'Percentage','Y(x1,x2,x3) = sin(sqrt(x^2 + y^2) / x^2 + y^2) + 20(x3-0.5)^2',3,
      10,'BF MI',
      11,'BF DT',
      12,'BF MI-DT',
      13,'Filter MI',
      14,'Filter DT',
      15,'Filter MI-DT',
      16,'mRMR MI'
      )
