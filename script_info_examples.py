import numpy as np
import info_examples

seed = 13
n_samples = 1000


# info_examples.ejemplo_MIDT('ejemploC_MIDT', n_samples, 2,11,
#                       (lambda x,y: np.sin(np.sqrt(x**2 + y**2))/(x**2 + y**2) ),[0,1],
#                       seed=seed)
# info_examples.ejemplo('ejemploA', n_samples, 1,11,
#                       (lambda x: 4*x**2 + 3),[0],
#                       seed=seed)
# info_examples.ejemplo('ejemploB', n_samples, 2,11,
#                       (lambda x,y: np.sqrt(x**2 + y**2)*(x**2 + y**2) ),[0,1],
#                       seed=seed)
# info_examples.ejemplo('ejemploB_2_3', n_samples, 4,11,
#                       (lambda x,y: np.sqrt(x**2 + y**2)*(x**2 + y**2) ),[2,3],
#                       seed=seed)
# info_examples.ejemplo('ejemploC', n_samples, 2,11,
#                       (lambda x,y: np.sin(np.sqrt(x**2 + y**2))/(x**2 + y**2) ),[0,1],
#                       seed=seed)
#
# info_examples.ejemplo('ejemploC_Uruido01', n_samples, 2,11,
#                       (lambda x,y: np.sin(np.sqrt(x**2 + y**2))/(x**2 + y**2) ),[0,1],
#                       ruido=np.random.rand(),porc_ruido=0.1,
#                       seed=seed)
# info_examples.ejemplo('ejemploC_Uruido05', n_samples, 2,11,
#                       (lambda x,y: np.sin(np.sqrt(x**2 + y**2))/(x**2 + y**2) ),[0,1],
#                       ruido=np.random.rand(),porc_ruido=0.5,
#                       seed=seed)
# info_examples.ejemplo('ejemploC_Nruido01', n_samples, 2,11,
#                       (lambda x,y: np.sin(np.sqrt(x**2 + y**2))/(x**2 + y**2) ),[0,1],
#                       ruido=np.random.normal(0.5,0.1),porc_ruido=0.1,
#                       seed=seed)
# info_examples.ejemplo('ejemploC_Nruido05', n_samples, 2,11,
#                       (lambda x,y: np.sin(np.sqrt(x**2 + y**2))/(x**2 + y**2) ),[0,1],
#                       ruido=np.random.normal(0.5,0.1),porc_ruido=0.5,
#                       seed=seed)
#
# Y(X) = 10 sin(pi x1 x2) + 20(x3 - 0.5)^2 + 10x4 +5x5
# info_examples.ejemplo('ejemploD', n_samples, 5,11,
#                       (lambda x1,x2,x3,x4,x5: 10*np.sin(np.pi*x1*x2) + 20*(x3 - 0.5)**2 + 10*x4 + 5*x5 ),[0,1,2,3,4],
#                       seed=seed)
# info_examples.ejemplo_MIDT('ejemploD_MIDT', n_samples, 5,11,
#                       (lambda x1,x2,x3,x4,x5: 10*np.sin(np.pi*x1*x2) + 20*(x3 - 0.5)**2 + 10*x4 + 5*x5 ),[0,1,2,3,4],
#                       seed=seed)

info_examples.ejemplo('ejemploE', n_samples, 3,11,
                      (lambda x,y,z: np.sin(np.sqrt(x**2 + y**2))/(x**2 + y**2) + 20*(z-0.5)**2 ),[0,1,2],
                      seed=seed)
info_examples.ejemplo_MIDT('ejemploE_MIDT', n_samples, 3,11,
                      (lambda x,y,z: np.sin(np.sqrt(x**2 + y**2))/(x**2 + y**2) + 20*(z-0.5)**2 ),[0,1,2],
                      seed=seed)
info_examples.ejemplo('ejemploE_Uruido01', n_samples, 3,11,
                      (lambda x,y,z: np.sin(np.sqrt(x**2 + y**2))/(x**2 + y**2) + 20*(z-0.5)**2 ),[0,1,2],
                      ruido=np.random.rand(),porc_ruido=0.1,
                      seed=seed)
info_examples.ejemplo('ejemploE_Uruido05', n_samples, 3,11,
                      (lambda x,y,z: np.sin(np.sqrt(x**2 + y**2))/(x**2 + y**2) + 20*(z-0.5)**2 ),[0,1,2],
                      ruido=np.random.rand(),porc_ruido=0.5,
                      seed=seed)
info_examples.ejemplo('ejemploE_Nruido01', n_samples, 3,11,
                      (lambda x,y,z: np.sin(np.sqrt(x**2 + y**2))/(x**2 + y**2) + 20*(z-0.5)**2 ),[0,1,2],
                      ruido=np.random.normal(0.5,0.1),porc_ruido=0.1,
                      seed=seed)
info_examples.ejemplo('ejemploE_Nruido05', n_samples, 3,11,
                      (lambda x,y,z: np.sin(np.sqrt(x**2 + y**2))/(x**2 + y**2) + 20*(z-0.5)**2 ),[0,1,2],
                      ruido=np.random.normal(0.5,0.1),porc_ruido=0.5,
                      seed=seed)
# info_examples.ejemplo_todos_algoritmos('ejemploE_todos_algoritmos', n_samples, 3,11,
#                       (lambda x,y,z: np.sin(np.sqrt(x**2 + y**2))/(x**2 + y**2) + 20*(z-0.5)**2 ),[0,1,2],
#                       seed=seed)

print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
