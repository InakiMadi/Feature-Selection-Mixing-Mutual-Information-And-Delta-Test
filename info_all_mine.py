#!/usr/bin/env python

from __future__ import division
from numpy  import array, shape, where, in1d, intersect1d, asarray, copy, delete, int64, float64, argmax, argmin, sum, insert
import math
import time
import nose
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict

import matplotlib.pyplot as plt

class Informacion:

    def __init__(self, X, Y = None, last_feature_is_Y = False):
        """
        """
        self.max_parameters = 5
        self.n_rows = X.shape[0]
        if Y is None and last_feature_is_Y:
            self.X = copy(X[...,:-1])
            self.Y = copy(X[...,-1])
        else:
            self.X = copy(X)
            self.Y = copy(Y)
        self.n_cols = self.X.shape[1]

    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y

    def unir(self,X,Y):
        return insert(arr=X,obj=X.shape[1],values=Y,axis=1)

    #-------------------------------#
    #-------------------------------#
    #       MUTUAL INFORMATION      #
    #-------------------------------#
    #-------------------------------#

    def single_entropy(self,log_base=2, debug = False, data = None):
        """
        Calculate the entropy of a random variable
        """
        X = None
        n_rows = self.n_rows
        if data is None:
            X = copy(self.X)
        else:
            X = copy(data)
        # Variable to return entropy
        summation = 0.0
        # Get uniques values of random variables
        X = X.reshape(-1)
        values_x = Counter(X)
        # Print debug info
        if debug:
            print('Entropy of')
            print(X[:])
        # For each random
        for x in values_x.keys():
            px = values_x.get(x) / n_rows
            if px > 0.0:
                summation += px * math.log(px, log_base)
            if debug:
                print('(%d) px:%f' % (x, px))
        if summation == 0.0:
            return summation
        else:
            return - summation


    def entropy(self, log_base=2, data=None,debug = False):
        suma = 0.0
        X = None
        n_cols = 0
        n_rows = 0
        if data is None:
            X = self.X
            n_rows = self.n_rows
            n_cols = self.n_cols
        else:
            X = data
            n_rows = X.shape[0]
            n_cols = X.shape[1]

        A = []
        B = []
        C = []
        D = []
        E = []
        F = []
        G = []
        H = []
        I = []
        J = []
        K = []
        L = []
        A = set(X[:,0])
        if 1 < n_cols:
            B = set(X[:,1])
            if 2 < n_cols:
                C = set(X[:,2])
                if 3 < n_cols:
                    D = set(X[:,3])
                    if 4 < n_cols:
                        E = set(X[:,4])
                        if 5 < n_cols:
                            F = set(X[:,5])
                            if 6 < n_cols:
                                G = set(X[:,6])
                                if 7 < n_cols:
                                    H = set(X[:,7])
                                    if 8 < n_cols:
                                        I = set(X[:,8])
                                        if 9 < n_cols:
                                            J = set(X[:,9])
                                            if 10 < n_cols:
                                                K = set(X[:,10])
                                                if 11 < n_cols:
                                                    L = set(X[:,11])
        joint = []
        for a in A:
            joint = where( X[:,0]==a )[0].tolist()
            if len(B) == 0:
                p = len(joint) / n_rows
                if p > 0.0:
                    suma += p * math.log(p, log_base)
            else:
                for b in B:
                    joint = [i for i in joint if i in where( X[:,1]==b )[0].tolist() ]
                    if len(C) == 0:
                        p = len(joint) / n_rows
                        if p > 0.0:
                            suma += p * math.log(p, log_base)
                    else:
                        for c in C:
                            joint = [i for i in joint if i in where( X[:,2]==c )[0].tolist() ]
                            if len(D) == 0:
                                p = len(joint) / n_rows
                                if p > 0.0:
                                    suma += p * math.log(p, log_base)
                            else:
                                for d in D:
                                    joint = [i for i in joint if i in where( X[:,3]==d )[0].tolist() ]
                                    if len(E) == 0:
                                        p = len(joint) / n_rows
                                        if p > 0.0:
                                            suma += p * math.log(p, log_base)
                                    else:
                                        for e in E:
                                            joint = [i for i in joint if i in where( X[:,4]==e )[0].tolist() ]
                                            if len(F) == 0:
                                                p = len(joint) / n_rows
                                                if p > 0.0:
                                                    suma += p * math.log(p, log_base)
                                            else:
                                                for f in F:
                                                    joint = [i for i in joint if i in where( X[:,5]==f )[0].tolist() ]
                                                    if len(G) == 0:
                                                        p = len(joint) / n_rows
                                                        if p > 0.0:
                                                            suma += p * math.log(p, log_base)
                                                    else:
                                                        for g in G:
                                                            joint = [i for i in joint if i in where( X[:,6]==g )[0].tolist() ]
                                                            if len(H) == 0:
                                                                p = len(joint) / n_rows
                                                                if p > 0.0:
                                                                    suma += p * math.log(p, log_base)
                                                            else:
                                                                for h in H:
                                                                    joint = [i for i in joint if i in where( X[:,7]==h )[0].tolist() ]
                                                                    if len(I) == 0:
                                                                        p = len(joint) / n_rows
                                                                        if p > 0.0:
                                                                            suma += p * math.log(p, log_base)
                                                                    else:
                                                                        for ii in I:
                                                                            joint = [i for i in joint if i in where( X[:,8]==ii )[0].tolist() ]
                                                                            if len(J) == 0:
                                                                                p = len(joint) / n_rows
                                                                                if p > 0.0:
                                                                                    suma += p * math.log(p, log_base)
                                                                            else:
                                                                                for j in J:
                                                                                    joint = [i for i in joint if i in where( X[:,9]==j )[0].tolist() ]
                                                                                    if len(K) == 0:
                                                                                        p = len(joint) / n_rows
                                                                                        if p > 0.0:
                                                                                            suma += p * math.log(p, log_base)
                                                                                    else:
                                                                                        for k in K:
                                                                                            joint = [i for i in joint if i in where( X[:,10]==k )[0].tolist() ]
                                                                                            if len(L) == 0:
                                                                                                p = len(joint) / n_rows
                                                                                                if p > 0.0:
                                                                                                    suma += p * math.log(p, log_base)
                                                                                            else:
                                                                                                for l in L:
                                                                                                    joint = [i for i in joint if i in where( X[:,11]==l )[0].tolist() ]
                                                                                                    p = len(joint) / n_rows
                                                                                                    if p > 0.0:
                                                                                                        suma += p * math.log(p, log_base)
        if suma == 0.0:
            return suma
        else:
            return -suma




    def entropy_2(self, log_base=2, data=None):
        """
        Calculate the entropy between two random variable
        """
        X = None
        n_rows = self.n_rows
        if data is None:
            X = self.X
        else:
            X = data

        # Variable to return MI
        summation = 0.0
        # Get uniques values of random variables
        t_s = time.time()
        values_x = set(X[:,0])
        # For each random
        for x in values_x:
            # List of positions where X_i = x_i.
            posiciones_x = where(X[:,0]==x)[0].tolist()
            # We will find if Y_i = y_i when X_i = x_i. Each time
            # this happens, we will remove the position from posiciones_x.
            # When we reach the end of posiciones_x, we update summation
            # and we will look for the y_is of the rest of posiciones_x, and repeat.
            while len(posiciones_x) > 0:
                # We get the first y_i, count it (uniform probability)
                # and we remove the position from posiciones_x.
                y = X[:,1][posiciones_x[0]]
                cont_y = 1
                del posiciones_x[0]
                # We iterate in posiciones_x in a specific way: if we remove a
                # position then we don't update i, else i += 1.
                # Note that len(posiciones_x) is being updated inside the loop.
                i = 0
                while i < len(posiciones_x):
                    # We find y_j.
                    z = X[:,1][posiciones_x[i]]
                    # If y_i == y_j, then we count it and remove.
                    if y == z:
                        cont_y += 1
                        del posiciones_x[i]
                    # Else, we iterate to the next element.
                    else:
                        i += 1
                # p(x,y) with uniform probability.
                pxy = cont_y / n_rows
                if pxy > 0.0:
                    summation += pxy * math.log(pxy, log_base)
        if summation == 0.0:
            return summation
        else:
            return - summation


    def mutual_information_1(self, log_base=2, data = None, debug = False):
        """
        Calculate and return Mutual information between two random variables
        """
        X = None
        if data is None:
            X = self.X
        else:
            X = data

        # Check if index are into the bounds
        assert (f_index >= 0 and f_index <= n_cols)
        assert(len(self.Y) > 0)
        # Variable to return MI
        summation = 0.0
        # Get uniques values of random variables
        f = X[:]
        values_x = Counter(f)
        values_y = Counter(self.Y)
        # Print debug info
        if debug:
            print('MI between')
            print(f)
            print(self.Y)
        # For each random
        for x in values_x.keys():
            # MI
            posiciones_x = where(f==x)[0].tolist()
            # We will find if Y_i = y_i when X_i = x_i. Each time
            # this happens, we will remove the position from posiciones_x.
            # When we reach the end of posiciones_x, we update summation
            # and we will look for the y_is of the rest of posiciones_x, and repeat.
            while len(posiciones_x) > 0:
                y = self.Y[posiciones_x[0]]
                cont_y = 1
                del posiciones_x[0]
                i = 0
                while i < len(posiciones_x):
                    z = self.Y[posiciones_x[i]]
                    if y == z:
                        cont_y += 1
                        del posiciones_x[i]
                    else:
                        i += 1
                pxy = cont_y / self.n_rows
                #print(cont_y)
                # Single Entropies
                px = values_x.get(x) / self.n_rows
                py = values_y.get(y) / self.n_rows
                # Summation
                if pxy > 0.0:
                    summation += pxy * math.log((pxy / (px*py)), log_base)
                if debug:
                    print('(%d,%d) x(%d,%d,%d) px:%f py:%f pxy:%f' % (x, y, values_x.get(x),values_y.get(y),cont_y, px, py, pxy))
                    print("  log: ",(pxy/(px*py)))
                    print("  suma:",summation)
        return summation

    # def mutual_information(self,data=None,log_base=2):
    #     X = None
    #     n_cols = 0
    #     if data is None:
    #         X = self.X
    #         n_cols = self.n_cols
    #     else:
    #         X = data
    #         n_cols = X.shape[1]
    #     X = self.unir(X,self.Y)
    #     n_cols += 1
    #     mi = 0.0
    #     mi -= self.entropy(data=X,log_base=log_base)
    #     for i in range(n_cols):
    #         mi += self.single_entropy(data=X[:,i])
    #     return mi
    def mutual_information(self,data=None,log_base=2):
        X = None
        n_cols = 0
        if data is None:
            X = self.X
            n_cols = self.n_cols
        else:
            X = data
            n_cols = X.shape[1]
        X = self.unir(X,self.Y)
        n_cols += 1
        mi = 0.0
        if n_cols%2 == 0:
            mi -= self.entropy(data=X,log_base=log_base)
        else:
            mi += self.entropy(data=X,log_base=log_base)
        # Todas las posibilidades con N columnas (2^N)
        for i in range(1,2**n_cols-1):
            # Obtenemos el número en binario.
            bin_i = bin(i)[2:]
            # Añadimos los 0 al principio que falten.
            # Por ejemplo, para 7 columnas, el 5 ha de ser 0000101.
            for j in range(n_cols - len(bin_i)):
                bin_i = '0' + bin_i
            # Ahora, añadimos las posiciones de los '1' en cols.
            cols = []
            for j in range( len(bin_i) ):
                if bin_i[j] == '1':
                    cols.append(j)
            if len(cols) == 1:
                mi += self.single_entropy(data=X[:,cols],log_base=log_base)
            elif len(cols) == 2:
                mi -= self.entropy_2(data=X[:,cols],log_base=log_base)
            elif len(cols)%2 == 0:
                mi -= self.entropy(data=X[:,cols],log_base=log_base)
            else:
                mi += self.entropy(data=X[:,cols],log_base=log_base)

        return mi

    def brute_force_MI(self):
        """
        A Force-brute algorithm that returns the maximum of all the Mutual Informations,
        ordered by position in bits.

        For example, the MI in position 5 means the MI of '00101', which means
        the MI of columns 2 and 4 (X[:,[2,4]]).

        Output: Returns the columns that reach the max MI from every possibility (2^N)
        and the max.
        """
        X = copy(self.X)
        n_cols = self.n_cols
        mi = []

        mi.append(self.mutual_information(data=X))
        # Todas las posibilidades con N columnas (2^N)
        for i in range(1,2**n_cols-1):
            # Obtenemos el número en binario.
            bin_i = bin(i)[2:]
            # Añadimos los 0 al principio que falten.
            # Por ejemplo, para 7 columnas, el 5 ha de ser 0000101.
            for j in range(n_cols - len(bin_i)):
                bin_i = '0' + bin_i
            # Ahora, añadimos las posiciones de los '1' en cols.
            cols = []
            for j in range( len(bin_i) ):
                if bin_i[j] == '1':
                    cols.append(j)
            # Realizamos la MI en las columnas cols.
            mi.append(self.mutual_information(data=X[:,cols]))
        # Ahora devolvemos el máximo y las columnas usadas
        print(mi)
        pos_max = argmax(mi)
        bin_pos_max = bin(pos_max)[2:]
        # Añadimos los 0 al principio que falten.
        for j in range(n_cols - len(bin_pos_max)):
            bin_pos_max = '0' + bin_pos_max
        # Ahora, añadimos las posiciones de los '1' en cols (sin incluir la Y)
        cols = []
        for j in range( len(bin_pos_max) ):
            if bin_pos_max[j] == '1':
                cols.append(j)
        # Si el óptimo no tiene 1s en binario, es que se alcanza con todas las columnas
        if len(cols) == 0:
            cols = list(range(n_cols))
        return (cols,mi[pos_max])


    #------------------#
    #------------------#
    #       DELTA      #
    #------------------#
    #------------------#

    # K Vecino Mas Cercano con un X modificado.
    def KVecinoMasCercano(self,x_index,data=None,k=1):
        """
        Returns the index of the K nearest neighbor of an x_i.
        """
        X = None
        if data is None:
            X = self.X
        else:
            X = data
        # p=2 means L2 distance (Euclidean).
        # K is k+1 as the point itself is the nearest.
        knn = NearestNeighbors(n_neighbors=k+1,p=2)
        knn.fit(X)
        # As X[i] is a single sample, we reshape it.
        # At the end, we return the index of the Kth element.
        # To return the list dropping the point itself out, it's [0][1:].
        return knn.kneighbors(X[x_index].reshape(1,-1), return_distance=False)[0][k]

    def TestDelta(self,data=None,k=1):
        """
        Returns the noise estimation variance calculated by the Delta Test.
        """
        X = None
        n_cols = 0
        if data is None:
            X = self.X
            n_cols = self.n_cols
        else:
            X = data
            n_cols = X.shape[1]
        suma = 0.0
        for i in range(n_cols):
            # For each element in the data, we find its Kth nearest neighbor.
            NN_index = self.KVecinoMasCercano(data=X,x_index=i,k=k)
            # We calculate the formula.
            suma += float(self.Y[i] - self.Y[NN_index])**2
        return suma / (2*n_cols)

    def brute_force_Delta(self,k=1):
        """
        A Force-brute algorithm that returns the minimum of all the Delta Tests,
        ordered by position in bits.

        For example, the Delta Test in position 5 means the Delta Test of '00101', which means
        the Delta Test of columns 2 and 4 (X[:,[2,4]]).

        Output: Returns the columns that reach the min Delta Test from every possibility (2^N)
        and the min.
        """
        delta = []
        delta.append( self.TestDelta(k=k) )
        # Todas las posibilidades con N columnas (2^N)
        for i in range(1,2**self.n_cols):
            # Obtenemos el número en binario.
            bin_i = bin(i)[2:]
            # Añadimos los 0 al principio que falten.
            # Por ejemplo, para 7 columnas, el 5 ha de ser 0000101.
            for j in range(self.n_cols - len(bin_i)):
                bin_i = '0' + bin_i
            # Ahora, añadimos las posiciones de los '1' en cols.
            cols = []
            for j in range( len(bin_i) ):
                if bin_i[j] == '1':
                    cols.append(j)
            # Realizamos la MI en las columnas cols.
            delta.append( self.TestDelta(data=self.X[:,cols],k=k) )
        # Ahora devolvemos el mínimo y las columnas que llegan a él.
        pos_min = argmin(delta)
        bin_pos_min = bin(pos_min)[2:]
        # Añadimos los 0 al principio que falten.
        for j in range(self.n_cols - len(bin_pos_min)):
            bin_pos_min = '0' + bin_pos_min
        # Ahora, añadimos las posiciones de los '1' en cols.
        cols = []
        for j in range( len(bin_pos_min) ):
            if bin_pos_min[j] == '1':
                cols.append(j)
        # Si el óptimo no tiene 1s en binario, es que se alcanza con todas las columnas
        if len(cols) == 0:
            cols = [0]
        return (cols,delta[pos_min],delta)




















    #
    # NOT USED
    #

    def filter_MI(self, subset_indexes = None):
        """
        Greedy maximizing algorithm.
        A Filter Algorithm based on Mutual Information.

        Input: num_features for how many features we would like,
        subset_indexes if we'd like to start with some features already selected.

        Output: The indexes of the features selected.
        """
        # We initialize some items.
        features = 0
        F_pos = list(range(self.n_cols))
        # We will be deleting columns and positions from F_pos. To return the indexes of the features selected,
        # not_excluded_pos control is really important.
        not_excluded_pos = list(range(self.n_cols))
        S_pos = subset_indexes
        # We will manipulate X so we need to copy it.
        X = copy(self.X)
        # We check if there are already features selected (and remove them from X).
        if S_pos is None:
            S_pos = []
        else:
            for i in S_pos:
                X = delete(arr=X, obj=i, axis=1)
                # F_pos and not_excluded_pos must have the same number of indexes, but
                # not_excluded_pos must have the control of the indexes of the original X.
                # So if there are features already selected, they will not be part
                # of the algorithm.
                del not_excluded_pos[i]
            F_pos = list(range(self.n_cols - len(S_pos)))
        # By default, if number of features we will want is not determined, then it will be the truncation of the half.
        # if num_features is None:
        #     features = int(self.n_cols/2)
        # else:
        #     features = num_features
        # assert(features < self.n_cols)

        # The initialization of mi_so_far and mi_is_ascending,
        # to stop when we stop ascending.
        mi_so_far = self.mutual_information(log_base=2,data=X)
        mi_is_ascending = True
        t_s = 0
        # Number of features deleted (number of iterations in the big loop).
        deleted_features = 0
        # If we need n features, it's the same to N - (number_of_features_already_selected + number_of_features_deleted) == n.
        # (self.n_cols - len(S_pos) - deleted_features) != features

        while mi_is_ascending and (len(F_pos) > 1):
            t_s = time.time()
            max_mi = -9999
            max_fi = -1
            # For each feature, we copy X, delete the feature (column),
            # for the rest of the features we sum the MI between each feature and the class,
            # then update the maximum. And repeat.
            for fi in F_pos:
                X_aux = delete(arr=X, obj=fi, axis=1)
                # We calculate the MI.
                mi = self.mutual_information(log_base=2,data=X_aux)
                # We update the maximum.
                if mi > max_mi:
                    max_mi = mi
                    max_fi = fi
            print(mi_so_far,max_mi)
            if max_mi > mi_so_far:
                mi_so_far = max_mi
                # We delete the feature which provided the maximum delta.
                X = delete(arr=X, obj=max_fi, axis=1)
                deleted_features += 1
                # We update the indexes to iterate with.
                #F_pos = F_pos[:max_fi] + [i-1 for i in F_pos[max_fi+1:]]
                F_pos = list(range(self.n_cols - len(S_pos) - deleted_features))
                # We update the list that controls the features of the original X.
                del not_excluded_pos[max_fi]
            else:
                mi_is_ascending = False

            #print("Min:", min_delta, min_fi, not_excluded_pos)
            #print("Tiempo:",time.time()-t_s)
        # We return the features already selected, and the columns that weren't deleted.
        # Also, we return the MI.
        return (S_pos + not_excluded_pos, mi_so_far)


    def mRMR_MI(self, num_features = None, subset_indexes = None):
        """
        Maximum Relevance Minimum Redundancy algorithm.
        A Filter Algorithm based on Mutual Information.

        Input: num_features for how many features we would like,
        subset_indexes if we'd like to start with some features already selected.

        Output: The indexes of the features selected.
        """
        features = 0
        F_pos = list(range(self.n_cols))
        S_pos = subset_indexes
        # By default, if number of features we will want is not determined, then it will be the truncation of the half.
        if num_features is None:
            features = int(self.n_cols/2)
        else:
            features = num_features
        assert(features < self.n_cols)
        # We check if there are already features selected (and remove them from X).
        if S_pos is None:
            S_pos = []
        # If there are features already selected, we won't look into those features.
        else:
            F_pos = [i for i in list(range(self.n_cols)) if i not in S_pos]

        # Until we have the number of features we want:
        while len(S_pos) != features:
            max_mrmr = -9999
            max_fi = -1
            t_s = 0
            # For each feature, we calculate the MI between the feature and the class (relevance).
            for fi in F_pos:
                t_s = time.time()
                relevance = self.mutual_information_fc(f_index=fi,log_base=2)
                redundancy = 0
                # For each selected feature, we calculate the MI between the new feature and the selected feature (redundance).
                for fj in S_pos:
                    redundancy += self.mutual_information_ff(f_index1=fi,f_index2=fj, log_base=2)
                # We calculate what we want to maximize as a mRMR algorithm.
                mrmr = 0.0
                if len(S_pos)>0:
                    mrmr = relevance - redundancy / len(S_pos)
                else:
                    mrmr = relevance - redundancy
                # We update the maximum.
                if mrmr > max_mrmr:
                    max_mrmr = mrmr
                    max_fi = fi
            # We drop the selected feature from the investigation.
            F_pos.remove(max_fi)
            # We add the selected feature to the list of selected features.
            S_pos.append(max_fi)
            #print("Tiempo:",time.time()-t_s)
        # We return the indexes of the selected features.
        return S_pos


    def filter_Delta(self, k=1, subset_indexes = None):
        """
        Greedy minimizing algorithm.
        A Filter Algorithm based on Delta Test.

        Input: k for kth nearest neighbor, num_features for how many features we would like,
        subset_indexes if we'd like to start with some features already selected, delta_is_zero to enable the condition
        to stop if we reach a delta = 0.

        Output: The indexes of the features selected.
        """
        assert(self.n_rows >= 2)
        # We initialize some items.
        features = 0
        F_pos = list(range(self.n_cols))
        # We will be deleting columns and positions from F_pos. To return the indexes of the features selected,
        # not_excluded_pos control is really important.
        not_excluded_pos = list(range(self.n_cols))
        S_pos = subset_indexes
        # We will manipulate X so we need to copy it.
        X = copy(self.X)
        # We check if there are already features selected (and remove them from X).
        if S_pos is None:
            S_pos = []
        else:
            for i in S_pos:
                X = delete(arr=X, obj=i, axis=1)
                # F_pos and not_excluded_pos must have the same number of indexes, but
                # not_excluded_pos must have the control of the indexes of the original X.
                # So if there are features already selected, they will not be part
                # of the algorithm.
                del not_excluded_pos[i]
            F_pos = list(range(self.n_cols - len(S_pos)))
        # By default, if number of features we will want is not determined, then it will be the truncation of the half.
        # if num_features is None:
        #     features = int(self.n_cols/2)
        # else:
        #     features = num_features
        # assert(features < self.n_cols)

        # The initialization of delta_so_far and delta_is_descending,
        # to stop when we stop descending. Also, we stop if we reach to a delta = 0.
        delta_so_far = self.TestDelta(data=X,k=k)
        delta_is_descending = True
        t_s = 0
        # Number of features deleted (number of iterations in the big loop).
        deleted_features = 0
        # If we need n features, it's the same to N - (number_of_features_already_selected + number_of_features_deleted) == n.
        # (self.n_cols - len(S_pos) - deleted_features) != features)

        while (delta_is_descending) and (len(F_pos) > 1):
            t_s = time.time()
            min_delta = 99999999999999999999999
            min_fi = -1
            # For each feature, we copy X, delete the feature (column), apply delta and update the minimum. And repeat.
            for fi in F_pos:
                X_aux = delete(arr=X, obj=fi, axis=1)
                # We calculate the delta.
                delta = self.TestDelta(data=X_aux,k=k)
                #print(delta)
                # We update the minimum.
                if delta < min_delta:
                    min_delta = delta
                    min_fi = fi
            #print("----------")
            if min_delta < delta_so_far:
                #print("Mínimo delta:",min_delta)
                #print()
                delta_so_far = min_delta
                # We delete the feature with the minimum delta.
                X = delete(arr=X, obj=min_fi, axis=1)
                deleted_features += 1
                # We update the indexes to iterate with.
                #F_pos = F_pos[:min_fi] + [i-1 for i in F_pos[min_fi+1:]]
                F_pos = list(range(self.n_cols - len(S_pos) - deleted_features))
                # We update the list that controls the features of the original X.
                del not_excluded_pos[min_fi]
            else:
                delta_is_descending = False

            #print("Min:", min_delta, min_fi, not_excluded_pos)
            #print("Tiempo:",time.time()-t_s)
        # We return the features already selected, and the columns that weren't deleted.
        # Also, we return the delta.
        return (S_pos + not_excluded_pos, delta_so_far)


    # ALL TOGETHER #

    def filter_MI_Delta(self, k=1, num_features = None, subset_indexes = None):
        """
        Greedy minimizing algorithm.
        A Filter Algorithm based on the mix of MI and Delta Test.

        Input: k for kth nearest neighbor, num_features for how many features we would like,
        subset_indexes if we'd like to start with some features already selected.

        Output: The indexes of the features selected.
        """
        # We initialize some items.
        features = 0
        F_pos = list(range(self.n_cols))
        # We will be deleting columns and positions from F_pos. To return the indexes of the features selected,
        # not_excluded_pos control is really important.
        not_excluded_pos = list(range(self.n_cols))
        S_pos = subset_indexes
        # We will manipulate X so we need to copy it.
        X = copy(self.X)
        # By default, if number of features we will want is not determined, then it will be the truncation of the half.
        if num_features is None:
            features = int(self.n_cols/2)
        else:
            features = num_features
        assert(features < self.n_cols)
        # We check if there are already features selected (and remove them from X).
        if S_pos is None:
            S_pos = []
        else:
            for i in S_pos:
                X = delete(arr=X, obj=i, axis=1)
                # F_pos and not_excluded_pos must have the same number of indexes, but
                # not_excluded_pos must have the control of the indexes of the original X.
                # So if there are features already selected, they will not be part
                # of the algorithm.
                del not_excluded_pos[i]
            F_pos = list(range(self.n_cols - len(S_pos)))

        # The initialization of delta_so_far, to check if we reach to a delta = 0.
        delta_so_far = -1
        t_s = 0
        # Number of features deleted (number of iterations in the big loop).
        deleted_features = 0
        # If we need n features, it's the same to N - (number_of_features_already_selected + number_of_features_deleted) == n.
        while (self.n_cols - len(S_pos) - deleted_features) != features:
            t_s = time.time()
            max_mi_delta = -9999
            max_fi = -1
            # For each feature, we copy X, delete the feature (column),
            # apply delta and for the rest of the features we sum the MI between each feature and the class,
            # then update the minimum. And repeat.
            for fi in F_pos:
                X_aux = delete(arr=X, obj=fi, axis=1)
                # We calculate the delta.
                delta = self.TestDelta(data=X_aux,k=k)
                # We calculate the MI: the sum of the MI between the features fj and the class.
                # We calculate the MI.
                mi = self.mutual_information(log_base=2,data=X_aux)
                # We maximize the MI and minimize the Delta.
                # In other words, we maximize (MI - Delta).
                mi_delta = mi - delta
                #print(mi_delta)
                # We update the maximum.
                if mi_delta > max_mi_delta:
                    max_mi_delta = mi_delta
                    max_fi = fi
            # We delete the feature with the maximum.
            X = delete(arr=X, obj=max_fi, axis=1)
            deleted_features += 1
            # We update the indexes to iterate with.
            F_pos = F_pos[:max_fi] + [i-1 for i in F_pos[max_fi+1:]]
            # We update the list that controls the features of the original X.
            # Also we return the MI and the Delta.
            mi = 0.0
            for fi in F_pos:
                mi += self.mutual_information_fc(f_index=fi,log_base=2,data=X)
            delta = self.TestDelta(data=X,k=k)
            del not_excluded_pos[max_fi]

            #print("Min:", min_delta, min_fi, not_excluded_pos)
            #print("Tiempo:",time.time()-t_s)
        # We return the features already selected, and the columns that weren't deleted.
        return (S_pos + not_excluded_pos, mi, delta)

    # Está mal la parte del Delta todavía
    def _MI_Delta(self, k=1, num_features = None, subset_indexes = None):
        """
        Maximum Relevance Minimum Redundancy algorithm.
        A Filter Algorithm based on Mutual Information.
        """
        features = 0
        F_pos = list(range(self.n_cols))
        S_pos = subset_indexes
        if num_features is None:
            features = int(self.n_cols/2)
        else:
            features = num_features
        assert(features < self.n_cols)
        if S_pos is None:
            S_pos = []
        else:
            F_pos = [i for i in list(range(self.n_cols)) if i not in S_pos]

        deleted_features = 0
        X = copy(self.X)
        while len(S_pos) != features:
            # MI
            max_mrmr = -9999
            max_fi = -1
            mrmrs = []
            fis = []
            max_fis = []
            for fi in F_pos:
                relevance = self.mutual_information_fc(f_index=fi,log_base=2)
                redundancy = 0
                for fj in S_pos:
                    redundancy += self.mutual_information_ff(f_index1=fi,f_index2=fj, log_base=2)
                mrmr = 0.0
                if len(S_pos)>0:
                    mrmr = relevance - redundancy / len(S_pos)
                else:
                    mrmr = relevance - redundancy
                mrmrs.append(mrmr)
                max_fis.append(fi)
                fis.append(fi)
            # Delta
            min_delta = 9999
            min_fi = -1
            deltas = []
            min_fis = []
            for fi in F_pos:
                X_aux = delete(arr=X, obj=fi, axis=1)
                delta = self.TestDelta_X(X=X_aux,k=k)
                deltas.append(delta)
                min_fis.append(fi)
            #
            real_max = -9999
            real_fi = -1
            for i in range(len(F_pos)):
                aux = mrmrs[i] - deltas[i]
                if aux > real_max:
                    real_max = aux
                    real_fi = fis[i]
            F_pos = F_pos[:real_fi] + [i-1 for i in F_pos[real_fi+1:]]
            S_pos.append(real_fi + deleted_features)
            X = delete(arr=X, obj=real_fi, axis=1)
            deleted_features += 1
        return S_pos


    #
    # LEARNERS
    #

    def logistic_regression_lerner(self,cols_pos = None,seed = 13, tol = 0.01):
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        # Separamos en train y test
        if cols_pos is None:
            x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=seed)
        else:
            X = self.X[:, cols_pos]
            x_train, x_test, y_train, y_test = train_test_split(X, self.Y, test_size=0.2, random_state=seed)
        x_train = array(x_train, int64)
        y_train = array(y_train, int64)
        x_test = array(x_test, int64)
        y_test = array(y_test, int64)

        # Ajuste de parámetros
        solver_final = 'newton-cg'
        # solvers = ['newton-cg', 'sag', 'saga', 'lbfgs']
        # s_final = 'NONE'
        # max = 0
        # for s in solvers:
        # 	m = LogisticRegressionCV(cv=5, random_state=seed,
        # 	         solver=s, multi_class='multinomial').fit(x_train,y_train)
        # 	p = m.score(x_test,y_test)
        # 	if(p > max):
        # 		max = p
        # 		s_final = s
        # solver_final = s_final
        # print("Solver con mejor resultado:",solver_final)

        # Modelo
        modelo_clas = LogisticRegressionCV(cv=5, random_state=seed,
                 multi_class='multinomial',solver=solver_final,tol=tol).fit(x_train,y_train)

        # Predecimos con el conjunto test.
        y_predicted = modelo_clas.predict(x_test)

        # Porcentaje bien clasificado
        p = sum(y_test == y_predicted) / len(y_test)
        # Devolvemos el porcentaje bien clasificado, el porcentaje mal clasificado,
        # el vector verdadero y el vector predicho.
        return (p,1-p, y_test,y_predicted)

    def linear_regression_lerner(self,seed = 13):
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        # Separamos en train y test
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=seed)
        x_train = array(x_train, float64)
        y_train = array(y_train, float64)
        x_test = array(x_test, float64)
        y_test = array(y_test, float64)

        # Ajuste de parámetros
        eps_final = 0.01
        # epss = [1, 0.1, 0.01, 0.001, 0.0001]
        # eps_final = -1
        # p_min = 1e10
        # for e in epss:
        # 	m = LassoCV(cv=5, eps=e, random_state=seed).fit(x_train,y_train)
        # 	#Predecimos en el conjunto test
        # 	y_predicted = m.predict(x_test)
        # 	#Realizamos el MSE entre lo predicho y lo verdadero
        # 	p = mean_squared_error(y_test,y_predicted)
        # 	print(p)
        # 	if(p < p_min):
        # 		p_min = p
        # 		eps_final = e
        # print("EPS con mejor resultado:",eps_final)

        # Modelo
        modelo_lin = LassoCV(cv=5, eps=eps_final, random_state=seed).fit(x_train,y_train)
        #Predecimos en el conjunto test
        y_predicted = modelo_lin.predict(x_test)
        #Realizamos el MSE entre lo predicho y lo verdadero
        mse = mean_squared_error(y_test,y_predicted)
        print( "Score de Regresión Lineal (R^2): ", modelo_lin.score(x_test,y_test) )
        print( "E_out (MSE): ", mse )
        print( "RSME: ", sqrt(mse) )

        return (y_test,y_predicted,mse)


    #
    # PLOT
    #
    # No probado
    def plot(self):
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        # Separamos en train y test
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=seed)
        x_train = array(x_train, float64)
        y_train = array(y_train, float64)
        x_test = array(x_test, float64)
        y_test = array(y_test, float64)

        # Visualización de datos
        x_train2 = x_train.copy()
        x_test2 = x_test.copy()
        pca_train = PCA(n_components=2)
        pca_test = PCA(n_components=2)
        x_train2 = pca_train.fit_transform(x_train2)
        x_test2 = pca_test.fit_transform(x_test2)
        plt.scatter( x_train2[:,0], x_train2[:,1], c=y_train )
        plt.xlabel("Primera característica tras dimensionalizar")
        plt.ylabel("Segunda característica tras dimensionalizar")
        plt.title("Clasificación del train al dimensionalizar con PCA")
        plt.show()
        plt.scatter( x_test2[:,0], x_test2[:,1], c=y_test )
        plt.xlabel("Primera característica tras dimensionalizar")
        plt.ylabel("Segunda característica tras dimensionalizar")
        plt.title("Clasificación del test al dimensionalizar con PCA")
        plt.show()
