#!/usr/bin/env python

from __future__ import division
from numpy  import array, shape, where, in1d, intersect1d, asarray, copy, delete, int64, float64, argmax, sum
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

    #-------------------------------#
    #-------------------------------#
    #       MUTUAL INFORMATION      #
    #-------------------------------#
    #-------------------------------#

    def single_entropy_X(self, index, log_base, debug = False):
        """
        Calculate the entropy of a random variable
        """
        # Check if index are into the bounds
        assert (index >= 0 and index <= self.n_rows)
        # Variable to return entropy
        summation = 0.0
        # Get uniques values of random variables
        t_s = time.time()
        values_x = Counter(self.X[index])
        # Print debug info
        if debug:
            print('Entropy of')
            print(self.X[index])
        # For each random
        for x in values_x.keys():
            px = values_x.get(x) / self.n_cols
            if px > 0.0:
                summation += px * math.log(px, log_base)
            if debug:
                print('(%d) px:%f' % (x, px))
        print("Tiempo SE:")
        print(time.time() - t_s)
        if summation == 0.0:
            return summation
        else:
            return - summation

    def single_entropy_Y(yself, index, log_base, debug = False):
        """
        Calculate the entropy of a random variable
        """
        # Check if index are into the bounds
        assert (index >= 0 and index <= self.n_rows)
        # Variable to return entropy
        summation = 0.0
        # Get uniques values of random variables
        t_s = time.time()
        values_x = Counter(self.Y[index])
        # Print debug info
        if debug:
            print('Entropy of')
            print(self.Y[index])
        # For each random
        for x in values_x.keys():
            px = values_x.get(x) / self.n_cols
            if px > 0.0:
                summation += px * math.log(px, log_base)
            if debug:
                print('(%d) py:%f' % (x, px))
        print("Tiempo SE:")
        print(time.time() - t_s)
        if summation == 0.0:
            return summation
        else:
            return - summation


    def entropy_XY(self, x_index, y_index, log_base, debug = False):
        """
        Calculate the entropy between two random variable
        """
        assert (x_index >= 0 and x_index <= self.n_rows)
        assert (y_index >= 0 and y_index <= self.n_rows)
        assert(len(self.Y) > 0)
        # Variable to return MI
        summation = 0.0
        # Get uniques values of random variables
        t_s = time.time()
        values_x = set(self.X[x_index])
        # Print debug info
        if debug:
            print('Entropy between')
            print(self.X[x_index])
            print(self.Y[y_index])
        # For each random
        for x in values_x:
            # List of positions where X_i = x_i.
            posiciones_x = where(self.X[x_index]==x)[0].tolist()
            # We will find if Y_i = y_i when X_i = x_i. Each time
            # this happens, we will remove the position from posiciones_x.
            # When we reach the end of posiciones_x, we update summation
            # and we will look for the y_is of the rest of posiciones_x, and repeat.
            while len(posiciones_x) > 0:
                # We get the first y_i, count it (uniform probability)
                # and we remove the position from posiciones_x.
                y = self.Y[y_index][posiciones_x[0]]
                cont_y = 1
                del posiciones_x[0]
                # We iterate in posiciones_x in a specific way: if we remove a
                # position then we don't update i, else i += 1.
                # Note that len(posiciones_x) is being updated inside the loop.
                i = 0
                while i < len(posiciones_x):
                    # We find y_j.
                    z = self.Y[y_index][posiciones_x[i]]
                    # If y_i == y_j, then we count it and remove.
                    if y == z:
                        cont_y += 1
                        del posiciones_x[i]
                    # Else, we iterate to the next element.
                    else:
                        i += 1
                # p(x,y) with uniform probability.
                pxy = cont_y / self.n_cols
                if pxy > 0.0:
                    summation += pxy * math.log(pxy, log_base)
                if debug:
                    print('(%d,%d) pxy:%f' % (x,y, pxy))
        print("Tiempo E:")
        print(time.time() - t_s)
        if summation == 0.0:
            return summation
        else:
            return - summation

    def entropy_XX(self, x_index, y_index, log_base, debug = False):
        """
        Calculate the entropy between two random variable
        """
        assert (x_index >= 0 and x_index <= self.n_rows)
        assert (y_index >= 0 and y_index <= self.n_rows)
        # Variable to return MI
        summation = 0.0
        # Get uniques values of random variables
        t_s = time.time()
        values_x = set(self.X[x_index])
        # Print debug info
        if debug:
            print('Entropy between')
            print(self.X[x_index])
            print(self.X[y_index])
        # For each random
        for x in values_x:
            # List of positions where X_i = x_i.
            posiciones_x = where(self.X[x_index]==x)[0].tolist()
            # We will find if Y_i = y_i when X_i = x_i. Each time
            # this happens, we will remove the position from posiciones_x.
            # When we reach the end of posiciones_x, we update summation
            # and we will look for the y_is of the rest of posiciones_x, and repeat.
            while len(posiciones_x) > 0:
                # We get the first y_i, count it (uniform probability)
                # and we remove the position from posiciones_x.
                y = self.X[y_index][posiciones_x[0]]
                cont_y = 1
                del posiciones_x[0]
                # We iterate in posiciones_x in a specific way: if we remove a
                # position then we don't update i, else i += 1.
                # Note that len(posiciones_x) is being updated inside the loop.
                i = 0
                while i < len(posiciones_x):
                    # We find y_j.
                    z = self.X[y_index][posiciones_x[i]]
                    # If y_i == y_j, then we count it and remove.
                    if y == z:
                        cont_y += 1
                        del posiciones_x[i]
                    # Else, we iterate to the next element.
                    else:
                        i += 1
                # p(x,y) with uniform probability.
                pxy = cont_y / self.n_cols
                if pxy > 0.0:
                    summation += pxy * math.log(pxy, log_base)
                if debug:
                    print('(%d,%d) pxy:%f' % (x,y, pxy))
        print("Tiempo E:")
        print(time.time() - t_s)
        if summation == 0.0:
            return summation
        else:
            return - summation

    def entropy_YY(self, x_index, y_index, log_base, debug = False):
        """
        Calculate the entropy between two random variable
        """
        assert (x_index >= 0 and x_index <= self.n_rows)
        assert (y_index >= 0 and y_index <= self.n_rows)
        assert(len(self.Y) > 0)
        # Variable to return MI
        summation = 0.0
        # Get uniques values of random variables
        t_s = time.time()
        values_x = set(self.Y[x_index])
        # Print debug info
        if debug:
            print('Entropy between')
            print(self.Y[x_index])
            print(self.Y[y_index])
        # For each random
        for x in values_x:
            # List of positions where X_i = x_i.
            posiciones_x = where(self.Y[x_index]==x)[0].tolist()
            # We will find if Y_i = y_i when X_i = x_i. Each time
            # this happens, we will remove the position from posiciones_x.
            # When we reach the end of posiciones_x, we update summation
            # and we will look for the y_is of the rest of posiciones_x, and repeat.
            while len(posiciones_x) > 0:
                # We get the first y_i, count it (uniform probability)
                # and we remove the position from posiciones_x.
                y = self.Y[y_index][posiciones_x[0]]
                cont_y = 1
                del posiciones_x[0]
                # We iterate in posiciones_x in a specific way: if we remove a
                # position then we don't update i, else i += 1.
                # Note that len(posiciones_x) is being updated inside the loop.
                i = 0
                while i < len(posiciones_x):
                    # We find y_j.
                    z = self.Y[y_index][posiciones_x[i]]
                    # If y_i == y_j, then we count it and remove.
                    if y == z:
                        cont_y += 1
                        del posiciones_x[i]
                    # Else, we iterate to the next element.
                    else:
                        i += 1
                # p(x,y) with uniform probability.
                pxy = cont_y / self.n_cols
                if pxy > 0.0:
                    summation += pxy * math.log(pxy, log_base)
                if debug:
                    print('(%d,%d) pxy:%f' % (x,y, pxy))
        print("Tiempo E:")
        print(time.time() - t_s)
        if summation == 0.0:
            return summation
        else:
            return - summation


    def mutual_information_fc(self, f_index, log_base, data = None, debug = False):
        """
        Calculate and return Mutual information between two random variables
        """
        X = None
        n_cols = 0
        if data is None:
            X = self.X
            n_cols = self.n_cols
        else:
            X = data
            n_cols = X.shape[1]

        # Check if index are into the bounds
        assert (f_index >= 0 and f_index <= n_cols)
        assert(len(self.Y) > 0)
        """
        H_X = self.single_entropy(x_index=x_index,log_base=log_base)
        H_Y = self.single_entropy(x_index=y_index,log_base=log_base)
        H_XY = self.entropy(x_index=x_index,y_index=y_index,log_base=log_base)
        return H_X + H_Y - H_XY
        """
        # Variable to return MI
        summation = 0.0
        # Get uniques values of random variables
        t_s = time.time()
        f = X[...,f_index]
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
                    print("Tiempo MI:")
                    print(time.time() - t_s)
        return summation

    def mutual_information_ff(self, f_index1, f_index2, log_base, data=None, debug = False):
        """
        Calculate and return Mutual information between two random variables
        """
        X = None
        n_cols = 0
        if data is None:
            X = self.X
            n_cols = self.n_cols
        else:
            X = data
            n_cols = X.shape[1]

        # Check if index are into the bounds
        assert (f_index1 >= 0 and f_index1 <= n_cols)
        assert (f_index2 >= 0 and f_index2 <= n_cols)
        """
        H_X = self.single_entropy(x_index=x_index,log_base=log_base)
        H_Y = self.single_entropy(x_index=y_index,log_base=log_base)
        H_XY = self.entropy(x_index=x_index,y_index=y_index,log_base=log_base)
        return H_X + H_Y - H_XY
        """
        # Variable to return MI
        summation = 0.0
        # Get uniques values of random variables
        t_s = time.time()
        f1 = X[...,f_index1]
        f2 = X[...,f_index2]
        values_x = Counter(f1)
        values_y = Counter(f2)
        # Print debug info
        if debug:
            print('MI between')
            print(f1)
            print(f2)
        # For each random
        for x in values_x.keys():
            # MI
            posiciones_x = where(f1==x)[0].tolist()
            # We will find if Y_i = y_i when X_i = x_i. Each time
            # this happens, we will remove the position from posiciones_x.
            # When we reach the end of posiciones_x, we update summation
            # and we will look for the y_is of the rest of posiciones_x, and repeat.
            while len(posiciones_x) > 0:
                y = f2[posiciones_x[0]]
                cont_y = 1
                del posiciones_x[0]
                i = 0
                while i < len(posiciones_x):
                    z = f2[posiciones_x[i]]
                    if y == z:
                        cont_y += 1
                        del posiciones_x[i]
                    else:
                        i += 1
                pxy = cont_y / self.n_rows
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
                    print("Tiempo MI:")
                    print(time.time() - t_s)
        return summation

    def mutual_information_XX(self, x_index, y_index, log_base, debug = False):
        """
        Calculate and return Mutual information between two random variables
        """
        # Check if index are into the bounds
        assert (x_index >= 0 and x_index <= self.n_rows)
        assert (y_index >= 0 and y_index <= self.n_rows)
        """
        H_X = self.single_entropy(x_index=x_index,log_base=log_base)
        H_Y = self.single_entropy(x_index=y_index,log_base=log_base)
        H_XY = self.entropy(x_index=x_index,y_index=y_index,log_base=log_base)
        return H_X + H_Y - H_XY
        """
        # Variable to return MI
        summation = 0.0
        # Get uniques values of random variables
        t_s = time.time()
        values_x = Counter(self.X[x_index])
        values_y = Counter(self.X[y_index])
        # Print debug info
        if debug:
            print('MI between')
            print(self.X[x_index])
            print(self.X[y_index])
        # For each random
        for x in values_x.keys():
            # MI
            posiciones_x = where(self.X[x_index]==x)[0].tolist()
            # We will find if Y_i = y_i when X_i = x_i. Each time
            # this happens, we will remove the position from posiciones_x.
            # When we reach the end of posiciones_x, we update summation
            # and we will look for the y_is of the rest of posiciones_x, and repeat.
            while len(posiciones_x) > 0:
                y = self.X[y_index][posiciones_x[0]]
                cont_y = 1
                del posiciones_x[0]
                i = 0
                while i < len(posiciones_x):
                    z = self.X[y_index][posiciones_x[i]]
                    if y == z:
                        cont_y += 1
                        del posiciones_x[i]
                    else:
                        i += 1
                pxy = cont_y / self.n_cols
                # Single Entropies
                px = values_x.get(x) / self.n_cols
                py = values_y.get(y) / self.n_cols
                # Summation
                if pxy > 0.0:
                    summation += pxy * math.log((pxy / (px*py)), log_base)
                if debug:
                    print('(%d,%d) px:%f py:%f pxy:%f' % (value_x, value_y, px, py, pxy))
        print("Tiempo MI:")
        print(time.time() - t_s)
        return summation

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

    def filter_MI(self, num_features = None, subset_indexes = None):
        """
        Greedy maximizing algorithm.
        A Filter Algorithm based on Delta Test.

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

        t_s = 0
        # Number of features deleted (number of iterations in the big loop).
        deleted_features = 0
        # If we need n features, it's the same to N - (number_of_features_already_selected + number_of_features_deleted) == n.
        while (self.n_cols - len(S_pos) - deleted_features) != features:
            t_s = time.time()
            max_mi = -9999
            max_fi = -1
            # For each feature, we copy X, delete the feature (column), for the rest of the features we sum the MI between each feature and the class, then update the maximum. And repeat.
            for fi in F_pos:
                X_aux = delete(arr=X, obj=fi, axis=1)
                # We control the indexes of the rest of the available features.
                F_sub_pos = F_pos[:fi] + [i-1 for i in F_pos[fi+1:]]
                # We calculate the MI: the sum of the MI between the features fj and the class.
                mi = 0.0
                for fj in F_sub_pos:
                    if fi != fj:
                        mi += self.mutual_information_fc(f_index=fj,log_base=2,data=X_aux)
                # We update the maximum.
                if mi > max_mi:
                    max_mi = mi
                    max_fi = fi
            # We delete the feature with the minimum delta.
            X = delete(arr=X, obj=max_fi, axis=1)
            deleted_features += 1
            # We update the indexes to iterate with.
            F_pos = F_pos[:max_fi] + [i-1 for i in F_pos[max_fi+1:]]
            # We update the list that controls the features of the original X.
            del not_excluded_pos[max_fi]

            #print("Min:", min_delta, min_fi, not_excluded_pos)
            #print("Tiempo:",time.time()-t_s)
        # We return the features already selected, and the columns that weren't deleted.
        return S_pos + not_excluded_pos


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
        for i in range(0,n_cols):
            # For each element in the data, we find its Kth nearest neighbor.
            NN_index = self.KVecinoMasCercano(data=X,x_index=i,k=k)
            # We calculate the formula.
            suma += float(self.Y[i] - self.Y[NN_index])**2
        return suma / (2*n_cols)

    def filter_Delta(self, k=1, num_features = None, subset_indexes = None, delta_is_zero = True):
        """
        Greedy minimizing algorithm.
        A Filter Algorithm based on Delta Test.

        Input: k for kth nearest neighbor, num_features for how many features we would like,
        subset_indexes if we'd like to start with some features already selected, delta_is_zero to enable the condition
        to stop if we reach a delta = 0.

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
        while ((self.n_cols - len(S_pos) - deleted_features) != features) and (delta_so_far != 0.0):
            t_s = time.time()
            min_delta = 9999
            min_fi = -1
            # For each feature, we copy X, delete the feature (column), apply delta and update the minimum. And repeat.
            for fi in F_pos:
                X_aux = delete(arr=X, obj=fi, axis=1)
                # We calculate the delta.
                delta = self.TestDelta(data=X_aux,k=k)
                # We update the minimum.
                if delta < min_delta:
                    min_delta = delta
                    min_fi = fi
            # We delete the feature with the minimum delta.
            X = delete(arr=X, obj=min_fi, axis=1)
            deleted_features += 1
            # We update the indexes to iterate with.
            F_pos = F_pos[:min_fi] + [i-1 for i in F_pos[min_fi+1:]]
            # We update the list that controls the features of the original X.
            del not_excluded_pos[min_fi]
            # If this condition is enabled, we update delta_so_far.
            if delta_is_zero:
                delta_so_far = min_delta

            #print("Min:", min_delta, min_fi, not_excluded_pos)
            #print("Tiempo:",time.time()-t_s)
        # We return the features already selected, and the columns that weren't deleted.
        return S_pos + not_excluded_pos


    # ALL TOGETHER #

    # Está mal la parte del Delta todavía
    def filter_MI_Delta(self, k=1, num_features = None, subset_indexes = None):
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
