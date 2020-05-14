#!/usr/bin/python

import numpy as np
from itertools import permutations
import bisect
from math import sqrt
import multiprocessing as mp


class ModelFunc:

    def __init__(self):
        pass

    def calculate(self, arg_matrix):
        raise NotImplementedError()


class Model:

    def __init__(self):
        pass

    def _preprocess_input(self, arg_matrix, target_matrix):
        # find rows with NaN values in either arg or target matrix
        nr = nan_rows(arg_matrix, target_matrix)

        # filter non-nan rows
        return arg_matrix[~nr], target_matrix[~nr]

    def solve(self, arg_matrix, target_matrix):
        raise NotImplementedError()


def nan_rows(a, b):
    a_nan = np.isnan(a).any(axis=1)
    b_nan = np.isnan(b).any(axis=1)
    return a_nan | b_nan


class CovSqrFunc(ModelFunc):

    def __init__(self, weights):
        ModelFunc.__init__(self)
        self.__weights = weights

    def calculate(self, arg_matrix):
        row_count, col_count = arg_matrix.shape
        a = np.ndarray((row_count, col_count+4))
        a[:, 0] = 1
        a[:, 1:3] = arg_matrix
        a[:, 3] = arg_matrix[:, 0]*arg_matrix[:, 1]
        a[:, 4] = arg_matrix[:, 0]**2
        a[:, 5] = arg_matrix[:, 1]**2
        return a.dot(self.__weights)


class CovLinFunc(ModelFunc):

    def __init__(self, weights):
        ModelFunc.__init__(self)
        self.__weights = weights

    def calculate(self, arg_matrix):
        row_count, col_count = arg_matrix.shape
        a = np.ndarray((row_count, col_count+1))
        a[:, 0] = 1
        a[:, 1:3] = arg_matrix
        return a.dot(self.__weights)


class CovLinModel(Model):

    def solve(self, arg_matrix, target_matrix):
        arg, target = self._preprocess_input(arg_matrix, target_matrix)

        # calculate
        row_count, col_count = arg.shape
        a = np.ndarray((row_count, col_count+1))
        a[:, 0] = 1
        a[:, 1:3] = arg
        at = a.T
        A = at.dot(a)
        B = at.dot(target)
        r = np.linalg.solve(A, B)
        return CovLinFunc(r)


class CovSqrModel(Model):

    def solve(self, arg_matrix, target_matrix):
        arg, target = self._preprocess_input(arg_matrix, target_matrix)

        # calculate
        row_count, col_count = arg.shape
        a = np.ndarray((row_count, col_count+4))
        a[:, 0] = 1
        a[:, 1:3] = arg
        a[:, 3] = arg[:, 0]*arg[:, 1]
        a[:, 4] = arg[:, 0]**2
        a[:, 5] = arg[:, 1]**2
        at = a.T
        A = at.dot(a)
        B = at.dot(target)
        r = np.linalg.solve(A, B)
        return CovSqrFunc(r)


class LayerItem:

    arg_indexes = property(lambda self: self.__arg_indexes)
    function = property(lambda self: self.__function)
    norm = property(lambda self: self.__norm)
    output = property(lambda self: self.__output)

    def __init__(self, arg_indexes, function, norm, output):
        self.__arg_indexes = arg_indexes
        self.__function = function
        self.__norm = norm
        self.__output = output

    def __str__(self):
        return "func(%s): %.3f" % (self.__arg_indexes, self.__norm)

    __repr__ = __str__


def calculate_model(a):
    arg_matrix, target_matrix, cov_indexes, model = a
    arg = arg_matrix[:, cov_indexes]
    try:
        func = model.solve(arg, target_matrix)
    except np.linalg.linalg.LinAlgError:
        # print("cannot find solution for pair %s" % str(cov_indexes))
        return None
    output = func.calculate(arg)
    norm = sqrt(((output-target_matrix)**2).sum())
    inductive_item = LayerItem(cov_indexes, func, norm, output)
    return norm, inductive_item


def build_funcs(arg_matrix, target_matrix, model_list, max_items):
    row_count, col_count = arg_matrix.shape
    result = []
    pool = mp.Pool(mp.cpu_count())
    args_seq = [(arg_matrix, target_matrix, cov_indexes, model) for cov_indexes in permutations(range(0, col_count), 2) for model in model_list]
    calc_result = pool.map(calculate_model, args_seq)
    pool.close()

    for item in calc_result:
        if not item:
            # error on processing... skip
            continue
        else:
            norm, inductive_item = item
            bisect.insort(result, (norm, inductive_item))
            if len(result) > max_items:
                del result[-1]
    return result


def gmdh(arg_matrix, target_matrix, model_list, max_layer_items, max_layers):
    layer = 1
    best_err = np.inf
    while layer <= max_layers:
        print("processing layer %s, args: %s" % (layer, arg_matrix.shape[1]))
        layer_funcs = build_funcs(arg_matrix, target_matrix, model_list, max_layer_items)
        if not layer_funcs:
            break
        err_norm, inductive_item = layer_funcs[0]
        if abs(err_norm - best_err) < 1.0e-6:
            print("error stopped falling")
            break
        print("err: %s" % err_norm)
        best_err = err_norm
        layer += 1
        outputs = map(lambda l: l[1].output, layer_funcs)
        arg_matrix = np.hstack(tuple([arg_matrix] + outputs))


if __name__ == '__main__':
    print("running")
    N = 90
    arg = np.random.rand(N, 80)
    target = np.random.rand(N, 2)
    funcs = gmdh(arg, target, [CovLinModel(), CovSqrModel()], 20, 5)
    print funcs

"""
f(x) = A0 + Sum(j=1,m| Aj*sin(x*Wj) + Bj*cos(x*Wj) )

m=1

f(x) = A0 + A1*sin(x*W1) + B1*cos(x*W1)
f(x-1) = A0 + A1*sin((x-1)*W1) + B1*cos((x-1)*W1) =
         A0 + A1*(sin(x*W1)*cos(W1) - cos(x*W1)*sin(W1)) + B1*(cos(x*W1)*cos(W1) + sin(x*W1)*sin(W1))

f(x+1) = A0 + A1*sin((x+1)*W1) + B1*cos((x+1)*W1) =
         A0 + A1*(sin(x*W1)*cos(W1) + cos(x*W1)*sin(W1)) + B1*(cos(x*W1)*cos(W1) - sin(x*W1)*sin(W1))

f(x-1) + f(x+1) = 2*A0 + 2*A1*sin(x*W1)*cos(W1) + 2*B1*cos(x*W1)*cos(W1) =
                  2*A0 + 2*cos(W1)*(A1*sin(x*W1) + B1*cos(x*W1)) = 
                  2*A0 + 2*cos(W1)*(f(x) - A0)
2*cos(W1) = (f(x-1) + f(x+1) - 2*A0) / (f(x) - A0)
2*cos(W1) = (f(x-1) + f(x+1)) / f(x) # since f(x) is shifted on A0, - it does not affect 

==> 

2*cos(p*Wp) = (f(x-p) + f(x+p)) / f(x)
==>
2*f(x)*cos(p*Wp) = f(x-p) + f(x-p)

"""