import numpy as np
import pandas as pd
import random
from itertools import permutations
import multiprocessing as mp
import math


_MULTIPROCESSING = False
_range = xrange


def _filter_nan(a, b):
    a_nan = np.isnan(a).any(axis=1)
    b_nan = np.isnan(b).any(axis=1)
    all_nan = a_nan | b_nan
    return a[~all_nan], b[~all_nan]


def _check_mat_shape(m, col_count=None, row_count=None):
    if len(m.shape) != 2:
        raise ValueError("matrix expected")
    if col_count is not None:
        if m.shape[1] != col_count:
            raise ValueError("invalid matrix size - %s columns needed, %s given" % (col_count, m.shape[1]))
    if row_count is not None:
        if m.shape[0] != row_count:
            raise ValueError("invalid matrix size - %s rows needed, %s given" % (row_count, m.shape[0]))


def _lsq(a, b):
    at = a.T
    x = at.dot(a)
    y = at.dot(b)
    return np.linalg.solve(x, y)


class ModelArg:

    def __init__(self, name):
        self.name = name


class BaseModel:

    input_args = property(lambda self: self.__input_args)
    output_args = property(lambda self: self.__output_args)
    input_names = property(lambda self: self.__input_names)
    output_names = property(lambda self: self.__output_names)
    input_arg_count = property(lambda self: len(self.__input_args))
    output_arg_count = property(lambda self: len(self.__output_args))

    def __init__(self, input_args, output_args):
        self.__input_args = input_args
        self.__output_args = output_args
        self.__input_names = map(lambda t: t.name, input_args)
        self.__output_names = map(lambda t: t.name, output_args)

    def calculate(self, input_df):
        raise NotImplementedError()

    def rename_output_arg(self, old_name, new_name):
        idx = self.__output_names.index(old_name)
        self.__output_args[idx] = ModelArg(new_name)
        self.__output_names[idx] = new_name


class BaseCovModel(BaseModel):

    input_arg0_name = property(lambda self: self.input_args[0].name)
    input_arg1_name = property(lambda self: self.input_args[1].name)

    def __init__(self, input_args, output_args):
        BaseModel.__init__(self, input_args, output_args)
        if self.input_arg_count != 2:
            raise ValueError("covariational models support only two input arguments")


class CovLinModel(BaseCovModel):

    def __init__(self, input_args, output_args, x):
        BaseCovModel.__init__(self, input_args, output_args)
        _check_mat_shape(x, col_count=len(output_args), row_count=3)
        self.__x = x

    def calculate(self, input_df):
        row_count = input_df.shape[0]
        a1 = input_df[self.input_arg0_name]
        a2 = input_df[self.input_arg1_name]
        a = np.ndarray((row_count, 3))
        a[:, 0] = 1
        a[:, 1] = a1
        a[:, 2] = a2
        output = a.dot(self.__x)
        return pd.DataFrame(data=output, columns=self.output_names)

    def __str__(self):
        r = [
            str(self.__x[0, 0]),
            str(self.__x[1, 0]) + "*" + self.input_arg0_name,
            str(self.__x[2, 0]) + "*" + self.input_arg1_name
        ]
        return ",".join(self.output_names) + " = " + " + ".join(r)

    __repr__ = __str__


class CovMulModel(BaseCovModel):

    def __init__(self, input_args, output_args, x):
        BaseCovModel.__init__(self, input_args, output_args)
        _check_mat_shape(x, col_count=len(output_args), row_count=4)
        self.__x = x

    def calculate(self, input_df):
        row_count = input_df.shape[0]
        a1 = input_df[self.input_arg0_name]
        a2 = input_df[self.input_arg1_name]
        a = np.ndarray((row_count, 4))
        a[:, 0] = 1
        a[:, 1] = a1
        a[:, 2] = a2
        a[:, 3] = a1*a2
        output = a.dot(self.__x)
        return pd.DataFrame(data=output, columns=self.output_names)

    def __str__(self):
        r = [
            str(self.__x[0, 0]),
            str(self.__x[1, 0]) + "*" + self.input_arg0_name,
            str(self.__x[2, 0]) + "*" + self.input_arg1_name,
            str(self.__x[3, 0]) + "*" + self.input_arg0_name + "*" + self.input_arg1_name
        ]
        return ",".join(self.output_names) + " = " + " + ".join(r)

    __repr__ = __str__


class CovSqrModel(BaseCovModel):

    def __init__(self, input_args, output_args, x):
        BaseCovModel.__init__(self, input_args, output_args)
        _check_mat_shape(x, col_count=len(output_args), row_count=6)
        self.__x = x

    def calculate(self, input_df):
        row_count = input_df.shape[0]
        a1 = input_df[self.input_arg0_name]
        a2 = input_df[self.input_arg1_name]
        a = np.ndarray((row_count, 6))
        a[:, 0] = 1
        a[:, 1] = a1
        a[:, 2] = a2
        a[:, 3] = a1*a2
        a[:, 4] = a1**2
        a[:, 5] = a2**2
        output = a.dot(self.__x)
        return pd.DataFrame(data=output, columns=self.output_names)

    def __str__(self):
        r = [
            str(self.__x[0, 0]),
            str(self.__x[1, 0]) + "*" + self.input_arg0_name,
            str(self.__x[2, 0]) + "*" + self.input_arg1_name,
            str(self.__x[3, 0]) + "*" + self.input_arg0_name + "*" + self.input_arg1_name,
            str(self.__x[4, 0]) + "*" + self.input_arg0_name + "^2",
            str(self.__x[5, 0]) + "*" + self.input_arg1_name + "^2"
        ]
        return ",".join(self.output_names) + " = " + " + ".join(r)

    __repr__ = __str__


class BaseModelGenerator:

    def __init__(self):
        pass

    def generate(self, input_df, output_df):
        raise NotImplementedError()


class CovLinModelGenerator(BaseModelGenerator):

    def generate(self, input_df, output_df):
        if input_df.shape[1] != 2:
            raise ValueError("two arguments dataframe expected as input_df")
        row_count = input_df.shape[0]
        a1_name = input_df.columns[0]
        a2_name = input_df.columns[1]
        a1 = input_df[a1_name]
        a2 = input_df[a2_name]
        a = np.ndarray((row_count, 3))
        a[:, 0] = 1
        a[:, 1] = a1
        a[:, 2] = a2
        roots = _lsq(a, output_df)
        return CovLinModel([ModelArg(a1_name), ModelArg(a2_name)], [ModelArg(a) for a in output_df.columns], roots)


class CovMulModelGenerator(BaseModelGenerator):

    def generate(self, input_df, output_df):
        if input_df.shape[1] != 2:
            raise ValueError("two arguments dataframe expected as input_df")
        row_count = input_df.shape[0]
        a1_name = input_df.columns[0]
        a2_name = input_df.columns[1]
        a1 = input_df[a1_name]
        a2 = input_df[a2_name]
        a = np.ndarray((row_count, 4))
        a[:, 0] = 1
        a[:, 1] = a1
        a[:, 2] = a2
        a[:, 3] = a1*a2
        roots = _lsq(a, output_df)
        return CovMulModel([ModelArg(a1_name), ModelArg(a2_name)], [ModelArg(a) for a in output_df.columns], roots)


class CovSqrModelGenerator(BaseModelGenerator):

    def generate(self, input_df, output_df):
        if input_df.shape[1] != 2:
            raise ValueError("two arguments dataframe expected as input_df")
        row_count = input_df.shape[0]
        a1_name = input_df.columns[0]
        a2_name = input_df.columns[1]
        a1 = input_df[a1_name]
        a2 = input_df[a2_name]
        a = np.ndarray((row_count, 6))
        a[:, 0] = 1
        a[:, 1] = a1
        a[:, 2] = a2
        a[:, 3] = a1*a2
        a[:, 4] = a1**2
        a[:, 5] = a2**2
        roots = _lsq(a, output_df)
        return CovSqrModel([ModelArg(a1_name), ModelArg(a2_name)], [ModelArg(a) for a in output_df.columns], roots)


def moving_average_filter(matrix, n):
    row_count = matrix.shape[0]
    col_count = 1
    is_vector = True
    if len(matrix.shape) == 2:
        col_count = matrix.shape[1]
        is_vector = False
    elif len(matrix.shape) > 2:
        raise ValueError("only vectors and two-dimension matrices supported")
    if is_vector:
        t = np.cumsum(matrix, dtype=float)
        t[n:] = t[n:] - t[:-n]
        t = t[n-1:]/n
        return t
    else:
        r = np.ndarray((row_count-n+1, col_count))
        for i in _range(col_count):
            t = np.cumsum(matrix[:, i], dtype=float)
            t[n:] = t[n:] - t[:-n]
            t = t[n-1:]/n
            r[:, i] = t
        return r


class LayerOutput:

    model = property(lambda self: self.__model)
    teach_norm = property(lambda self: self.__teach_norm)
    check_norm = property(lambda self: self.__check_norm)
    check_regularity = property(lambda self: self.__check_regularity)
    shift_criterion = property(lambda self: self.__shift_criterion)

    def __init__(self, model, teach_norm, check_norm, check_regularity, shift_criterion):
        self.__model = model
        self.__teach_norm = teach_norm
        self.__check_norm = check_norm
        self.__check_regularity = check_regularity
        self.__shift_criterion = shift_criterion


def calculate_model(a):
    split_df, arg_df, target_df, generator, filter_nan, arg_names = a
    teach_args_df = arg_df[split_df["sample"] == ModelWorkspace.TEACH_SAMPLE_NAME]
    teach_target_df = target_df[split_df["sample"] == ModelWorkspace.TEACH_SAMPLE_NAME]
    check_args_df = arg_df[split_df["sample"] == ModelWorkspace.CHECK_SAMPLE_NAME]
    check_target_df = target_df[split_df["sample"] == ModelWorkspace.CHECK_SAMPLE_NAME]
    if filter_nan:
        teach_args_df, teach_target_df = _filter_nan(teach_args_df, teach_target_df)
        check_args_df, check_target_df = _filter_nan(check_args_df, check_target_df)
    try:
        model = generator.generate(teach_args_df, teach_target_df)
        check_model = generator.generate(check_args_df, check_target_df)
    except np.linalg.linalg.LinAlgError:
        # print("cannot find solution for pair %s" % str(cov_indexes))
        return None
    output_teach = model.calculate(teach_args_df)
    output_check = model.calculate(check_args_df)
    check_output_teach = check_model.calculate(teach_args_df)
    check_output_check = check_model.calculate(check_args_df)
    check_target_sq_sum = float((check_target_df**2).sum())
    # calculate error norms
    norm_teach = math.sqrt(((output_teach-teach_target_df)**2).sum())
    check_diffs = float(((output_check-check_target_df)**2).sum())
    norm_check = math.sqrt(check_diffs)
    # calculate regularity
    check_regularity = check_diffs / check_target_sq_sum
    # calculate shift criterion
    shift_teach_part = float(((output_teach - check_output_teach)**2).sum())
    shift_check_part = float(((output_check - check_output_check)**2).sum())
    target_sq_sum = float((teach_target_df**2).sum()) + check_target_sq_sum
    shift_criterion = (shift_teach_part + shift_check_part)/target_sq_sum
    return LayerOutput(model, norm_teach, norm_check, check_regularity, shift_criterion)


def collect_leaf_args(root_model, model_resolver):
    result = {}
    for arg in root_model.input_args:
        model = model_resolver(arg.name)
        if model is None:
            result[arg.name] = arg
        else:
            result.update(collect_leaf_args(model, model_resolver))
    return result


def collect_models(root_model, model_resolver):
    result = {}
    for arg in root_model.input_args:
        model = model_resolver(arg.name)
        if model is not None:
            result[arg.name] = model
            result.update(collect_models(model, model_resolver))
    return result


def pretty_repr_model(root_model, model_resolver, tab=0):
    tab = " "*tab
    R = tab + str(root_model)
    for arg in root_model.input_args:
        model = model_resolver(arg.name)
        if model is not None:
            R = R + "\n" + tab + pretty_repr_model(model, model_resolver, 2)
    return R


def recurse_calculate(root_model, model_resolver, df):
    for arg in root_model.input_names:
        model = model_resolver(arg)
        if model is not None:
            recurse_calculate(model, model_resolver, df)
    output = root_model.calculate(df)
    for arg in root_model.output_names:
        df[arg] = output[arg]


class InductiveModel(BaseModel):

    root_model = property(lambda self: self.__root_model)
    models = property(lambda self: self.__models)

    def __init__(self, output_args, root_model, model_arg_resolver):
        leaf_args = collect_leaf_args(root_model, model_arg_resolver)
        models = collect_models(root_model, model_arg_resolver)
        rpr = pretty_repr_model(root_model, model_arg_resolver)
        BaseModel.__init__(self, leaf_args.values(), output_args)
        self.__root_model = root_model
        self.__models = models
        self.__rpr = rpr

    def calculate(self, input_df):
        def model_resolver(arg_name):
            return self.__models.get(arg_name)
        df = input_df.copy()
        recurse_calculate(self.__root_model, model_resolver, df)
        return df

    def __str__(self):
        return self.__rpr

    __repr__ = __str__



class ModelWorkspace:

    TEACH_SAMPLE_NAME = "teach"
    CHECK_SAMPLE_NAME = "check"

    df = property(lambda self: self.__df)

    def __init__(self, df):
        self.__df = df
        self.__model_counter = 0
        self.__df["sample"] = [ModelWorkspace.TEACH_SAMPLE_NAME for i in _range(self.__df.shape[0])]
        self.__layer_item_by_name = {}
        self.__best_layer_item = None
        self.__layer_arg_by_name = {}

    def add_moving_avg(self, arg_name, n, new_arg_name=None):
        if new_arg_name is None:
            new_arg_name = "%s_mavg_%s" % (arg_name, n)
        self.__df[new_arg_name] = np.nan
        new_arg_col_index = self.__df.columns.get_loc(new_arg_name)
        self.__df.iloc[n-1:, new_arg_col_index] = moving_average_filter(self.__df[arg_name].values, n)

    def add_lag(self, arg_name, shift=1, new_arg_name=None):
        if new_arg_name is None:
            new_arg_name = "%s_lag_%s" % (arg_name, shift)
        self.__df[new_arg_name] = np.nan
        new_arg_col_index = self.__df.columns.get_loc(new_arg_name)
        if shift > 0:
            self.__df.iloc[shift:, new_arg_col_index] = self.__df[arg_name].values[:-shift]
        elif shift < 0:
            self.__df.iloc[:shift, new_arg_col_index] = self.__df[arg_name].values[-shift:]
        else:
            self.__df[new_arg_name] = self.__df[arg_name]

    def add_zscore(self, arg_name, new_arg_name=None):
        if new_arg_name is None:
            new_arg_name = "%s_zscore" % arg_name
        v = self.__df[arg_name]
        mean = v.mean()
        deviation = v.std()
        v = v - mean
        v = v / deviation
        self.__df[new_arg_name] = v

    def split_subsamples_random(self, teach_ratio=0.5):
        sample_col = self.__df.columns.get_loc("sample")
        for i in _range(self.__df.shape[0]):
            if random.random() > teach_ratio:
                self.__df.iloc[i, sample_col] = ModelWorkspace.CHECK_SAMPLE_NAME
            else:
                self.__df.iloc[i, sample_col] = ModelWorkspace.TEACH_SAMPLE_NAME

    def split_subsamples_odd(self):
        sample_col = self.__df.columns.get_loc("sample")
        self.__df.iloc[::2, sample_col] = ModelWorkspace.TEACH_SAMPLE_NAME
        self.__df.iloc[1::2, sample_col] = ModelWorkspace.CHECK_SAMPLE_NAME

    def build_models(self, generator_list, result_names, arg_names=None, filter_nan=True, best_model_count=None, criterion=lambda x: x.shift_criterion):
        if arg_names is None:
            non_arg_names = set(["sample"]).union(set(result_names))
            arg_names = filter(lambda x: x not in non_arg_names, self.__df.columns)
        col_count = len(arg_names)
        calc_args = []
        target_df = self.__df[result_names]
        split_df = self.__df[["sample"]]
        for cov_indexes in permutations(range(0, col_count), 2):
            for generator in generator_list:
                a1 = arg_names[cov_indexes[0]]
                a2 = arg_names[cov_indexes[1]]
                arg_df = self.__df[[a1, a2]]
                calc_args.append((split_df, arg_df, target_df, generator, filter_nan, [a1, a2]))
        if _MULTIPROCESSING:
            pool = mp.Pool(mp.cpu_count())
            calc_result = pool.map(calculate_model, calc_args)
            pool.close()
        else:
            calc_result = map(calculate_model, calc_args)
        calc_result = filter(lambda x: x is not None, calc_result)
        if best_model_count is not None:
            calc_result.sort(key=criterion)
            calc_result = calc_result[:best_model_count]
        return calc_result

    def add_layer_items(self, name_prefix, layer_items, criterion=lambda x: x.shift_criterion):
        result = []
        for item in layer_items:
            out = item.model.calculate(self.__df)
            self.__model_counter += 1
            for col in out.columns:
                col_name = "%s%i_%s" % (name_prefix, self.__model_counter, col)
                item.model.rename_output_arg(col, col_name)
                self.__df[col_name] = out[col]
                self.__layer_item_by_name[col_name] = item
                if self.__best_layer_item is None:
                    self.__best_layer_item = item
                elif criterion(self.__best_layer_item) > criterion(item):
                    self.__best_layer_item = item
                self.__layer_arg_by_name[col_name] = ModelArg(col_name)
                result.append(col_name)
        return result

    def extract_inductive_model(self):
        def resolver(arg_name):
            layer_item = self.__layer_item_by_name.get(arg_name)
            if layer_item:
                return layer_item.model
            return None
        return InductiveModel(self.__best_layer_item.model.output_args, self.__best_layer_item.model, resolver)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    N = 150
    target = "a4"

    d = np.ndarray((N, 6))
    d[:, 0] = np.arange(N)
    d[:, 1] = np.sin(d[:, 0]/200)
    d[:, 2] = d[:, 0]**0.5
    d[:, 3] = np.exp(d[:, 0]/20)
    d[:, 4] = np.cos(d[:, 0]/25)
    d[:, 5] = 10.0 / (d[:, 0]+10)

    #d = np.random.random((N, 6))
    all_columns = ["a%i" % i for i in _range(6)]
    base_args = filter(lambda x: x != target, all_columns)

    df = pd.DataFrame(data=d, columns=all_columns)
    w = ModelWorkspace(df)
    for b in base_args:
        w.add_lag(b)
        w.add_lag(b, 2)

    generators = [CovLinModelGenerator(), CovMulModelGenerator(), CovSqrModelGenerator()]
    w.split_subsamples_random()

    err = None
    layer = 1
    prev_layer_args = []
    layer_args = base_args
    while True:
        layer_items = w.build_models(generators, [target], arg_names=layer_args, best_model_count=10)
        if not layer_items:
            print("no more models produced")
            break
        best_error = layer_items[0].shift_criterion
        if err is not None:
            if best_error > err:
                print("error stopped falling")
                break
            else:
                err = best_error
        else:
            err = best_error
        print "Layer %s - err %s, arg count=%s, feature count=%s" % (layer, best_error, len(layer_args), w.df.shape[1])
        print "regularity - ", layer_items[0].check_regularity
        prev_layer_args = layer_args
        layer_args = w.add_layer_items("L%s_" % layer, layer_items)

        layer += 1
        if layer >= 5:
            print("stopping, too many layers")
            break
    rm = w.extract_inductive_model()
    print "most important args -", rm.input_names
    print rm
    net_out = rm.calculate(df[map(lambda x: x.name, rm.input_args)])
    net_out[target] = df[target]

    for c in net_out.columns:
        if c.startswith("L"):
            f, ax = plt.subplots()
            ax.plot(df[target], label=target)
            ax.plot(net_out[c], label=c)
            ax.legend()
            f.show()
