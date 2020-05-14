
import numpy as np
import math

MAX_HARMONICS = 21

CosPolyTable = [
    [  1,   0,    0,     0,     0,      0,      0,       0,       0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0],
    [  0,   1,    0,     0,     0,      0,      0,       0,       0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0],
    [ -1,   0,    2,     0,     0,      0,      0,       0,       0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0],
    [  0,  -3,    0,     4,     0,      0,      0,       0,       0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0],
    [  1,   0,   -8,     0,     8,      0,      0,       0,       0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0],
    [  0,   5,    0,   -20,     0,     16,      0,       0,       0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0],
    [ -1,   0,   18,     0,   -48,      0,     32,       0,       0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0],
    [  0,  -7,    0,    56,     0,   -112,      0,      64,       0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0],
    [  1,   0,  -32,     0,   160,      0,   -256,       0,     128,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0],
    [  0,   9,    0,  -120,     0,    432,      0,    -576,       0,         256,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0],
    [ -1,   0,   50,     0,  -400,      0,   1120,       0,   -1280,           0,         512,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0],
    [  0, -11,    0,   220,     0,  -1232,      0,    2816,       0,       -2816,           0,        1024,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0],
    [  1,   0,  -72,     0,   840,      0,  -3584,       0,    6912,           0,       -6144,           0,        2048,           0,           0,           0,           0,           0,           0,           0,           0,           0],
    [  0,  13,    0,  -364,     0,   2912,      0,   -9984,       0,       16640,           0,      -13312,           0,        4096,           0,           0,           0,           0,           0,           0,           0,           0],
    [ -1,   0,   98,     0, -1568,      0,   9408,       0,  -26880,           0,       39424,           0,      -28672,           0,        8192,           0,           0,           0,           0,           0,           0,           0],
    [  0, -15,    0,   560,     0,  -6048,      0,   28800,       0,      -70400,           0,       92160,           0,      -61440,           0,       16384,           0,           0,           0,           0,           0,           0],
    [  1,   0, -128,     0,  2688,      0, -21504,       0,   84480,           0,     -180224,           0,      212992,           0,     -131072,           0,       32768,           0,           0,           0,           0,           0],
    [  0,  17,    0,  -816,     0,  11424,      0,  -71808,       0,      239360,           0,     -452608,           0,      487424,           0,     -278528,           0,       65536,           0,           0,           0,           0],
    [ -1,   0,  162,     0, -4320,      0,  44352,       0, -228096,           0,      658944,           0,    -1118208,           0,     1105920,           0,     -589824,           0,      131072,           0,           0,           0],
    [  0, -19,    0,  1140,     0, -20064,      0,  160512,       0,     -695552,           0,     1770496,           0,    -2723840,           0,     2490368,           0,    -1245184,           0,      262144,           0,           0],
    [  1,   0, -200,     0,  6600,      0, -84480,       0,  549120,           0,    -2050048,           0,     4659200,           0,    -6553600,           0,     5570560,           0,    -2621440,           0,      524288,           0],
    [  0,  21,    0, -1540,     0,  33264,      0, -329472,       0,     1793792,           0,    -5870592,           0,    12042240,           0,   -15597568,           0,    12386304,           0,    -5505024,           0,     1048576],
]


def print_harmonic_table(harmonic_count=MAX_HARMONICS):
    def NextPoly(poly_1, poly_2):
        m = max(poly_1.keys())+1
        R = {}
        for i in range(m):
            R[i] = 2*poly_1[i]
        T = {}
        for k in R.keys():
            T[k+1] = R[k]
        T[0] = 0
        for k in poly_2.keys():
            T[k] -= poly_2[k]
        return T

    R2 = {1: 1, 0: 0}
    R1 = {2: 2, 1: 0, 0: -1}

    for i in range(harmonic_count):
        print("[",)
        for j in range(max(R1.keys())+1):
            print("%10i, " % R1[j],)
        for j in range(20-max(R1.keys())+1):
            print("%10i, " % 0,)
        print("], ",)
        tmp = R1
        R1 = NextPoly(R1, R2)
        R2 = tmp


def check_harmonic_params(data, harmonic_count):
    if len(data.shape) > 1:
        raise ValueError("vector required (len(shape) == 1)")
    if harmonic_count <= 0:
        raise ValueError("harmonic_count parameter must be positive")
    if harmonic_count > MAX_HARMONICS:
        raise ValueError("too many harmonics: %s, max is %s" % (harmonic_count, MAX_HARMONICS))
    input_row_count = data.shape[0]
    if input_row_count < harmonic_count*3 + 1:
        raise ValueError("too few input points for this number of harmonics. At least %s points needed" % (harmonic_count*3 + 1))


def dscale(data):
    data = data.copy()
    variance = np.var(data)
    deviation = math.sqrt(variance)
    mean = np.mean(data)
    data -= mean
    data /= deviation
    return data, mean, deviation


def create_ap_matrix(data, harmonic_count, deviation_scale=False):
    check_harmonic_params(data, harmonic_count)
    input_row_count = data.shape[0]
    if deviation_scale:
        data = dscale(data)[0]
    row_count = input_row_count - harmonic_count*2
    a = np.ndarray((row_count, harmonic_count))
    b = np.ndarray((row_count, 1))
    for i in range(row_count):
        di = i + harmonic_count
        for p in range(harmonic_count):
            a[i, p] = data[di+p] + data[di-p]
        b[i, 0] = data[di-harmonic_count] + data[di+harmonic_count]
    at = a.T
    a_sq = at.dot(a)
    b_sq = at.dot(b)
    return np.linalg.solve(a_sq, b_sq)


def create_harmonic_poly(data, harmonic_count, deviation_scale=False):
    ap = create_ap_matrix(data, harmonic_count, deviation_scale)
    polynom = np.ndarray((harmonic_count+1,))
    for i in range(harmonic_count+1):
        polynom[i] = CosPolyTable[harmonic_count][i]
    for i in range(harmonic_count):
        for j in range(i+1):
            polynom[j] -= CosPolyTable[i][j] * ap[i, 0]
    return polynom


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def find_freq(data, harmonic_count, deviation_scale=False, moving_avg_win_size=0):
    if moving_avg_win_size:
        data = moving_average(data, moving_avg_win_size)
    poly = create_harmonic_poly(data, harmonic_count, deviation_scale)
    roots = np.polynomial.polynomial.polyroots(poly)
    non_zero_count = 0
    for i in range(harmonic_count):
        c = roots[i]
        if c > 1.0:
            # this may happen if there is too small amount of data
            c = 1.0
        if c < -1.0:
            # this may happen if there is too small amount of data
            c = -1.0
        roots[i] = math.acos(c)
        if abs(roots[i]) >= 1.0e-5:
            non_zero_count += 1
    if non_zero_count != harmonic_count:
        roots.resize((non_zero_count,))
    return roots


def create_harmonic_matrix(row_count, freq):
    harmonic_count = freq.shape[0]
    r = np.ndarray((row_count, harmonic_count*2 + 1))
    r[:, 0] = 1.0
    for i in range(harmonic_count):
        for t in range(row_count):
            r[t, i*2 + 1] = math.sin(t*freq[i])
            r[t, i*2 + 2] = math.cos(t*freq[i])
    return r


class HarmonicParameters:

    amplitudes = property(lambda self: self.__amplitudes)
    freq = property(lambda self: self.__freq)

    def __init__(self, amplitudes, freq):
        self.__amplitudes = amplitudes
        self.__freq = freq

    def calculate(self, time_vector):
        if len(time_vector.shape) > 1:
            raise ValueError("single-dimension time vector expected")
        harmonic_count = self.__freq.shape[0]
        row_count = time_vector.shape[0]
        r = np.zeros((row_count,))
        r[:] = self.__amplitudes[0]
        for i in range(harmonic_count):
            f = self.__freq[i]
            fv = time_vector*f
            sv = np.sin(fv)
            cv = np.cos(fv)
            r += sv*self.__amplitudes[i*2 + 1]
            r += cv*self.__amplitudes[i*2 + 2]
        return r


def find_harmonic_parameters(data, harmonic_count, deviation_scale=False, moving_avg_win_size=0):
    mean = None
    deviation = None
    if deviation_scale:
        data, mean, deviation = dscale(data)
    freq = find_freq(data, harmonic_count, deviation_scale=False, moving_avg_win_size=moving_avg_win_size)
    data_row_count = data.shape[0]
    hm = create_harmonic_matrix(data_row_count, freq)
    hm_t = hm.T
    a = hm_t.dot(hm)
    b = hm_t.dot(data)
    args = np.linalg.solve(a, b)
    if deviation_scale:
        # descale back if needed
        args *= deviation
        args[0] += mean
    return HarmonicParameters(args, freq)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    import random
    import seaborn as sns
    #data = pd.read_csv("./sample.csv")
    mat = np.ndarray((100, 3))
    mat[:, 0] = np.arange(100)
    mat[:, 1] = np.sin(mat[:, 0])
    mat[:, 2] = np.cos(mat[:, 0])
    #mat[:, 3] = mat[:, 0]**2
    data = pd.DataFrame(
        data=mat,
        columns=["t", "sin", "cos"]
    )
    genders = [random.choice(["male", "female"]) for x in xrange(100)]
    data["gender"] = genders
    f, ax = plt.subplots()
    g = sns.FacetGrid(data, col="gender", hue="gender")
    g.map(plt.scatter, "t", "sin", alpha=0.7)
    g.add_legend()
    #g.add_legend()
    #f.show()
    plt.show()
    plt.show()

    """
    import seaborn
    import random
    # print_harmonic_table()
    N = 500
    data = np.ndarray((N, ))
    for i in range(N):
        data[i] = 14 + 10.0*math.cos(0.07*i)+2*math.sin(0.07*i) + 5*math.sin(0.2*i) - 3*math.cos(0.2*i) + random.random()*2
    harm_params = find_harmonic_parameters(data, 6, True, 15)
    print("freq = ", harm_params.freq)
    print("amp  = ", harm_params.amplitudes)

    x = np.arange(0, data.shape[0])
    calculated = harm_params.calculate(np.arange(0, data.shape[0]))
    err = math.sqrt(np.sum((calculated-data)**2))
    print("err  = ", err)
    f, ax = plt.subplots()
    ax.plot(data, label="original")
    ax.plot(calculated, label="calculated")
    ax.set_xlabel('time')  # Add an x-label to the axes.
    ax.set_ylabel('value')  # Add a y-label to the axes.
    ax.set_title("Harmonic")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    f.show()
    """

    """
    plt.plot(x, calculated)
    plt.plot(x, data)

    plt.ylabel('Y values')
    plt.show()
    """
