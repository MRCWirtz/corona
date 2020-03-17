import numpy as np


def logistic_function(x, a, b):
    return a / (1 + np.exp(-b*a*x) * (a / 16))
