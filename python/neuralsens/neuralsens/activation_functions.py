from scipy.special import expit
import numpy as np

try:
    from torch import sigmoid, tanh, eye, diag, zeros

    allow_torch = True
except ImportError:
    print(
        "torch installation could not be found, only numpy activation functions allowed"
    )
    allow_torch = False


def fill_diagonal_torch(arr, x):
    k = range(arr.size(0))
    arr[[k] * len(arr.size())] = x
    return arr


def activation_function(func: str, use_torch: bool = False):
    def identity(x):
        return x

    def relu(x):
        return x * (x > 0)

    if not use_torch or not allow_torch:

        def logistic(x):
            return expit(x)

        def stanh(x):
            return np.tanh(x)

    elif use_torch and allow_torch:

        def logistic(x):
            return sigmoid(x)

        def stanh(x):
            return tanh(x)

    actfunc = {
        "logistic": logistic,
        "sigmoid": logistic,
        "expit": logistic,
        "identity": identity,
        "tanh": stanh,
        "relu": relu,
    }
    return actfunc[func]


def der_activation_function(func: str, use_torch: bool = False):
    if not use_torch or not allow_torch:

        def logistic(x):
            fx = expit(x)
            return np.diag(fx * (1 - fx))

        def identity(x):
            return np.identity(x.shape[0], dtype=int)

        def stanh(x):
            return np.diag(1 - np.tanh(x) ** 2)

        def relu(x):
            return np.diag(x > 0)

    elif use_torch and allow_torch:

        def logistic(x):
            fx = sigmoid(x)
            return diag(fx * (1 - fx)).float()

        def identity(x):
            return eye(x.size(0), dtype=int).float()

        def stanh(x):
            return diag(1 - np.tanh(x) ** 2).float()

        def relu(x):
            return diag(x > 0).float()

    deractfunc = {
        "logistic": logistic,
        "sigmoid": logistic,
        "expit": logistic,
        "identity": identity,
        "tanh": stanh,
        "relu": relu,
    }
    return deractfunc[func]


def der_2_activation_function(func: str, use_torch: bool = False):
    if not use_torch or not allow_torch:

        def logistic(x):
            zeros_3d = np.zeros((x.shape[0], x.shape[0], x.shape[0]), dtype=float)
            fx = expit(x)
            np.fill_diagonal(zeros_3d, fx * (1 - fx) * (1 - 2 * fx))
            return zeros_3d

        def identity(x):
            return np.zeros((x.shape[0], x.shape[0], x.shape[0]), dtype=int)

        def stanh(x):
            zeros_3d = np.zeros((x.shape[0], x.shape[0], x.shape[0]), dtype=float)
            fx = tanh(x)
            np.fill_diagonal(zeros_3d, -2 * fx * (1 - fx ^ 2))
            return zeros_3d

        def relu(x):
            return np.zeros((x.shape[0], x.shape[0], x.shape[0]), dtype=int)

    elif use_torch and allow_torch:

        def logistic(x):
            zeros_3d = zeros((x.size(0), x.size(0), x.size(0))).float()
            fx = sigmoid(x)
            return fill_diagonal_torch(zeros_3d, fx * (1 - fx) * (1 - 2 * fx))

        def identity(x):
            return zeros((x.size(0), x.size(0), x.size(0)), dtype=int)

        def stanh(x):
            zeros_3d = zeros((x.size(0), x.size(0), x.size(0))).float()
            fx = tanh(x)
            return fill_diagonal_torch(zeros_3d, -2 * fx * (1 - fx ^ 2))

        def relu(x):
            return zeros((x.size(0), x.size(0), x.size(0)), dtype=int)

    der2actfunc = {
        "logistic": logistic,
        "sigmoid": logistic,
        "expit": logistic,
        "identity": identity,
        "tanh": stanh,
        "relu": relu,
    }
    return der2actfunc[func]


def der_3_activation_function(func: str, use_torch: bool = False):
    if not use_torch or not allow_torch:

        def logistic(x):
            zeros_4d = np.zeros(
                (x.shape[0], x.shape[0], x.shape[0], x.shape[0]), dtype=float
            )
            fx = expit(x)
            np.fill_diagonal(zeros_4d, fx * (1 - fx) * (1 - 2 * fx) * (1 - 3 * fx))
            return zeros_4d

        def identity(x):
            return np.zeros((x.shape[0], x.shape[0], x.shape[0], x.shape[0]), dtype=int)

        def stanh(x):
            zeros_4d = np.zeros(
                (x.shape[0], x.shape[0], x.shape[0], x.shape[0]), dtype=float
            )
            fx = tanh(x)
            np.fill_diagonal(zeros_4d, -2 * (1 - 4 * fx ** 2 + fx ** 4))
            return zeros_4d

        def relu(x):
            return np.zeros((x.shape[0], x.shape[0], x.shape[0], x.shape[0]), dtype=int)

    elif use_torch and allow_torch:

        def logistic(x):
            zeros_4d = zeros((x.size(0), x.size(0), x.size(0), x.size(0))).float()
            fx = sigmoid(x)
            return fill_diagonal_torch(
                zeros_4d, fx * (1 - fx) * (1 - 2 * fx) * (1 - 3 * fx)
            )

        def identity(x):
            return zeros((x.size(0), x.size(0), x.size(0), x.size(0)), dtype=int)

        def stanh(x):
            zeros_4d = zeros((x.size(0), x.size(0), x.size(0), x.size(0))).float()
            fx = tanh(x)
            return fill_diagonal_torch(zeros_4d, -2 * (1 - 4 * fx ** 2 + fx ** 4))

        def relu(x):
            return zeros((x.size(0), x.size(0), x.size(0), x.size(0)), dtype=int)

    der3actfunc = {
        "logistic": logistic,
        "sigmoid": logistic,
        "expit": logistic,
        "identity": identity,
        "tanh": stanh,
        "relu": relu,
    }
    return der3actfunc[func]
