import pandas as pd
from numpy import triu_indices, meshgrid, logical_not, isnan
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
import seaborn as sns
from adjustText import adjust_text

from neuralsens.activation_functions import (
    activation_function,
    der_activation_function,
    der_2_activation_function,
    der_3_activation_function,
)

plt.style.use("ggplot")


def calculate_first_partial_derivatives_mlp(
    wts,
    bias,
    actfunc,
    X: pd.core.frame.DataFrame,
    dev: str = "cpu",
    sens_origin_layer: int = 0,
    sens_end_layer="last",
    sens_origin_input: bool = True,
    sens_end_input: bool = False,
    use_torch: bool = False,
):
    """
    Calculate the first partial derivatives of a Multi-Layer Perceptron (MLP) model.

    Parameters:
    wts (list of numpy arrays or torch tensors): List of weight matrices for each layer of the MLP.
    bias (list of numpy arrays or torch tensors): List of bias vectors for each layer of the MLP.
    actfunc (list of callable functions): List of activation functions for each layer of the MLP.
    X (pd.core.frame.DataFrame): Input data as a pandas DataFrame.
    dev (str): Device specification, either "cpu" or "gpu". If "gpu" is chosen or use_torch is True, it uses PyTorch for calculations, otherwise, it uses NumPy. Default is "cpu".
    sens_origin_layer (int): The starting layer index for sensitivity calculation. Default is 0.
    sens_end_layer (int or str): The ending layer index for sensitivity calculation. If "last", it calculates sensitivity up to the last layer. Default is "last".
    sens_origin_input (bool, optional): flag to indicate if derivatives shall be calculated from inputs (True)
        of the origin layer or the outputs (False). Defaults to True.
    sens_end_input (bool, optional): flag to indicate if derivatives shall be calculated to inputs (True)
        of the end layer or the outputs (False). Defaults to False.
    use_torch (bool): If True, use PyTorch for calculations even if dev is "cpu". Default is False.

    Returns:
    W (list of numpy arrays or torch tensors): List of weight matrices for each layer of the MLP.
    Z (list of numpy arrays or torch tensors): List of input values for each layer of the MLP.
    O (list of numpy arrays or torch tensors): List of output values for each layer of the MLP.
    D (list of numpy arrays or torch tensors): List of derivatives for each layer of the MLP.
    D_accum (list of numpy arrays or torch tensors): List of accumulated sensitivity values for each calculated layer.
    counter (int): Number of layers between sens_origin_layer and sens_end_layer. It serves to know the final derivatives to return in the jacobian_mlp.
    mlpstr (list of integer): Structure of the mlp model.
    """
    # Import necessary modules based on processor, use torch when processor is gpu or use_torch is True
    if dev == "gpu" or use_torch:
        try:
            from torch import (
                eye as identity,
                vstack,
                ones,
                tensor,
                device,
                matmul,
                from_numpy,
                is_tensor,
                stack,
            )
            from numpy import ndarray, asarray

            def float_array(x, dev="cpu") -> tensor:
                dev = device(dev)
                if isinstance(x, pd.DataFrame):
                    return from_numpy(x.to_numpy()).float().to(dev)
                if isinstance(x, list):
                    if isinstance(x[0], ndarray):
                        x = asarray(x)
                    elif is_tensor(x[0]):
                        return stack(x).float().to(dev)
                if isinstance(x, ndarray):
                    return from_numpy(x).float().to(dev)
                if is_tensor(x):
                    return x.clone().detach().to(dev)
                return tensor(x, device=dev)

            use_torch = True

        except ImportError:
            from numpy import identity, vstack, array, ones, matmul

            def float_array(x, dev="cpu") -> array:
                return array(x.numpy())

            use_torch = False

    else:
        from numpy import identity, vstack, array, ones, array, matmul

        def float_array(x, dev="cpu") -> array:
            return array(x)

        use_torch = False

    ### Initialize all the necessary variables
    # Structure of the mlp model
    mlpstr = [wts[0].shape[0]] + [lyr.shape[1] for lyr in wts]

    # Derivative and activation functions for each neuron layer
    deractfunc = [der_activation_function(af, use_torch) for af in actfunc]
    actfunc = [activation_function(af, use_torch) for af in actfunc]

    # Number of samples to be cached (it is used several times)
    n_samples = X.shape[0]
    r_samples = range(n_samples)

    # Weights of input layer
    W = [identity(X.shape[1])]

    # Input of input layer
    # inputs = [np.hstack((ones((len(X_train),1), dtype=int), X_train))]
    Z = [matmul(float_array(X, dev), float_array(W[0], dev))]

    # Output of input layer
    O = [actfunc[0](Z[0])]

    # Derivative of input layer
    D = [float_array([deractfunc[0](Z[0][irow,]) for irow in r_samples], dev=dev)]

    # Let's go over all the layers calculating each variable
    for lyr in range(1, len(mlpstr)):
        # Calculate weights of each layer
        W.append(float_array(vstack((bias[lyr - 1], wts[lyr - 1])), dev=dev))

        # Calculate input of each layer
        # Add columns of 1 for the bias
        aux = ones((O[lyr - 1].shape[0], O[lyr - 1].shape[1] + 1))
        aux[:, 1:] = O[lyr - 1]
        Z.append(matmul(float_array(aux, dev), float_array(W[lyr], dev)))

        # Calculate output of each layer
        O.append(actfunc[lyr](Z[lyr]))

        # Calculate derivative of each layer
        D.append(
            float_array([deractfunc[lyr](Z[lyr][irow,]) for irow in r_samples], dev=dev)
        )

    # Now, let's calculate the derivatives of interest
    D_accum = [identity(mlpstr[sens_origin_layer]) for irow in r_samples]
    if sens_origin_input:
        D_accum = [D[sens_origin_layer]]

    counter = 0
    # Only perform further operations if origin is not equal to end layer
    if not (sens_origin_layer == sens_end_layer):
        for layer in range(sens_origin_layer + 1, sens_end_layer):
            counter += 1
            # Calculate the derivatives of the layer based on the previous and the weights
            if (layer == sens_end_layer) and sens_end_input:
                D_accum.append(D_accum[counter - 1] @ W[layer][1:, :])
            else:
                D_accum.append(D_accum[counter - 1] @ W[layer][1:, :] @ D[layer])
    return W, Z, O, D, D_accum, counter, mlpstr


def jacobian_mlp(
    wts,
    bias,
    actfunc,
    X: pd.core.frame.DataFrame,
    y: pd.core.frame.DataFrame,
    sens_origin_layer: int = 0,
    sens_end_layer="last",
    sens_origin_input: bool = True,
    sens_end_input: bool = False,
    dev: str = "cpu",
    use_torch: bool = False,
):
    """Obtain first derivatives of MLP model given an input space.

    First derivatives of MLP model may be used to analyze the variable
    relationships between inputs and outputs.

    Args:
        wts (list[float]): list of weight matrixes of the MLP layers.
        bias (list[float]): list of bias vectors of the MLP layers.
        actfunc (list[str]): list of names of the activation function of the MLP layers.
        X (pd.core.frame.DataFrame): data.frame with the input data
        y (pd.core.frame.DataFrame): data.frame with the output data
        sens_origin_layer (int, optional): layer from where the derivatives shall be calculated.
            Defaults to 0 (input layer).
        sens_end_layer (str | int, optional): layer to where the derivatives shall be calculated.
            Defaults to "last" (output layer).
        sens_origin_input (bool, optional): flag to indicate if derivatives shall be calculated from inputs (True)
            of the origin layer or the outputs (False). Defaults to True.
        sens_end_input (bool, optional): flag to indicate if derivatives shall be calculated to inputs (True)
            of the end layer or the outputs (False). Defaults to False.
        dev (str, optional): device where calculations shall be performed ("gpu" or "cpu").
            Only available if pytorch installation could be found. Defaults to "cpu".
        use_torch (bool, optional): Flag to indicate if pytorch Tensor shall be used. Defaults to False.

    Raises:
        ValueError: End layer to analyze cannot be smaller or equal to zero.
        ValueError: Origin layer to analyze cannot be smaller or equal to zero.
        ValueError: There must be a layer of neurons or weights between end layer and origin layer.
        ValueError: Origin layer should be less than number of layers in the model.
        ValueError: End layer should be less than number of layers in the model.

    Returns:
        Jacobian_MLP: custom object storing the first partial derivatives of the MLP model.
    """
    # Import necessary modules based on processor, use torch when processor is gpu or use_torch is True
    if dev == "gpu" or use_torch:
        try:
            from torch import (
                mean,
                std,
                tensor,
                device,
                square,
                from_numpy,
                is_tensor,
                stack,
                sqrt,
            )
            from numpy import ndarray, asarray

            def float_array(x, dev="cpu") -> tensor:
                dev = device(dev)
                if isinstance(x, pd.DataFrame):
                    return from_numpy(x.to_numpy()).float().to(dev)
                if isinstance(x, list):
                    if isinstance(x[0], ndarray):
                        x = asarray(x)
                    elif is_tensor(x[0]):
                        return stack(x).float().to(dev)
                if isinstance(x, ndarray):
                    return from_numpy(x).float().to(dev)
                if is_tensor(x):
                    return x.clone().detach().to(dev)
                return tensor(x, device=dev)

            use_torch = True

        except ImportError:
            from numpy import array, mean, std, square, sqrt

            def float_array(x, dev="cpu") -> array:
                return array(x.numpy())

            use_torch = False

    else:
        from numpy import array, mean, std, square, sqrt

        def float_array(x, dev="cpu") -> array:
            return array(x)

        use_torch = False

    # Check validity of inputs
    if sens_end_layer == "last":
        sens_end_layer = len(actfunc)

    if sens_end_layer <= 0:
        raise ValueError("End layer to analyze cannot be smaller or equal to zero.")

    if sens_origin_layer < 0:
        raise ValueError("Origin layer to analyze cannot be smaller or equal to zero.")

    if not (sens_end_layer > sens_origin_layer) or (
        (sens_end_layer == sens_origin_layer)
        and (sens_origin_input and not sens_end_input)
    ):
        raise ValueError(
            "There must be a layer of neurons or weights between end layer and origin layer."
        )

    if sens_origin_layer > len(actfunc):
        raise ValueError(
            "Origin layer should be less than number of layers in the model."
        )

    if sens_end_layer > len(actfunc):
        raise ValueError("End layer should be less than number of layers in the model.")

    _, _, _, _, D_accum, counter, mlpstr = calculate_first_partial_derivatives_mlp(
        wts,
        bias,
        actfunc,
        X,
        dev,
        sens_origin_layer,
        sens_end_layer,
        sens_origin_input,
        sens_end_input,
        use_torch,
    )

    # Calculate sensitivity measures for each input and output
    meanSens = mean(D_accum[counter], axis=0)
    stdSens = std(D_accum[counter], axis=0)
    meansquareSens = sqrt(mean(square(D_accum[counter]), axis=0))

    # Store the information extracted from sensitivity analysis
    input_name = X.columns.values.tolist()
    sens = [
        pd.DataFrame(
            {
                "mean": float_array(meanSens[:, icol], "cpu"),
                "std": float_array(stdSens[:, icol], "cpu"),
                "mean_squared": float_array(meansquareSens[:, icol], "cpu"),
            },
            index=input_name,
        )
        for icol in range(meanSens.shape[1])
    ]
    raw_sens = [
        pd.DataFrame(
            float_array(D_accum[counter][:, :, out], "cpu"),
            index=range(X.shape[0]),
            columns=input_name,
        )
        for out in range(D_accum[counter].shape[2])
    ]

    # Create output name for creating self
    output_name = y.columns.to_list()
    if D_accum[counter].shape[2] > 1:
        output_name = ["_".join([y.name, lev]) for lev in y.unique()]
    return Jacobian_mlp(sens, raw_sens, mlpstr, X, input_name, output_name)


# Define self class
class Jacobian_mlp:
    """A class used to store first partial derivatives of an MLP model.

    This class and its analogue for second partial derivatives are the main
    instruments of the package. It stores not only the partial derivatives but
    also the sensitivity metrics useful to interpret the input-output relationship.
    Methods of this class are used to interpret the information given by
    the first partial derivatives. It is not intended to be created outside the
    jacobian_mlp() function.

    Attributes:
        sens (list[]): list of sensitivity metrics stored in pandas DataFrames for
            each of the outputs.
        raw_sens (list[]): list of partial derivatives matrix for each of the outputs.
        mlp_struct (list[int]): structure of the neural network as a list of neurons
            per layer.
        X (pd.DataFrame): Dataframe with the input variable samples used to calculate
            the partial derivatives stored in raw_sens.
        input_names (list[str]): name of the input variables.
        output_name (list[str]): name of the output variables.

    Methods:
        summary(): print sensitivity measures
        info(): print partial derivatives
        sensitivityPlots(): plot sensitivity measures
        featurePlots(): plot partial derivatives in a SHAP-alike manner
        timePlots(): plot partial derivatives with respect a time variable

    """

    def __init__(
        self,
        sens: list,
        raw_sens: list,
        mlp_struct,
        X: pd.core.frame.DataFrame,
        input_name,
        output_name,
    ):
        """Constructs all the necessary attributes for the Jacobian_mlp object.

        Args:
            sens (list[]): list of sensitivity metrics stored in pandas DataFrames for
                each of the outputs.
            raw_sens (list[]): list of partial derivatives matrix for each of the outputs.
            mlp_struct (list[int]): structure of the neural network as a list of neurons
                per layer.
            X (pd.DataFrame): Dataframe with the input variable samples used to calculate
                the partial derivatives stored in raw_sens.
            input_names (list[str]): name of the input variables.
            output_name (list[str]): name of the output variables.
        """
        self.__sens = sens
        self.__raw_sens = raw_sens
        self.__mlp_struct = mlp_struct
        self.__X = X
        self.__input_name = input_name
        self.__output_name = output_name

    @property
    def sens(self):
        return self.__sens

    @property
    def raw_sens(self):
        return self.__raw_sens

    @property
    def mlp_struct(self):
        return self.__mlp_struct

    @property
    def X(self):
        return self.__X

    @property
    def input_name(self):
        return self.__input_name

    @property
    def output_name(self):
        return self.__output_name

    def __repr__(self) -> str:
        return f"<Jacobian of {self.mlp_struct} MLP network with inputs {self.input_name} and output {self.output_name}>"

    def __str__(self):
        self.summary()
        return ""

    def summary(self):
        """Prints the sensitivity measures."""
        print("Sensitivity analysis of", str(self.mlp_struct), "MLP network.\n")
        print("Sensitivity measures of each output:\n")
        for out in range(len(self.sens)):
            print("$" + self.output_name[out], "\n")
            print(self.sens[out])

    def info(self, n=5):
        """Prints the partial derivatives

        Args:
            n (int, optional): number of partial derivatives to display. Defaults to 5.
        """
        print("Sensitivity analysis of", str(self.mlp_struct), "MLP network.\n")
        print(self.X.shape[0], "samples\n")
        print(
            "Sensitivities of each output (only ",
            min([n, self.raw_sens[0].shape[0]]),
            " first samples):\n",
            sep="",
        )
        for out in range(len(self.raw_sens)):
            print("$" + self.output_name[out], "\n")
            print(self.raw_sens[out][: min([n, self.raw_sens[out].shape[0]])])

    def plot(self, type="sens", **kwargs):
        """Generate plot based on partial derivatives.

        Args:
            type (str, optional): Type of plot to generate, accepted options are "sens", "features" and "time.
                Defaults to "sens".
        """
        if type == "sens":
            self.sensitivityPlots(**kwargs)
        elif type == "features":
            self.featurePlots(**kwargs)
        elif type == "time":
            self.timePlots(**kwargs)
        else:
            print("The specifyed type", type, "is not an accepted plot type.")

    def sensitivityPlots(self, **kwargs):
        sensitivity_plots(self, **kwargs)

    def featurePlots(self):
        pass

    def timePlots(self):
        pass


def calculate_second_partial_derivatives_mlp(
    wts,
    bias,
    actfunc,
    X: pd.core.frame.DataFrame,
    dev: str = "cpu",
    sens_origin_layer: int = 0,
    sens_end_layer="last",
    sens_origin_input: bool = True,
    use_torch: bool = False,
):
    """
    Calculate the first partial derivatives of a Multi-Layer Perceptron (MLP) model.

    Parameters:
    wts (list of numpy arrays or torch tensors): List of weight matrices for each layer of the MLP.
    bias (list of numpy arrays or torch tensors): List of bias vectors for each layer of the MLP.
    actfunc (list of callable functions): List of activation functions for each layer of the MLP.
    X (pd.core.frame.DataFrame): Input data as a pandas DataFrame.
    dev (str): Device specification, either "cpu" or "gpu". If "gpu" is chosen or use_torch is True, it uses PyTorch for calculations, otherwise, it uses NumPy. Default is "cpu".
    sens_origin_layer (int): The starting layer index for sensitivity calculation. Default is 0.
    sens_end_layer (int or str): The ending layer index for sensitivity calculation. If "last", it calculates sensitivity up to the last layer. Default is "last".
    sens_origin_input (bool, optional): flag to indicate if derivatives shall be calculated from inputs (True)
        of the origin layer or the outputs (False). Defaults to True.
    use_torch (bool): If True, use PyTorch for calculations even if dev is "cpu". Default is False.

    Returns:
    W (list of numpy arrays or torch tensors): List of weight matrices for each layer of the MLP.
    Z (list of numpy arrays or torch tensors): List of input values for each layer of the MLP.
    O (list of numpy arrays or torch tensors): List of output values for each layer of the MLP.
    D (list of numpy arrays or torch tensors): List of derivatives for each layer of the MLP.
    D2 (list of numpy arrays or torch tensors): List of second derivatives for each layer of the MLP.
    D_accum (list of numpy arrays or torch tensors): List of accumulated sensitivity values for each calculated layer.
    Q (list of numpy arrays or torch tensors): List of accumulated second partial derivatives of the input of the layer with respect to the inputs of the model.
    H (list of numpy arrays or torch tensors): List of accumulated second partial derivatives of the output of the layer with respect to the inputs of the model.
    counter (int): Number of layers between sens_origin_layer and sens_end_layer. It serves to know the final derivatives to return in the jacobian_mlp.
    mlpstr (list of integer): Structure of the mlp model.
    """
    # Import necessary modules based on processor, use torch when processor is gpu or use_torch is True
    if dev == "gpu" or use_torch:
        try:
            from torch import (
                eye as identity,
                vstack,
                ones,
                tensor,
                device,
                matmul,
                from_numpy,
                is_tensor,
                stack,
                zeros,
            )
            from numpy import ndarray, asarray

            def float_array(x, dev="cpu") -> tensor:
                dev = device(dev)
                if isinstance(x, pd.DataFrame):
                    return from_numpy(x.to_numpy()).float().to(dev)
                if isinstance(x, list):
                    if isinstance(x[0], ndarray):
                        x = asarray(x)
                    elif is_tensor(x[0]):
                        return stack(x).float().to(dev)
                if isinstance(x, ndarray):
                    return from_numpy(x).float().to(dev)
                if is_tensor(x):
                    return x.clone().detach().to(dev)
                return tensor(x, device=dev)

            use_torch = True

        except ImportError:
            from numpy import identity, vstack, array, ones, matmul, zeros

            def float_array(x, dev="cpu") -> array:
                return array(x.numpy())

            use_torch = False

    else:
        from numpy import identity, vstack, array, ones, array, matmul, zeros

        def float_array(x, dev="cpu") -> array:
            return array(x)

        use_torch = False

    ### Initialize all the necessary variables
    # Structure of the mlp model
    mlpstr = [wts[0].shape[0]] + [lyr.shape[1] for lyr in wts]

    # Derivative and activation functions for each neuron layer
    der2actfunc = [der_2_activation_function(af, use_torch) for af in actfunc]
    deractfunc = [der_activation_function(af, use_torch) for af in actfunc]
    actfunc = [activation_function(af, use_torch) for af in actfunc]

    # Number of samples to be cached (it is used several times)
    n_samples = X.shape[0]
    r_samples = range(n_samples)
    n_inputs = X.shape[1]

    # Weights of input layer
    W = [identity(X.shape[1])]

    # Input of input layer
    # inputs = [np.hstack((ones((len(X_train),1), dtype=int), X_train))]
    Z = [matmul(float_array(X, dev), float_array(W[0], dev))]

    # Output of input layer
    O = [actfunc[0](Z[0])]

    # Derivative of input layer
    D = [float_array([deractfunc[0](Z[0][irow,]) for irow in r_samples], dev=dev)]

    # Second derivative of input layer
    D2 = [float_array([der2actfunc[0](Z[0][irow,]) for irow in r_samples], dev=dev)]

    # Let's go over all the layers calculating each variable
    for lyr in range(1, len(mlpstr)):
        # Calculate weights of each layer
        W.append(float_array(vstack((bias[lyr - 1], wts[lyr - 1])), dev=dev))

        # Calculate input of each layer
        # Add columns of 1 for the bias
        aux = ones((O[lyr - 1].shape[0], O[lyr - 1].shape[1] + 1))
        aux[:, 1:] = O[lyr - 1]
        Z.append(matmul(float_array(aux, dev), float_array(W[lyr], dev)))

        # Calculate output of each layer
        O.append(actfunc[lyr](Z[lyr]))

        # Calculate derivative of each layer
        D.append(
            float_array([deractfunc[lyr](Z[lyr][irow,]) for irow in r_samples], dev=dev)
        )

        # Calculate second derivative of each layer
        D2.append(
            float_array(
                [der2actfunc[lyr](Z[lyr][irow,]) for irow in r_samples], dev=dev
            )
        )

    # Now, let's calculate the derivatives of interest
    if sens_end_layer == "last":
        sens_end_layer = len(actfunc)

    # Initialize derivatives
    D_accum = [identity(mlpstr[sens_origin_layer]) for irow in r_samples]
    if sens_origin_input:
        D_accum = [D[sens_origin_layer]]
    Q = [zeros((n_samples, n_inputs, n_inputs, n_inputs))]
    H = [D2[sens_origin_layer]]

    counter = 0
    # Only perform further operations if origin is not equal to end layer
    if not (sens_origin_layer == sens_end_layer):
        for layer in range(sens_origin_layer + 1, sens_end_layer):
            counter += 1
            # First derivatives
            d_accum = D_accum[counter - 1] @ W[layer][1:, :]

            # Second derivatives
            q = matmul(
                matmul(H[counter - 1], W[layer][1:, :]).swapaxes(0, 1), D[layer]
            ).swapaxes(0, 1)
            h = (
                matmul(
                    d_accum, matmul(d_accum, D2[layer].swapaxes(0, 1)).swapaxes(0, 2)
                )
                .swapaxes(0, 2)
                .swapaxes(0, 1)
            )

            # Store results
            D_accum.append(d_accum @ D[layer])
            Q.append(matmul(H[counter - 1], W[layer][1:, :]))
            H.append(h + q)

    return W, Z, O, D, D2, D_accum, Q, H, counter, mlpstr


def hessian_mlp(
    wts,
    bias,
    actfunc,
    X: pd.core.frame.DataFrame,
    y: pd.core.frame.DataFrame,
    sens_origin_layer: int = 0,
    sens_end_layer="last",
    sens_origin_input: bool = True,
    sens_end_input: bool = False,
    dev: str = "cpu",
    use_torch: bool = False,
):
    """Obtain second derivatives of MLP model given an input space.

    Second derivatives of MLP model may be used to analyze the variable
    relationships between inputs and outputs.

    Args:
        wts (list[float]): list of weight matrixes of the MLP layers.
        bias (list[float]): list of bias vectors of the MLP layers.
        actfunc (list[str]): list of names of the activation function of the MLP layers.
        X (pd.core.frame.DataFrame): data.frame with the input data
        y (pd.core.frame.DataFrame): data.frame with the output data
        sens_origin_layer (int, optional): layer from where the derivatives shall be calculated.
            Defaults to 0 (input layer).
        sens_end_layer (str | int, optional): layer to where the derivatives shall be calculated.
            Defaults to "last" (output layer).
        sens_origin_input (bool, optional): flag to indicate if derivatives shall be calculated from inputs (True)
            of the origin layer or the outputs (False). Defaults to True.
        sens_end_input (bool, optional): flag to indicate if derivatives shall be calculated to inputs (True)
            of the end layer or the outputs (False). Defaults to False.
        dev (str, optional): device where calculations shall be performed ("gpu" or "cpu").
            Only available if pytorch installation could be found. Defaults to "cpu".
        use_torch (bool, optional): Flag to indicate if pytorch Tensor shall be used. Defaults to False.

    Raises:
        ValueError: End layer to analyze cannot be smaller or equal to zero.
        ValueError: Origin layer to analyze cannot be smaller or equal to zero.
        ValueError: There must be a layer of neurons or weights between end layer and origin layer.
        ValueError: Origin layer should be less than number of layers in the model.
        ValueError: End layer should be less than number of layers in the model.

    Returns:
        Hessian_MLP: custom object storing the second partial derivatives of the MLP model.
    """
    # Import necessary modules based on processor, use torch when processor is gpu or use_torch is True
    if dev == "gpu" or use_torch:
        try:
            from torch import (
                mean,
                std,
                tensor,
                device,
                square,
                from_numpy,
                is_tensor,
                stack,
                sqrt,
            )
            from numpy import ndarray, asarray

            def float_array(x, dev="cpu") -> tensor:
                dev = device(dev)
                if isinstance(x, pd.DataFrame):
                    return from_numpy(x.to_numpy()).float().to(dev)
                if isinstance(x, list):
                    if isinstance(x[0], ndarray):
                        x = asarray(x)
                    elif is_tensor(x[0]):
                        return stack(x).float().to(dev)
                if isinstance(x, ndarray):
                    return from_numpy(x).float().to(dev)
                if is_tensor(x):
                    return x.clone().detach().to(dev)
                return tensor(x, device=dev)

            use_torch = True

        except ImportError:
            from numpy import array, mean, std, square, sqrt

            def float_array(x, dev="cpu") -> array:
                return array(x.numpy())

            use_torch = False

    else:
        from numpy import array, mean, std, array, square, sqrt

        def float_array(x, dev="cpu") -> array:
            return array(x)

        use_torch = False

    # Check validity of inputs
    if sens_end_layer == "last":
        sens_end_layer = len(actfunc)

    if sens_end_layer <= 0:
        raise ValueError("End layer to analyze cannot be smaller or equal to zero.")

    if sens_origin_layer < 0:
        raise ValueError("Origin layer to analyze cannot be smaller or equal to zero.")

    if not (sens_end_layer > sens_origin_layer) or (
        (sens_end_layer == sens_origin_layer)
        and (sens_origin_input and not sens_end_input)
    ):
        raise ValueError(
            "There must be a layer of neurons or weights between end layer and origin layer."
        )

    if sens_origin_layer > len(actfunc):
        raise ValueError(
            "Origin layer should be less than number of layers in the model."
        )

    if sens_end_layer > len(actfunc):
        raise ValueError("End layer should be less than number of layers in the model.")

    (
        _,
        _,
        _,
        _,
        _,
        D_accum,
        Q,
        H,
        counter,
        mlpstr,
    ) = calculate_second_partial_derivatives_mlp(
        wts,
        bias,
        actfunc,
        X,
        dev,
        sens_origin_layer,
        sens_end_layer,
        sens_origin_input,
        use_torch,
    )

    if sens_end_input:
        raw_sens = Q[counter]
    else:
        raw_sens = H[counter]

    # Calculate sensitivity measures for each input and output
    meanSens = mean(raw_sens, axis=0)
    stdSens = std(raw_sens, axis=0)
    meansquareSens = sqrt(mean(square(raw_sens), axis=0))

    # Store the information extracted from sensitivity analysis
    input_name = X.columns.values.tolist()
    metrics = ["mean", "std", "mean_squared"]

    sens = [
        pd.DataFrame(
            float_array(
                [meanSens[:, :, out], stdSens[:, :, out], meansquareSens[:, :, out]],
                dev=dev,
            ).T.reshape(meanSens.shape[1], -1),
            columns=pd.MultiIndex.from_product(
                [metrics, input_name], names=["metric", "input"]
            ),
            index=input_name,
        )
        for out in range(meanSens.shape[2])
    ]

    raw_sens = [
        pd.DataFrame(
            raw_sens[:, :, :, out]
            .T.reshape(raw_sens.shape[1] * raw_sens.shape[2], -1)
            .T,
            index=range(raw_sens.shape[0]),
            columns=pd.MultiIndex.from_product(
                [input_name, input_name], names=["input", "input"]
            ),
        )
        for out in range(raw_sens.shape[3])
    ]
    for out in range(meanSens.shape[2]):
        # Replace values on measures because don't know why they are not ordered correctly
        sens[out]["mean"] = meanSens[:, :, out]
        sens[out]["std"] = stdSens[:, :, out]
        sens[out]["mean_squared"] = meansquareSens[:, :, out]

    # Create output name for creating self
    output_name = y.columns.to_list()
    if D_accum[counter].shape[2] > 1:
        output_name = ["_".join([y.name, lev]) for lev in y.unique()]
    return Hessian_mlp(sens, raw_sens, mlpstr, X, input_name, output_name)


# Define self class
class Hessian_mlp:
    """A class used to store second partial derivatives of an MLP model.

    This class and its analogue for first partial derivatives are the main
    instruments of the package. It stores not only the partial derivatives but
    also the sensitivity metrics useful to interpret the input-output relationship.
    Methods of this class are used to interpret the information given by
    the v partial derivatives. It is not intended to be created outside the
    jacobian_mlp() function.

    Attributes:
        sens (list[]): list of sensitivity metrics stored in pandas DataFrames for
            each of the outputs.
        raw_sens (list[]): list of second partial derivatives matrix for each of the outputs.
        mlp_struct (list[int]): structure of the neural network as a list of neurons
            per layer.
        X (pd.DataFrame): Dataframe with the input variable samples used to calculate
            the partial derivatives stored in raw_sens.
        input_names (list[str]): name of the input variables.
        output_name (list[str]): name of the output variables.

    Methods:
        summary(): print sensitivity measures
        info(): print partial derivatives
        sensitivityPlots(): plot sensitivity measures
        featurePlots(): plot partial derivatives in a SHAP-alike manner
        timePlots(): plot partial derivatives with respect a time variable

    """

    def __init__(
        self,
        sens: list,
        raw_sens: list,
        mlp_struct,
        X: pd.core.frame.DataFrame,
        input_name,
        output_name,
    ):
        """Constructs all the necessary attributes for the Hessian_mlp object.

        Args:
            sens (list[]): list of sensitivity metrics stored in pandas DataFrames for
                each of the outputs.
            raw_sens (list[]): list of second partial derivatives matrix for each of the outputs.
            mlp_struct (list[int]): structure of the neural network as a list of neurons
                per layer.
            X (pd.DataFrame): Dataframe with the input variable samples used to calculate
                the partial derivatives stored in raw_sens.
            input_names (list[str]): name of the input variables.
            output_name (list[str]): name of the output variables.
        """
        self.__sens = sens
        self.__raw_sens = raw_sens
        self.__mlp_struct = mlp_struct
        self.__X = X
        self.__input_name = input_name
        self.__output_name = output_name

    @property
    def sens(self):
        return self.__sens

    @property
    def raw_sens(self):
        return self.__raw_sens

    @property
    def mlp_struct(self):
        return self.__mlp_struct

    @property
    def X(self):
        return self.__X

    @property
    def input_name(self):
        return self.__input_name

    @property
    def output_name(self):
        return self.__output_name

    def __repr__(self) -> str:
        return f"<Hessian of {self.mlp_struct} MLP network with inputs {self.input_name} and output {self.output_name}>"

    def __str__(self):
        self.summary()
        return ""

    def summary(self):
        """Print the sensitivity metrics"""
        print("Sensitivity analysis of", str(self.mlp_struct), "MLP network.\n")
        print("Sensitivity measures of each output:\n")
        for out in range(len(self.sens)):
            print("$" + self.output_name[out], "\n")
            print(self.sens[out])

    def info(self, n=5):
        """Prints the partial derivatives

        Args:
            n (int, optional): number of partial derivatives to display. Defaults to 5.
        """
        print("Sensitivity analysis of", str(self.mlp_struct), "MLP network.\n")
        print(self.X.shape[0], "samples\n")
        print(
            "Sensitivities of each output (only ",
            min([n, self.raw_sens[0].shape[0]]),
            " first samples):\n",
            sep="",
        )
        for out in range(len(self.raw_sens)):
            print("$" + self.output_name[out], "\n")
            print(self.raw_sens[out][: min([n, self.raw_sens[out].shape[0]])])

    def plot(self, type="sens", **kwargs):
        """Generate plot based on second partial derivatives.

        Args:
            type (str, optional): Type of plot to generate, accepted options are "sens", "features" and "time.
                Defaults to "sens".
        """
        if type == "sens":
            self.sensitivityPlots(**kwargs)
        elif type == "features":
            self.featurePlots(**kwargs)
        elif type == "time":
            self.timePlots(**kwargs)
        else:
            print("The specifyed type", type, "is not an accepted plot type.")

    def sensitivityPlots(self, **kwargs):
        temp_self = self.hesstosens()
        sensitivity_plots(temp_self, **kwargs)

    def featurePlots(self):
        pass

    def timePlots(self):
        pass

    def hesstosens(self, dev="gpu"):
        # Import necessary modules based on processor, use torch when processor is gpu
        if dev == "gpu":
            try:
                from torch import (
                    mean,
                    std,
                    tensor,
                    device,
                    from_numpy,
                    is_tensor,
                    stack,
                )
                from numpy import ndarray, asarray

                def float_array(x, dev="cpu") -> tensor:
                    dev = device(dev)
                    if isinstance(x, pd.DataFrame):
                        return from_numpy(x.to_numpy()).float().to(dev)
                    if isinstance(x, list):
                        if isinstance(x[0], ndarray):
                            x = asarray(x)
                        elif is_tensor(x[0]):
                            return stack(x).float().to(dev)
                    if isinstance(x, ndarray):
                        return from_numpy(x).float().to(dev)
                    if is_tensor(x):
                        return x.clone().detach().to(dev)
                    return tensor(x, device=dev)

            except ImportError:
                from numpy import (
                    array,
                    mean,
                    std,
                )

                def float_array(x, dev="cpu") -> array:
                    return array(x)

        else:
            from numpy import mean, std, array

            def float_array(x, dev="cpu") -> array:
                return array(x)

        # Function to convert from hessian to jacobian
        temp_self = self
        for out in range(len(self.raw_sens)):
            n_inputs = len(temp_self.input_name)
            index_of_interest = triu_indices(n_inputs)
            mean = temp_self.sens[out]["mean"].to_numpy()[index_of_interest]
            std = temp_self.sens[out]["std"].to_numpy()[index_of_interest]
            mean_squared = temp_self.sens[out]["mean_squared"].to_numpy()[
                index_of_interest
            ]
            input_names = meshgrid(temp_self.input_name, temp_self.input_name)
            input_names = (
                input_names[0].astype(object) + "_" + input_names[1].astype(object)
            )
            input_names = input_names[index_of_interest]
            temp_self.sens[out] = pd.DataFrame(
                {"mean": mean, "std": std, "mean_squared": mean_squared},
                index=input_names,
            )
            raw_sens = (
                temp_self.raw_sens[out]
                .to_numpy()
                .reshape(
                    temp_self.raw_sens[out].to_numpy().shape[0], n_inputs, n_inputs
                )
            )
            raw_sens = float_array(
                [
                    raw_sens[:, x, y]
                    for x, y in zip(index_of_interest[0], index_of_interest[1])
                ]
            )
            temp_self.raw_sens[out] = pd.DataFrame(
                raw_sens.T, index=range(raw_sens.shape[1]), columns=input_names
            )
        return temp_self


def calculate_third_partial_derivatives_mlp(
    wts,
    bias,
    actfunc,
    X: pd.core.frame.DataFrame,
    dev: str = "cpu",
    sens_origin_layer: int = 0,
    sens_end_layer="last",
    sens_origin_input: bool = True,
    use_torch: bool = False,
):
    """
    Calculate the first partial derivatives of a Multi-Layer Perceptron (MLP) model.

    Parameters:
    wts (list of numpy arrays or torch tensors): List of weight matrices for each layer of the MLP.
    bias (list of numpy arrays or torch tensors): List of bias vectors for each layer of the MLP.
    actfunc (list of callable functions): List of activation functions for each layer of the MLP.
    X (pd.core.frame.DataFrame): Input data as a pandas DataFrame.
    dev (str): Device specification, either "cpu" or "gpu". If "gpu" is chosen or use_torch is True, it uses PyTorch for calculations, otherwise, it uses NumPy. Default is "cpu".
    sens_origin_layer (int): The starting layer index for sensitivity calculation. Default is 0.
    sens_end_layer (int or str): The ending layer index for sensitivity calculation. If "last", it calculates sensitivity up to the last layer. Default is "last".
    sens_origin_input (bool, optional): flag to indicate if derivatives shall be calculated from inputs (True)
        of the origin layer or the outputs (False). Defaults to True.
    use_torch (bool): If True, use PyTorch for calculations even if dev is "cpu". Default is False.

    Returns:
    W (list of numpy arrays or torch tensors): List of weight matrices for each layer of the MLP.
    Z (list of numpy arrays or torch tensors): List of input values for each layer of the MLP.
    O (list of numpy arrays or torch tensors): List of output values for each layer of the MLP.
    D (list of numpy arrays or torch tensors): List of derivatives for each layer of the MLP.
    D2 (list of numpy arrays or torch tensors): List of second derivatives for each layer of the MLP.
    D3 (list of numpy arrays or torch tensors): List of third derivatives for each layer of the MLP.
    D_accum (list of numpy arrays or torch tensors): List of accumulated sensitivity values for each calculated layer.
    Q (list of numpy arrays or torch tensors): List of accumulated second partial derivatives of the input of the layer with respect to the inputs of the model.
    H (list of numpy arrays or torch tensors): List of accumulated second partial derivatives of the output of the layer with respect to the inputs of the model.
    J (list of numpy arrays or torch tensors): List of accumulated third partial derivatives of the input of the layer with respect to the inputs of the model.
    K (list of numpy arrays or torch tensors): List of accumulated third partial derivatives of the output of the layer with respect to the inputs of the model.
    counter (int): Number of layers between sens_origin_layer and sens_end_layer. It serves to know the final derivatives to return in the jacobian_mlp.
    mlpstr (list of integer): Structure of the mlp model.
    """
    # Import necessary modules based on processor, use torch when processor is gpu or use_torch is True
    if dev == "gpu" or use_torch:
        try:
            from torch import (
                eye as identity,
                vstack,
                ones,
                tensor,
                device,
                matmul,
                from_numpy,
                is_tensor,
                stack,
                zeros,
                einsum,
            )
            from numpy import ndarray, asarray

            def float_array(x, dev="cpu") -> tensor:
                dev = device(dev)
                if isinstance(x, pd.DataFrame):
                    return from_numpy(x.to_numpy()).float().to(dev)
                if isinstance(x, list):
                    if isinstance(x[0], ndarray):
                        x = asarray(x)
                    elif is_tensor(x[0]):
                        return stack(x).float().to(dev)
                if isinstance(x, ndarray):
                    return from_numpy(x).float().to(dev)
                if is_tensor(x):
                    return x.clone().detach().to(dev)
                return tensor(x, device=dev)

            use_torch = True

        except ImportError:
            from numpy import (
                identity,
                vstack,
                array,
                ones,
                array,
                matmul,
                zeros,
                einsum,
            )

            def float_array(x, dev="cpu") -> array:
                return array(x.numpy())

            use_torch = False

    else:
        from numpy import identity, vstack, array, ones, array, matmul, zeros, einsum

        def float_array(x, dev="cpu") -> array:
            return array(x)

        use_torch = False

    ### Initialize all the necessary variables
    # Structure of the mlp model
    mlpstr = [wts[0].shape[0]] + [lyr.shape[1] for lyr in wts]

    # Derivative and activation functions for each neuron layer
    der3actfunc = [der_3_activation_function(af, use_torch) for af in actfunc]
    der2actfunc = [der_2_activation_function(af, use_torch) for af in actfunc]
    deractfunc = [der_activation_function(af, use_torch) for af in actfunc]
    actfunc = [activation_function(af, use_torch) for af in actfunc]

    # Number of samples to be cached (it is used several times)
    n_samples = X.shape[0]
    r_samples = range(n_samples)
    n_inputs = X.shape[1]

    # Weights of input layer
    W = [identity(X.shape[1])]

    # Input of input layer
    # inputs = [np.hstack((ones((len(X_train),1), dtype=int), X_train))]
    Z = [matmul(float_array(X, dev), W[0])]

    # Output of input layer
    O = [actfunc[0](Z[0])]

    # First Derivative of input layer
    D = [float_array([deractfunc[0](Z[0][irow,]) for irow in r_samples], dev=dev)]

    # Second derivative of input layer
    D2 = [float_array([der2actfunc[0](Z[0][irow,]) for irow in r_samples], dev=dev)]

    # Third derivative of input layer
    D3 = [float_array([der3actfunc[0](Z[0][irow,]) for irow in r_samples], dev=dev)]

    # Let's go over all the layers calculating each variable
    for lyr in range(1, len(mlpstr)):
        # Calculate weights of each layer
        W.append(vstack((bias[lyr - 1], wts[lyr - 1])))

        # Calculate input of each layer
        # Add columns of 1 for the bias
        aux = ones((O[lyr - 1].shape[0], O[lyr - 1].shape[1] + 1))
        aux[:, 1:] = O[lyr - 1]
        Z.append(matmul(aux, W[lyr]))

        # Calculate output of each layer
        O.append(actfunc[lyr](Z[lyr]))

        # Calculate first derivative of each layer
        D.append(
            float_array([deractfunc[lyr](Z[lyr][irow,]) for irow in r_samples], dev=dev)
        )

        # Calculate second derivative of each layer
        D2.append(
            float_array(
                [der2actfunc[lyr](Z[lyr][irow,]) for irow in r_samples], dev=dev
            )
        )

        # Calculate third derivative of each layer
        D3.append(
            float_array(
                [der3actfunc[lyr](Z[lyr][irow,]) for irow in r_samples], dev=dev
            )
        )

    # Now, let's calculate the derivatives of interest
    if sens_end_layer == "last":
        sens_end_layer = len(actfunc)

    # Initialize cross derivatives
    D_ = [identity(mlpstr[sens_origin_layer]) for irow in r_samples]
    if sens_origin_input:
        D_ = [D[sens_origin_layer]]
    Q = [zeros((n_samples, n_inputs, n_inputs, n_inputs))]
    H = [D2[sens_origin_layer]]

    K = [zeros((n_samples, n_inputs, n_inputs, n_inputs, n_inputs))]
    J = [D3[sens_origin_layer]]

    counter = 0
    # Only perform further operations if origin is not equal to end layer
    if not (sens_origin_layer == sens_end_layer):
        for layer in range(sens_origin_layer + 1, sens_end_layer):
            counter += 1

            # First derivatives
            z_ = D_[counter - 1] @ W[layer][1:, :]
            D_.append(z_ @ D[layer])

            # Second derivatives
            q = matmul(
                matmul(H[counter - 1], W[layer][1:, :]).swapaxes(0, 1), D[counter]
            ).swapaxes(0, 1)
            h = matmul(
                z_, matmul(z_, D2[layer].swapaxes(0, 1)).swapaxes(0, 2)
            ).swapaxes(0, 1)
            Q.append(q)
            H.append(h + q)

            # Third derivatives
            j = matmul(
                matmul(K[counter - 1], W[layer][1:, :]).swapaxes(0, 2), D[counter]
            ).swapaxes(0, 2)
            k = einsum(
                "sijrk,smr->sijmk",
                einsum(
                    "siqrk,sjq->sijrk", einsum("spqrk,sip->siqrk", D3[layer], z_), z_
                ),
                z_,
            )
            l = (
                einsum(
                    "siqk,sjmq->sijmk",
                    einsum("spqk,sip->siqk", D2[layer], z_),
                    matmul(H[counter - 1], W[layer][1:, :]),
                )
                + einsum(
                    "sjqk,simq->sijmk",
                    einsum("spqk,sjp->sjqk", D2[layer], z_),
                    matmul(H[counter - 1], W[layer][1:, :]),
                )
                + einsum(
                    "smqk,sijq->sijmk",
                    einsum("spqk,smp->smqk", D2[layer], z_),
                    matmul(H[counter - 1], W[layer][1:, :]),
                )
            )

            J.append(j)
            K.append(k + j + l)

    return W, Z, O, D, D2, D3, D_, Q, H, J, K, counter, mlpstr


def jerkian_mlp(
    wts,
    bias,
    actfunc,
    X: pd.core.frame.DataFrame,
    y: pd.core.frame.DataFrame,
    sens_origin_layer: int = 0,
    sens_end_layer="last",
    sens_origin_input: bool = True,
    sens_end_input: bool = False,
    dev: str = "cpu",
    use_torch: bool = False,
):
    """Obtain third derivatives of MLP model given an input space.

    Third derivatives of MLP model may be used to analyze the variable
    relationships between inputs and outputs.

    Args:
        wts (list[float]): list of weight matrixes of the MLP layers.
        bias (list[float]): list of bias vectors of the MLP layers.
        actfunc (list[str]): list of names of the activation function of the MLP layers.
        X (pd.core.frame.DataFrame): data.frame with the input data
        y (pd.core.frame.DataFrame): data.frame with the output data
        sens_origin_layer (int, optional): layer from where the derivatives shall be calculated.
            Defaults to 0 (input layer).
        sens_end_layer (str | int, optional): layer to where the derivatives shall be calculated.
            Defaults to "last" (output layer).
        sens_origin_input (bool, optional): flag to indicate if derivatives shall be calculated from inputs (True)
            of the origin layer or the outputs (False). Defaults to True.
        sens_end_input (bool, optional): flag to indicate if derivatives shall be calculated to inputs (True)
            of the end layer or the outputs (False). Defaults to False.
        dev (str, optional): device where calculations shall be performed ("gpu" or "cpu").
            Only available if pytorch installation could be found. Defaults to "cpu".
        use_torch (bool, optional): Flag to indicate if pytorch Tensor shall be used. Defaults to False.

    Raises:
        ValueError: End layer to analyze cannot be smaller or equal to zero.
        ValueError: Origin layer to analyze cannot be smaller or equal to zero.
        ValueError: There must be a layer of neurons or weights between end layer and origin layer.
        ValueError: Origin layer should be less than number of layers in the model.
        ValueError: End layer should be less than number of layers in the model.

    Returns:
        Jerkian_MLP: custom object storing the third partial derivatives of the MLP model.
    """

    # Import necessary modules based on processor, use torch when processor is gpu or use_torch is True
    if dev == "gpu" or use_torch:
        try:
            from torch import mean, std, square, nanmean, any, sqrt

            def nanstd(x, axis):
                # Flatten:
                shape = x.shape
                x_reshaped = x.reshape(shape[0], -1)
                # Drop all rows containing any nan:
                x_reshaped = x_reshaped[~any(x_reshaped.isnan(), dim=1)]
                # Reshape back:
                x = x_reshaped.reshape(x_reshaped.shape[0], *shape[1:])
                return std(x, axis=axis)

            use_torch = True

        except ImportError:
            from numpy import mean, std, square, nanmean, nanstd, sqrt

            use_torch = False

    else:
        from numpy import mean, std, square, nanmean, nanstd, sqrt

        use_torch = False

    # Check validity of inputs
    if sens_end_layer == "last":
        sens_end_layer = len(actfunc)

    if sens_end_layer <= 0:
        raise ValueError("End layer to analyze cannot be smaller or equal to zero.")

    if sens_origin_layer < 0:
        raise ValueError("Origin layer to analyze cannot be smaller or equal to zero.")

    if not (sens_end_layer > sens_origin_layer) or (
        (sens_end_layer == sens_origin_layer)
        and (sens_origin_input and not sens_end_input)
    ):
        raise ValueError(
            "There must be a layer of neurons or weights between end layer and origin layer."
        )

    if sens_origin_layer > len(actfunc):
        raise ValueError(
            "Origin layer should be less than number of layers in the model."
        )

    if sens_end_layer > len(actfunc):
        raise ValueError("End layer should be less than number of layers in the model.")

    (
        _,
        _,
        _,
        _,
        _,
        _,
        D_,
        _,
        _,
        J,
        K,
        counter,
        mlpstr,
    ) = calculate_third_partial_derivatives_mlp(
        wts,
        bias,
        actfunc,
        X,
        dev,
        sens_origin_layer,
        sens_end_layer,
        sens_origin_input,
        use_torch,
    )

    if sens_end_input:
        raw_sens = J[counter]
    else:
        raw_sens = K[counter]

    # Calculate sensitivity measures for each input and output
    meanSens = nanmean(raw_sens, axis=0)
    stdSens = nanstd(raw_sens, axis=0)
    meansquareSens = sqrt(mean(square(raw_sens), axis=0))

    # Store the information extracted from sensitivity analysis
    input_name = X.columns.values.tolist()

    sens = [
        pd.concat(
            [
                pd.DataFrame(
                    meanSens[:, :, :, out].T.reshape(meanSens.shape[0], -1),
                    index=input_name,
                    columns=pd.MultiIndex.from_product(
                        [["mean"], input_name, input_name],
                        names=["metric", "input", "input"],
                    ),
                ),
                pd.DataFrame(
                    stdSens[:, :, :, out].T.reshape(stdSens.shape[0], -1),
                    index=input_name,
                    columns=pd.MultiIndex.from_product(
                        [["std"], input_name, input_name],
                        names=["metric", "input", "input"],
                    ),
                ),
                pd.DataFrame(
                    meansquareSens[:, :, :, out].T.reshape(meansquareSens.shape[0], -1),
                    index=input_name,
                    columns=pd.MultiIndex.from_product(
                        [["mean_squared"], input_name, input_name],
                        names=["metric", "input", "input"],
                    ),
                ),
            ],
            axis=1,
        )
        for out in range(meanSens.shape[3])
    ]

    raw_sens = [
        pd.DataFrame(
            raw_sens[:, :, :, :, out]
            .T.reshape(raw_sens.shape[1] * raw_sens.shape[2] * raw_sens.shape[3], -1)
            .T,
            index=range(raw_sens.shape[0]),
            columns=pd.MultiIndex.from_product(
                [input_name, input_name, input_name], names=["input", "input", "input"]
            ),
        )
        for out in range(raw_sens.shape[4])
    ]

    # Create output name for creating self
    output_name = y.columns.to_list()
    if D_[counter].shape[2] > 1:
        output_name = ["_".join([y.name, lev]) for lev in y.unique()]
    return Jerkian_mlp(sens, raw_sens, mlpstr, X, input_name, output_name)


# Define self class
class Jerkian_mlp:
    """A class used to store third partial derivatives of an MLP model.

    This class and its analogues for first and second partial derivatives are the main
    instruments of the package. It stores not only the partial derivatives but
    also the sensitivity metrics useful to interpret the input-output relationship.
    Methods of this class are used to interpret the information given by
    the v partial derivatives. It is not intended to be created outside the
    jerkian_mlp() function.

    Attributes:
        sens (list[]): list of sensitivity metrics stored in pandas DataFrames for
            each of the outputs.
        raw_sens (list[]): list of third partial derivatives matrix for each of the outputs.
        mlp_struct (list[int]): structure of the neural network as a list of neurons
            per layer.
        X (pd.DataFrame): Dataframe with the input variable samples used to calculate
            the partial derivatives stored in raw_sens.
        input_names (list[str]): name of the input variables.
        output_name (list[str]): name of the output variables.

    Methods:
        summary(): print sensitivity measures
        info(): print partial derivatives
        sensitivityPlots(): plot sensitivity measures
        featurePlots(): plot partial derivatives in a SHAP-alike manner
        timePlots(): plot partial derivatives with respect a time variable

    """

    def __init__(
        self,
        sens: list,
        raw_sens: list,
        mlp_struct,
        X: pd.core.frame.DataFrame,
        input_name,
        output_name,
    ):
        self.__sens = sens
        self.__raw_sens = raw_sens
        self.__mlp_struct = mlp_struct
        self.__X = X
        self.__input_name = input_name
        self.__output_name = output_name

    @property
    def sens(self):
        return self.__sens

    @property
    def raw_sens(self):
        return self.__raw_sens

    @property
    def mlp_struct(self):
        return self.__mlp_struct

    @property
    def X(self):
        return self.__X

    @property
    def input_name(self):
        return self.__input_name

    @property
    def output_name(self):
        return self.__output_name

    def __repr__(self) -> str:
        return f"<Jerkian of {self.mlp_struct} MLP network with inputs {self.input_name} and output {self.output_name}>"

    def __str__(self):
        self.summary()
        return ""

    def summary(self):
        print("Sensitivity analysis of", str(self.mlp_struct), "MLP network.\n")
        print("Sensitivity measures of each output:\n")
        for out in range(len(self.sens)):
            print("$" + self.output_name[out], "\n")
            print(self.sens[out])

    def info(self, n=5):
        print("Sensitivity analysis of", str(self.mlp_struct), "MLP network.\n")
        print(self.X.shape[0], "samples\n")
        print(
            "Sensitivities of each output (only ",
            min([n, self.raw_sens[0].shape[0]]),
            " first samples):\n",
            sep="",
        )
        for out in range(len(self.raw_sens)):
            print("$" + self.output_name[out], "\n")
            print(self.raw_sens[out][: min([n, self.raw_sens[out].shape[0]])])

    def plot(self, type="sens", **kwargs):
        if type == "sens":
            self.sensitivityPlots(**kwargs)
        elif type == "features":
            self.featurePlots(**kwargs)
        elif type == "time":
            self.timePlots(**kwargs)
        else:
            print("The specifyed type", type, "is not an accepted plot type.")

    def sensitivityPlots(self, **kwargs):
        temp_self = self.hesstosens()
        sensitivity_plots(temp_self, **kwargs)

    def featurePlots(self):
        pass

    def timePlots(self):
        pass

    def hesstosens(self):
        pass


from matplotlib.colors import LinearSegmentedColormap


def sensitivity_plots(jacobian, plot_type=None, inp_var=None, figsize=(15, 10)):
    """Generate plots to interpret sensitivities of MLP

    Generate three plots based on partial derivatives of a Jacobian_MLP object:
    Plot 1: Label plot representing the relationship between average sensitivity (x-axis)
        vs sensitivity standard deviation (y-axis).
    Plot 2: Bar plot of the square root of mean square sensitivities for each input variable.
    Plot 3: Density plot of the distribution of output sensitivities with regard to each input variable.

    Args:
        jacobian (Jacobian_mlp): Jacobian_mlp object with the partial derivatives of the MLP model.
        plot_type (list[str]): Name of the plots that shall be shown. If None, all plots are created.
            Name of plots are 'mean_sd', 'square' and 'raw' respectively. Default to None.
        inp_var (list[str]): Name of input variables to include in the plots. If None,
            all variables are included. Default to None.
        figsize (tuple(int, int)): Size of the plot figure. Default to (15, 10).
    """
    sens = jacobian.sens
    raw_sens = jacobian.raw_sens

    if plot_type is None:
        plot_type = ["mean_sd", "square", "raw"]
    fig, ax = plt.subplots(len(plot_type), len(jacobian.raw_sens), figsize=figsize)
    fig.suptitle("Sensitivity plots")

    if isinstance(plot_type, str):
        plot_type = [plot_type]

    if inp_var is not None:
        sens = sens.loc[inp_var, :]
        raw_sens = raw_sens.loc[:, inp_var]

    for out, raw_sens in enumerate(jacobian.raw_sens):
        if hasattr(ax, "shape"):
            if len(ax.shape) > 1:
                axes = [[out, i] for i in range(len(plot_type))]
            else:
                axes = [i for i in range(len(plot_type))]
        else:
            axes = [i for i in range(len(plot_type))]
            ax = [ax]
        sens_out = sens[out]

        # Plot mean-std plot
        plot_index = 0
        if "mean_sd" in plot_type:
            ax[axes[plot_index]].set_xlim(
                [
                    min(sens_out["mean"]) - 0.2 * max(abs(sens_out["mean"])),
                    max(sens_out["mean"]) + 0.2 * max(abs(sens_out["mean"])),
                ]
            )
            ax[axes[plot_index]].hlines(
                y=0,
                xmin=min(sens_out["mean"]) - 0.2 * max(abs(sens_out["mean"])),
                xmax=max(sens_out["mean"]) + 0.2 * max(abs(sens_out["mean"])),
                color="blue",
            )
            ax[axes[plot_index]].vlines(
                x=0, ymin=0, ymax=1.2 * max(sens_out["std"]), color="blue"
            )
            ax[axes[plot_index]].scatter(x=0, y=0, s=150, c="blue")
            for i, txt in enumerate(sens_out.index.values.tolist()):
                ax[axes[plot_index]].annotate(
                    txt,
                    xy=(sens_out.loc[txt, "mean"], sens_out.loc[txt, "std"]),
                    xycoords="data",
                    va="center",
                    ha="center",
                    fontsize="large",
                    bbox=dict(boxstyle="round", fc="w", ec="gray"),
                )
            ax[axes[plot_index]].set_xlabel("mean(Sens)")
            ax[axes[plot_index]].set_ylabel("std(Sens)")
            ax[axes[plot_index]].title.set_text(
                "Sensitivity plots for output " + str(out)
            )
            plot_index += 1

        # Plot variable importance mean_squared
        if "square" in plot_type:

            def normalize(value, min_val, max_val):
                if value > 0:
                    # Scale positive values to [0.5, 1]
                    return 0.5 + 0.5 * (value / max_val)
                elif value < 0:
                    # Scale negative values to [0, 0.5]
                    return 0.5 * (value - min_val) / (-min_val)
                else:
                    # Zero value
                    return 0.5

            # Normalize the 'mean' values to the range [0, 1]
            min_mean = -sens_out["mean"].abs().max()
            max_mean = sens_out["mean"].abs().max()
            normalized_means = sens_out.sort_values(
                by=["mean_squared"], ascending=True
            )["mean"].apply(normalize, args=(min_mean, max_mean))

            # Create a custom colormap (red to black to cyan)
            cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", ["red", "black", "cyan"]
            )

            # Get colors from the colormap
            colors = cmap(normalized_means)

            sns.barplot(
                x=sens_out.index.values.tolist(),
                y=sens_out["mean_squared"].values,
                order=sens_out.sort_values(by=["mean_squared"], ascending=True).index,
                palette=colors,
                ax=ax[axes[plot_index]],
            )
            ax[axes[plot_index]].set_xlabel("Input variables")
            ax[axes[plot_index]].set_ylabel("sqrt(mean(S^2))")
            ax[axes[plot_index]].legend([], [], frameon=False)
            plot_index += 1

        # Plot density of sensitivities
        if "raw" in plot_type:
            raw_sens.plot.kde(ax=ax[axes[plot_index]])
            ax[axes[plot_index]].set_xlabel("Sens")
            ax[axes[plot_index]].set_ylabel("density(Sens)")
        plt.tight_layout()


def alpha_sens_curves(
    jacobian,
    tol: float = None,
    max_alpha: int = 16,
    alpha_step: int = 1,
    curve_equal_origin: bool = False,
    dev: str = "cpu",
    use_torch: bool = False,
    title: str = "alpha-curves",
    alpha_bar: int = 1,
    inp_var=None,
    kind: str = "line",
    figsize: tuple = (10, 8),
):
    """Generate plots of alpha curves of partial derivatives

    Args:
        jacobian (list[Float] | Jacobian_mlp): Object storing partial derivatives
        tol (float, optional): Minimum change between consecutive alpha-mean values to continue calculating
            means. If None, Defaults to None.
        max_alpha (int, optional): Maximum alpha value to calculate. Defaults to 100.
        alpha_step (int, optional): Specify increment between consecutive alphas. Defaults to 1.
        curve_equal_origin (bool, optional): Flag to specify if all alpha curves must begin at (1, 0).
            Defaults to False.
        dev (str, optional): device where calculations shall be performed ("gpu" or "cpu").
            Only available if pytorch installation could be found. Defaults to "cpu".
        use_torch (bool, optional): Flag to indicate if pytorch Tensor shall be used. Defaults to False.
        inp_var (list[str], optional): List of variable names to show in alpha curves. If None,
            a list of str [X1, ..., Xn] is generated and used as input variables. Defaults to None.
        title [str]: title to show above alpha sensitivity curves plot
        alpha_bar ([int], [str]): alpha to create bar plot. Could be defined as 'inf' to show bar plot for alpha=np.inf.
            Defaults to 1.
        kind [str]: type of alpha plot that should be created. Options are 'bar' (alpha mean value for a given alpha_bar)
            or 'line' (default alpha-means plot). Defaults to 'line'.
        figsize [tuple(int, int)]: Size of figure created. Defaults to (10, 8).
    """
    # Import necessary modules based on processor, use torch when processor is gpu or use_torch is True
    if dev == "gpu" or use_torch:
        try:
            from torch import arange, log10, abs
            from numpy import empty, inf

        except ImportError:
            from numpy import arange, floor, log10, abs, empty, inf

    else:
        from numpy import arange, floor, log10, abs, empty, inf

    # Check if field in variable
    if hasattr(jacobian, "raw_sens"):
        # Initialize variables
        if inp_var is None:
            inp_var = jacobian.X.columns
        raw_sens = jacobian.raw_sens
        input_name = inp_var
    else:
        input_name = inp_var
        if inp_var is not None:
            if len(inp_var) != jacobian.shape[1]:
                warnings.warn(
                    f"Number of columns names {len(inp_var)} does not match number of input variables {jacobian.shape[1]}, default variable names would be used."
                )
                inp_var = None
        if inp_var is None:
            input_name = ["X{}".format(x) for x in (arange(0, jacobian.shape[1]))]
        raw_sens = [pd.DataFrame(jacobian, columns=input_name)]
    alphas = arange(1, max_alpha + 1, alpha_step)
    # Select the color map named rainbow
    cmap = cm.get_cmap(name="rainbow")

    # Detect type of alpha plot
    if kind not in ["line", "bar"]:
        raise ValueError(
            f"Plot type {kind} not admitted. Choose 'line' or 'bar' to create the alpha plot."
        )

    # Go through each output although this analysis is meant only for regression
    if kind == "line":
        texts = []
        fig, ax = plt.subplots(
            len(raw_sens),
            2,
            sharey=True,
            gridspec_kw=dict(width_ratios=[max_alpha, 4]),
            figsize=figsize,
        )
        fig.subplots_adjust(wspace=0.1)  # adjust space between axes

        for out, rs in enumerate(raw_sens):
            if len(raw_sens) > 1:
                axes = ax[[out, 0]]
            else:
                axes = ax
            for inp, input in enumerate(input_name):
                alpha_curves = alpha_curve(
                    rs.loc[:, input],
                    tol=tol,
                    max_alpha=max_alpha,
                    alpha_step=alpha_step,
                    dev=dev,
                )
                if curve_equal_origin:
                    alpha_curves -= alpha_curves[0]

                var_color = cmap(inp / rs.shape[1])

                axes[0].plot(
                    alphas[
                        logical_not(isnan(alpha_curves))
                        | logical_not(alpha_curves == float("+inf"))
                    ],
                    alpha_curves[
                        logical_not(isnan(alpha_curves))
                        | logical_not(alpha_curves == float("+inf"))
                    ],
                    label=input,
                    color=var_color,
                )

                alpha_notnan = alpha_curves[logical_not(isnan(alpha_curves))]
                alpha_notinf = alpha_notnan[logical_not(alpha_notnan == float("+inf"))]
                max_alpha_mean = max(abs(rs.iloc[:, inp]))
                try:
                    con_between_axes = ConnectionPatch(
                        xyA=[alphas[len(alpha_notinf) - 1], alpha_notinf[-1]],
                        xyB=[max_alpha + 3, max_alpha_mean],
                        coordsA="data",
                        coordsB="data",
                        axesA=ax[0],
                        axesB=ax[1],
                        color=var_color,
                        lw=1,
                        linestyle="dashed",
                    )
                    ax[1].add_artist(con_between_axes)

                    ax[1].plot(
                        [max_alpha + 3, max_alpha + 3.7],
                        [max_alpha_mean, max_alpha_mean],
                        color=var_color,
                        linestyle="dotted",
                        alpha=0.5,
                    )
                except IndexError:
                    pass
                ax[1].plot(
                    [max_alpha + 3, max_alpha + 3.1],
                    [max_alpha_mean, max_alpha_mean],
                    color=var_color,
                )
                ax[1].scatter(max_alpha + 3.1, max_alpha_mean, color=var_color)
                texts.append(
                    ax[1].text(
                        max_alpha + 3.7,
                        max_alpha_mean,
                        s=input,
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="lightgray",
                            edgecolor=var_color,
                            linewidth=1,
                        ),
                    )
                )
            adjust_text(texts, only_move={"points": "xy", "text": "y", "objects": "xy"})
            ax[0].set_ylabel(r"$(ms_{X,j}^\alpha(f))$")
            ax[0].set_xlabel(r"$\alpha$")
            xlabels = alphas.astype(str).tolist()
            ax[0].set_xticks(ticks=alphas, labels=xlabels)
            ax[1].set_xticks(ticks=[max_alpha + 3.1], labels=[r"$\infty$"])
            ax[1].yaxis.tick_right()
            ax[1].set_xlim([max_alpha + 2.9, max_alpha + 4])
            sec_yaxis = ax[1].secondary_yaxis("right")
            sec_yaxis.set_ylabel(r"$(ms_{X,j}^\alpha(f))^\alpha$")
            fig.suptitle(title)

    elif kind == "bar":
        fig, ax = plt.subplots(len(raw_sens), 1)
        for out, rs in enumerate(raw_sens):
            # Calculate scale factor
            N = rs.shape[0]
            alpha_values = empty(rs.shape[1])
            var_color = []
            if alpha_bar == "inf":
                for inp, input in enumerate(input_name):
                    alpha_values[inp] = max(abs(rs[input]))
                    var_color.append(cmap(inp / rs.shape[1]))
            else:
                for inp, input in enumerate(input_name):
                    scl_factor = 1
                    if not all(v == 0 for v in rs[input]):
                        scl_factor = 10 ** (
                            max(floor(log10(abs(rs[rs != 0.0][input])))) + 1
                        )
                    scl_rs = abs(rs[input]) / scl_factor
                    alpha_values[inp] = scl_factor * sum(scl_rs**alpha_bar / N) ** (
                        1 / alpha_bar
                    )
                    if alpha_values[inp] == inf:
                        scl_factor *= 5
                        scl_rs = abs(rs) / scl_factor
                        alpha_values[inp] = scl_factor * sum(
                            scl_rs**alpha_bar / N
                        ) ** (1 / alpha_bar)
                    var_color.append(cmap(inp / rs.shape[1]))
            alpha_order = alpha_values.argsort()
            alpha_values = alpha_values[alpha_order]
            var_color = [var_color[i] for i in alpha_order]
            input_name = [input_name[i] for i in alpha_order]
            ax.bar(input_name, alpha_values, color=var_color)
            ax.set_ylabel(r"$(ms_{X,j}^\alpha(f))$")
            ax.set_xlabel("Input variables")
            ax.title.set_text(title + f". Alpha: {alpha_bar}")

    return ax


def alpha_curve(
    raw_sens: list,
    tol: float = None,
    max_alpha: int = 100,
    alpha_step: int = 1,
    dev: str = "cpu",
    use_torch: bool = False,
):
    """Calculate alpha curve for a given variable

    Args:
        raw_sens (list): _description_Object storing partial derivatives
        tol (float, optional): Minimum change between consecutive alpha-mean values to continue calculating
            means. If None, Defaults to None.
        max_alpha (int, optional): Maximum alpha value to calculate. Defaults to 100.
        alpha_step (int, optional): Specify increment between consecutive alphas. Defaults to 1.
        dev (str, optional): device where calculations shall be performed ("gpu" or "cpu").
            Only available if pytorch installation could be found. Defaults to "cpu".
        use_torch (bool, optional): Flag to indicate if pytorch Tensor shall be used. Defaults to False.


    Returns:
        list[Float]: Calculated alpha curve
    """
    # Import necessary modules based on processor, use torch when processor is gpu or use_torch is True
    if dev == "gpu" or use_torch:
        try:
            from torch import tensor, device, arange, log10, abs, from_numpy, is_tensor
            from numpy import ndarray, asarray, empty, inf

            def float_array(x, dev="cpu") -> tensor:
                dev = device(dev)
                if isinstance(x, pd.DataFrame):
                    return from_numpy(x.to_numpy()).float()
                if isinstance(x, list):
                    x = asarray(x)
                if isinstance(x, ndarray):
                    return from_numpy(x).float()
                if is_tensor(x):
                    return x.clone().detach()
                return tensor(x, device=dev)

        except ImportError:
            from numpy import array, arange, floor, log10, abs, empty, inf

            def float_array(x, dev="cpu") -> array:
                return array(x.numpy())

    else:
        from numpy import array, arange, floor, log10, abs, empty, inf

        def float_array(x, dev="cpu") -> array:
            return array(x)

    # Obtain maximum sensitivity
    max_sens = max(abs(raw_sens))

    # Check if tolerance has been passed
    if tol is None:
        tol = 0.0001 * max_sens

    # Calculate scale factor
    N = raw_sens.shape[0]
    scl_factor = 1
    if not all(v == 0 for v in raw_sens):
        scl_factor = 10 ** (max(floor(log10(abs(raw_sens[raw_sens != 0.0])))) + 1)
    alpha = arange(1, max_alpha + 1, alpha_step)
    scl_rs = abs(raw_sens) / scl_factor
    alpha_curve = empty(alpha.shape)
    for i, a in enumerate(alpha):
        alpha_curve[i,] = scl_factor * sum(scl_rs**a / N) ** (1 / a)
        if alpha_curve[i,] == inf:
            scl_factor *= 5
            scl_rs = abs(raw_sens) / scl_factor
            alpha_curve[i,] = scl_factor * sum(scl_rs**a / N) ** (1 / a)
        if i > 1:
            if alpha_curve[i,] < alpha_curve[i - 1,]:
                scl_factor /= 10
                scl_rs = abs(raw_sens) / scl_factor
                alpha_curve[i,] = scl_factor * sum(scl_rs**a / N) ** (1 / a)
    alpha_curve = float_array(alpha_curve)
    return alpha_curve
