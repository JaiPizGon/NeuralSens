import pandas as pd
from numpy import triu_indices, meshgrid, nanmax, inf, linspace, vstack, empty, trapz, array, round, nan_to_num, trapz
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import warnings
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.express as px

from traitlets import Float
from neuralsens.activation_functions import (
    activation_function,
    der_activation_function,
    der_2_activation_function,
    der_3_activation_function,
)

plt.style.use("ggplot")


def jacobian_mlp(
    wts,
    bias,
    actfunc,
    X: pd.core.frame.DataFrame,
    y: pd.core.frame.DataFrame,
    sens_origin_layer: int = 0,
    sens_end_layer = "last",
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
                eye as identity,
                vstack,
                ones,
                mean,
                std,
                tensor,
                device,
                square,
                matmul,
                from_numpy,
                is_tensor,
                stack,
                sqrt
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
                mean,
                std,
                array,
                square,
                matmul,
                sqrt
            )

            def float_array(x, dev="cpu") -> array:
                return array(x.numpy())

            use_torch = False

    else:
        from numpy import (
            identity,
            vstack,
            array,
            ones,
            mean,
            std,
            array,
            square,
            matmul,
            sqrt
        )

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
        return f'<Jacobian of {self.mlp_struct} MLP network with inputs {self.input_name} and output {self.output_name}>'
    
    def __str__(self):
        self.summary()
        return ""

    def summary(self):
        """Prints the sensitivity measures.
        """
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

    def plot(self, type="sens"):
        """Generate plot based on partial derivatives.

        Args:
            type (str, optional): Type of plot to generate, accepted options are "sens", "features" and "time. 
                Defaults to "sens".
        """
        if type == "sens":
            self.sensitivityPlots()
        elif type == "features":
            self.featurePlots()
        elif type == "time":
            self.timePlots()
        else:
            print("The specifyed type", type, "is not an accepted plot type.")

    def sensitivityPlots(self):
        sensitivity_plots(self)

    def featurePlots(self):
        pass

    def timePlots(self):
        pass


def hessian_mlp(
    wts,
    bias,
    actfunc,
    X: pd.core.frame.DataFrame,
    y: pd.core.frame.DataFrame,
    sens_origin_layer: int = 0,
    sens_end_layer = "last",
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
                eye as identity,
                vstack,
                ones,
                mean,
                std,
                tensor,
                device,
                square,
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
            
            from numpy import (
                identity,
                vstack,
                array,
                ones,
                mean,
                std,
                array,
                square,
                zeros,
                matmul,
            )

            def float_array(x, dev="cpu") -> array:
                return array(x.numpy())

            use_torch = False

    else:
        from numpy import (
            identity,
            vstack,
            array,
            ones,
            mean,
            std,
            array,
            square,
            matmul,
            zeros,
        )

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
    H = [D2[0]]

    counter = 0
    # Only perform further operations if origin is not equal to end layer
    if not (sens_origin_layer == sens_end_layer):
        for layer in range(sens_origin_layer + 1, sens_end_layer):
            counter += 1
            # First derivatives
            d_accum = D_accum[counter - 1] @ D[layer - 1] @ W[layer][1:, :]

            # Second derivatives
            q = matmul(
                matmul(H[counter - 1], W[layer][1:, :]).swapaxes(0, 1), D[counter]
            ).swapaxes(0, 1)
            h = (
                matmul(
                    d_accum, matmul(d_accum, D2[layer].swapaxes(0, 1)).swapaxes(0, 2)
                )
                .swapaxes(0, 2)
                .swapaxes(0, 1)
            )

            # Store results
            D_accum.append(d_accum)
            Q.append(q)
            H.append(h + q)

    if sens_end_input:
        raw_sens = Q[counter]
    else:
        raw_sens = H[counter]

    # Calculate sensitivity measures for each input and output
    meanSens = mean(raw_sens, axis=0)
    stdSens = std(raw_sens, axis=0)
    meansquareSens = mean(square(raw_sens), axis=0)

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
        self.__input_name= input_name
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
        return f'<Hessian of {self.mlp_struct} MLP network with inputs {self.input_name} and output {self.output_name}>'
    
    def __str__(self):
        self.summary()
        return ""

    def summary(self):
        """Print the sensitivity metrics
        """
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

    def plot(self, type="sens"):
        """Generate plot based on second partial derivatives.

        Args:
            type (str, optional): Type of plot to generate, accepted options are "sens", "features" and "time. 
                Defaults to "sens".
        """
        if type == "sens":
            self.sensitivityPlots()
        elif type == "features":
            self.featurePlots()
        elif type == "time":
            self.timePlots()
        else:
            print("The specifyed type", type, "is not an accepted plot type.")

    def sensitivityPlots(self):
        temp_self = self.hesstosens()
        sensitivity_plots(temp_self)

    def featurePlots(self):
        pass

    def timePlots(self):
        pass

    def hesstosens(self, dev="gpu"):
        # Import necessary modules based on processor, use torch when processor is gpu or use_torch is True
        if dev == "gpu" or use_torch:
            try:
                from torch import (
                    eye as identity,
                    vstack,
                    ones,
                    mean,
                    std,
                    tensor,
                    device,
                    square,
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
                from numpy import (
                    identity,
                    vstack,
                    array,
                    ones,
                    mean,
                    std,
                    array,
                    square,
                    zeros,
                    matmul,
                )

                def float_array(x, dev="cpu") -> array:
                    return array(x.numpy())

                use_torch = False

        else:
            from numpy import (
                identity,
                vstack,
                array,
                ones,
                mean,
                std,
                array,
                square,
                matmul,
                zeros,
            )

            def float_array(x, dev="cpu") -> array:
                return array(x)

            use_torch = False
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



def jerkian_mlp(
    wts,
    bias,
    actfunc,
    X: pd.core.frame.DataFrame,
    y: pd.core.frame.DataFrame,
    sens_origin_layer: int = 0,
    sens_end_layer = "last",
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
            from torch import (
                eye as identity,
                vstack,
                ones,
                mean,
                std,
                tensor,
                device,
                square,
                matmul,
                zeros,
                from_numpy,
                is_tensor,
                einsum,
                nanmean,
                any
            )
            from numpy import ndarray, asarray
            def nanstd(x, axis):
                #Flatten:
                shape = x.shape
                x_reshaped = x.reshape(shape[0],-1)
                #Drop all rows containing any nan:
                x_reshaped = x_reshaped[~any(x_reshaped.isnan(),dim=1)]
                #Reshape back:
                x = x_reshaped.reshape(x_reshaped.shape[0],*shape[1:])
                return std(x, axis=axis)

            def float_array(x, dev="cpu") -> tensor:
                dev = device(dev)
                if isinstance(x, pd.DataFrame):
                    return from_numpy(x.to_numpy()).float()
                if isinstance(x, list):
                    try:
                        x = asarray([y.numpy() for y in x])
                    except:
                        x = asarray(x)
                if isinstance(x, ndarray):
                    return from_numpy(x).float()
                if is_tensor(x):
                    return x.clone().detach()
                return tensor(x, device=dev)

            use_torch = True

        except ImportError:
            
            from numpy import (
                identity,
                vstack,
                array,
                ones,
                mean,
                std,
                array,
                square,
                matmul,
                zeros,
                einsum,
                nanmean,
                nanstd
            )

            def float_array(x, dev="cpu") -> array:
                return array(x.numpy())

            use_torch = False

    else:
        from numpy import (
            identity,
            vstack,
            array,
            ones,
            mean,
            std,
            array,
            square,
            matmul,
            zeros,
            einsum,
            nanmean,
            nanstd
        )

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
            float_array(
                [deractfunc[lyr](Z[lyr][irow,]) for irow in r_samples], dev=dev
            )
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
    H = [D2[0]]

    K = [zeros((n_samples, n_inputs, n_inputs, n_inputs, n_inputs))]
    J = [D3[0]]
    
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
                    matmul(
                        H[counter - 1], W[layer][1:, :]
                    ).swapaxes(0, 1), 
                D[counter]
            ).swapaxes(0, 1)
            h = matmul(z_, matmul(z_, D2[layer].swapaxes(0, 1)).swapaxes(0, 2)).swapaxes(0, 1)
            Q.append(q)
            H.append(h + q)

            # Third derivatives
            j = matmul(
                    matmul(
                        K[counter - 1], W[layer][1:, :]
                    ).swapaxes(0, 2), 
                D[counter]
            ).swapaxes(0, 2)
            k = einsum('sijrk,smr->sijmk',einsum('siqrk,sjq->sijrk',einsum('spqrk,sip->siqrk', D3[layer], z_), z_), z_)
            l = einsum('siqk,sjmq->sijmk',einsum('spqk,sip->siqk', D2[layer], z_), matmul(H[counter - 1], W[layer][1:, :])) + \
                einsum('sjqk,simq->sijmk',einsum('spqk,sjp->sjqk', D2[layer], z_), matmul(H[counter - 1], W[layer][1:, :])) + \
                einsum('smqk,sijq->sijmk',einsum('spqk,smp->smqk', D2[layer], z_), matmul(H[counter - 1], W[layer][1:, :]))
            
            J.append(j)
            K.append(k + j + l)

    if sens_end_input:
        raw_sens = J[counter]
    else:
        raw_sens = K[counter]

    # Calculate sensitivity measures for each input and output
    meanSens = nanmean(raw_sens, axis=0)
    stdSens = nanstd(raw_sens, axis=0)
    meansquareSens = mean(square(raw_sens), axis=0)

    # Store the information extracted from sensitivity analysis
    input_name = X.columns.values.tolist()
    metrics = ["mean", "std", "mean_squared"]

    sens = [
        pd.concat([pd.DataFrame(meanSens[:, :, :, out].T.reshape(meanSens.shape[0],-1), index=input_name,columns=pd.MultiIndex.from_product(
                [["mean"],input_name, input_name], names=['metric',"input", "input"]
            )), 
            pd.DataFrame(stdSens[:, :, :, out].T.reshape(stdSens.shape[0],-1), index=input_name, columns=pd.MultiIndex.from_product(
                [["std"],input_name, input_name], names=['metric',"input", "input"]
            )),
            pd.DataFrame(meansquareSens[:, :, :, out].T.reshape(meansquareSens.shape[0],-1), index=input_name,columns=pd.MultiIndex.from_product(
                [["mean_squared"],input_name, input_name], names=['metric',"input", "input"]
            ))], axis=1)
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
    # for out in range(meanSens.shape[3]):
    #     # Replace values on measures because don't know why they are not ordered correctly
    #     sens[out]["mean"] = meanSens[:, :, :, out]
    #     sens[out]["std"] = stdSens[:, :, :, out]
    #     sens[out]["mean_squared"] = meansquareSens[:, :, :, out]

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
        self.__input_name= input_name
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
        return self.__input_names

    @property
    def output_name(self):
        return self.__output_name
    
    def __repr__(self) -> str:
        return f'<Jerkian of {self.mlp_struct} MLP network with inputs {self.input_name} and output {self.output_name}>'
    
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

    def plot(self, type="sens"):
        if type == "sens":
            self.sensitivityPlots()
        elif type == "features":
            self.featurePlots()
        elif type == "time":
            self.timePlots()
        else:
            print("The specifyed type", type, "is not an accepted plot type.")

    def sensitivityPlots(self):
        temp_self = self.hesstosens()
        sensitivity_plots(temp_self)

    def featurePlots(self):
        pass

    def timePlots(self):
        pass

    def hesstosens(self):
        pass


def sensitivity_plots(jacobian, plot_type=None, inp_var=None):
    """Generate plots to interpret sensitivities of MLP
    
    Generate three plots based on partial derivatives of a Jacobian_MLP object:
    Plot 1: Label plot representing the relationship between average sensitivity (x-axis)
        vs sensitivity standard deviation (y-axis). 
    Plot 2: Bar plot of the square root of mean square sensitivities for each input variable. 
    Plot 3: Density plot of the distribution of output sensitivities with regard to each input variable.

    Args:
        jacobian (Jacobian_mlp): Jacobian_mlp object with the partial derivatives of the MLP model.
        plot_type (list[str]): Name of the plots that shall be shown. If None, all plots are created. 
            Name of plots are 'mean_sd', 'square' and 'raw' respectively.Default to None.
        inp_var (list[str]): Name of input variables to include in the plots. If None, 
            all variables are included. Default to None. 
    """
    sens = jacobian.sens
    raw_sens = jacobian.raw_sens
    
    fig, ax = plt.subplots(3, len(jacobian.raw_sens))
    fig.suptitle("Sensitivity plots")
    if plot_type is None:
        plot_type = ['mean_sd', 'square', 'raw']
    
    if type(plot_type) == str:
        plot_type = [plot_type]
    
    if inp_var is not None:
        sens = sens.loc[inp_var, :]
        raw_sens = raw_sens.loc[:, inp_var]
        
    for out, raw_sens in enumerate(jacobian.raw_sens):
        if len(ax.shape) > 1:
            axes = [[out, i] for i in range(len(plot_type))]
        else:
            axes = [i for i in range(len(plot_type))]
        sens_out = sens[out]

        # Plot mean-std plot
        plot_index = 0
        if 'mean_sd' in plot_type:
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
            ax[axes[plot_index]].vlines(x=0, ymin=0, ymax=1.2 * max(sens_out["std"]), color="blue")
            ax[axes[plot_index]].scatter(x=0, y=0, s=150, c="blue")
            for i, txt in enumerate(sens_out.index.values.tolist()):
                ax[axes[plot_index]].annotate(
                    txt,
                    xy=(sens_out["mean"][i], sens_out["std"][i]),
                    xycoords="data",
                    va="center",
                    ha="center",
                    fontsize="large",
                    bbox=dict(boxstyle="round", fc="w", ec="gray"),
                )
            ax[axes[plot_index]].set_xlabel("mean(Sens)")
            ax[axes[plot_index]].set_ylabel("std(Sens)")
            ax[axes[plot_index]].title.set_text("Sensitivity plots for output " + str(out))
            plot_index += 1

        # Plot variable importance mean_squared
        if 'square' in plot_type:
            sns.barplot(x=sens_out.index.values.tolist(), y=sens_out['mean_squared'].values,
                        order=sens_out.sort_values(by=['mean_squared'], ascending=True).index,
                        palette='Blues_d', ax=ax[axes[plot_index]])
            ax[axes[plot_index]].set_xlabel("Input variables")
            ax[axes[plot_index]].set_ylabel("mean(Sens^2)")
            plot_index += 1

        # Plot density of sensitivities
        if 'raw' in plot_type:
            raw_sens.plot.kde(ax=ax[axes[2]])
            ax[axes[2]].set_xlabel("Sens")
            ax[axes[2]].set_ylabel("density(Sens)")
            plt.tight_layout()

def alpha_sens_curves(
    jacobian,
    tol: float = None,
    max_alpha: int = 100,
    alpha_step: int = 1,
    curve_equal_origin: bool = False,
    dev: str = "cpu",
    use_torch: bool = False,
    columns = None,
    title: str = 'alpha-curves',
    show_scaled: bool = False,
    alpha_bar: int = 1,
    inp_var=None,
    kind: str = 'line'
):
    """Generate plots of alpha curves of partial derivatives

    Args:
        jacobian (list[Float] | Jacobian_mlp): Object storing partial derivatives
        tol (float, optional): Minimum change between consecutive alpha-mean values to continue calculating
            means. If None, Defaults to None.
        max_alpha (int, optional): Maximum alpha value to calculate. Defaults to 100.
        alpha_step (int, optional): Specify increment between consecutive alphas. Defaults to 1.
        curve_equal_origin (bool, optional): Flag to specify if all alpha curves must begin at (1, 0). D
        efaults to False.
        dev (str, optional): device where calculations shall be performed ("gpu" or "cpu"). 
            Only available if pytorch installation could be found. Defaults to "cpu".
        use_torch (bool, optional): Flag to indicate if pytorch Tensor shall be used. Defaults to False.
        inp_var (list[str], optional): List of variable names to show in alpha curves. If None,
            a list of str [X1, ..., Xn] is generated and used as input variables. Defaults to None.
        title [str]: title to show above alpha sensitivity curves plot
        show_scaled (bool, optional): Flag to indicate if second plot of curves divided by maximum shall be plotted. 
            Defaults to False.
        alpha_bar ([int], [str]): alpha to create bar plot. Could be defined as 'inf' to show bar plot for alpha=np.inf.
            Defaults to 1. 
        kind [str]: type of alpha plot that should be created. Options are 'bar' (alpha mean value for a given alpha_bar)
            or 'line' (default alpha-means plot). Defaults to 'line'.
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
            
            from numpy import (
                array,
                arange,
                floor,
                log10,
                abs,
                empty, 
                inf
            )

            def float_array(x, dev="cpu") -> array:
                return array(x.numpy())

    else:
        from numpy import (
            array,
            arange,
            floor,
            log10,
            abs,
            empty, 
            inf
        )

        def float_array(x, dev="cpu") -> array:
            return array(x)

    # Check if field in variable 
    if hasattr(jacobian, 'raw_sens'):
        # Initialize variables
        if inp_var is None:
            inp_var = jacobian.X.columns
        raw_sens = jacobian.raw_sens[0].loc[:, inp_var]
        input_name = inp_var
    else:
        input_name = columns
        if columns is not None:
            if len(columns) != jacobian.shape[1]:
                warnings.warn(f"Number of columns names {len(columns)} does not match number of input variables {jacobian.shape[1]}, default variable names would be used.")
                columns = None
        if columns is None:
            input_name = ['X{}'.format(x) for x in (arange(0, jacobian.shape[1]))]
        raw_sens = [pd.DataFrame(jacobian, columns=input_name)]
    alphas = arange(1, max_alpha + 1, alpha_step)
    # Select the color map named rainbow
    cmap = cm.get_cmap(name="rainbow")
    
    # Detect type of alpha plot
    if kind not in ['line', 'bar']:
        raise ValueError(f"Plot type {kind} not admitted. Choose 'line' or 'bar' to create the alpha plot.")
    
    # Go through each output although this analysis is meant only for regression
    if kind == 'line':
        if show_scaled:
            fig, ax = plt.subplots(len(raw_sens), 2)
            fig.suptitle(title)
            for out, rs in enumerate(raw_sens):
                if len(ax.shape) > 1:
                    axes = [[out, 0], [out, 1]]
                else:
                    axes = [0, 1]
                for inp, input in enumerate(input_name):
                    alpha_curves = alpha_curve(
                        rs.iloc[:, inp], tol=tol, max_alpha=max_alpha, alpha_step=alpha_step, dev=dev
                    )
                    if curve_equal_origin:
                        alpha_curves -= alpha_curves[0]
                    
                    alpha_curves_scl = alpha_curves
                    max_alpha_mean = max(abs(rs.iloc[:, inp]))
                    if max_alpha_mean != 0:
                        alpha_curves_scl = alpha_curves / max_alpha_mean
                    var_color = cmap(inp / rs.shape[1])
                    ax[axes[0]].plot(
                        alphas, alpha_curves, label=input, color=var_color
                    )
                    ax[axes[0]].axhline(
                        y=max_alpha_mean, color=var_color, linestyle="dashed"
                    )
                    ax[axes[0]].text(
                        alphas[-1], alpha_curves[-1],
                        input, 
                        color=var_color,
                        path_effects=[pe.withStroke(linewidth=1, foreground="gray")]
                    )
                    ax[axes[0]].text(
                        max([1,max_alpha - 20]), max_alpha_mean,
                        'max({0}): {1:.{2}f}'.format(input, max_alpha_mean, 2),
                        color=var_color,
                        path_effects=[pe.withStroke(linewidth=1, foreground="gray")]
                    )
                    ax[axes[1]].plot(
                        alphas, alpha_curves_scl, label=input, color=var_color
                    )
                ax[axes[1]].axhline(y=1, color="gray", linestyle="dashed")
                ax[axes[1]].legend()
                ax[axes[0]].title.set_text("Alpha curves")
                ax[axes[1]].title.set_text("Alpha curves / max(sens)")
                ax[axes[0]].xaxis.get_major_locator().set_params(integer=True)
                ax[axes[1]].xaxis.get_major_locator().set_params(integer=True)
                ax[axes[0]].set_ylabel(r'$(ms_{X,j}^\alpha(f))^\alpha$')
                ax[axes[1]].set_ylabel(r'$(ms_{X,j}^\alpha(f))^\alpha$')
                ax[axes[0]].set_xlabel(r'$\alpha$')
                ax[axes[1]].set_xlabel(r'$\alpha$')
                return ax
        else:
            fig, ax = plt.subplots(len(raw_sens), 1)
            # fig.suptitle("Alpha sensitivity curves")
            for out, rs in enumerate(raw_sens):
                if len(raw_sens) > 1:
                    axes = [[out, 0]]
                else:
                    axes = [0]
                for inp, input in enumerate(input_name):
                    alpha_curves = alpha_curve(
                        rs.iloc[:, inp], tol=tol, max_alpha=max_alpha, alpha_step=alpha_step, dev=dev
                    )
                    if curve_equal_origin:
                        alpha_curves -= alpha_curves[0]
                    
                    alpha_curves_scl = alpha_curves
                    max_alpha_mean = max(abs(rs.iloc[:, inp]))
                    if max_alpha_mean != 0:
                        alpha_curves_scl = alpha_curves / max_alpha_mean
                    var_color = cmap(inp / rs.shape[1])
                    ax.plot(
                        alphas, alpha_curves, label=input, color=var_color
                    )
                    ax.axhline(
                        y=max_alpha_mean, color=var_color, linestyle="dashed"
                    )
                    ax.text(
                        alphas[-1], alpha_curves[-1],
                        input, 
                        color=var_color,
                        path_effects=[pe.withStroke(linewidth=1, foreground="gray")]
                    )
                    ax.text(
                        max([1,max_alpha - 20]), max_alpha_mean,
                        'max({0}): {1:.{2}f}'.format(input, max_alpha_mean, 2),
                        color=var_color,
                        path_effects=[pe.withStroke(linewidth=1, foreground="gray")]
                    )
                ax.set_ylabel(r'$(ms_{X,j}^\alpha(f))$')
                ax.set_xlabel(r'$\alpha$')
                ax.title.set_text(title)
                ax.xaxis.get_major_locator().set_params(integer=True)
                return ax
    elif kind == 'bar':
        fig, ax = plt.subplots(len(raw_sens), 1)
        for out, rs in enumerate(raw_sens):
            # Calculate scale factor
            N = rs.shape[0]
            alpha_values = empty(rs.shape[1])
            var_color = []
            if alpha_bar == 'inf':
                for inp, input in enumerate(input_name):
                    alpha_values[inp] = max(abs(rs[input]))
                    var_color.append(cmap(inp / rs.shape[1]))
            else:
                for inp, input in enumerate(input_name):
                    scl_factor = 1
                    if not all(v == 0 for v in rs[input]):
                        scl_factor = 10 ** (max(floor(log10(abs(rs[rs != 0.0][input])))) + 1) 
                    scl_rs = abs(rs[input]) / scl_factor
                    alpha_values[inp] = scl_factor * sum(scl_rs ** alpha_bar / N) ** (1 / alpha_bar) 
                    if alpha_values[inp] == inf:
                        scl_factor *= 5
                        scl_rs = abs(rs) / scl_factor
                        alpha_values[inp] = scl_factor * sum(scl_rs ** alpha_bar / N) ** (1 / alpha_bar) 
                    var_color.append(cmap(inp / rs.shape[1]))
            alpha_order = alpha_values.argsort()
            alpha_values = alpha_values[alpha_order]
            var_color = [var_color[i] for i in alpha_order]
            input_name = [input_name[i] for i in alpha_order]
            ax.bar(input_name, alpha_values, color=var_color)
            ax.set_ylabel(r'$(ms_{X,j}^\alpha(f))$')
            ax.set_xlabel('Input variables')
            ax.title.set_text(title + f'. Alpha: {alpha_bar}')
            return ax
        
def alpha_curve(
    raw_sens: list,
    tol: float = None,
    max_alpha: int = 100,
    alpha_step: int = 1,
    dev: str = "cpu",
    use_torch: bool = False
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
            
            from numpy import (
                array,
                arange,
                floor,
                log10,
                abs,
                empty, 
                inf
            )

            def float_array(x, dev="cpu") -> array:
                return array(x.numpy())

    else:
        from numpy import (
            array,
            arange,
            floor,
            log10,
            abs,
            empty, 
            inf
        )

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
        alpha_curve[i,] = scl_factor * sum(scl_rs ** a / N) ** (1 / a) 
        if alpha_curve[i,] == inf:
            scl_factor *= 5
            scl_rs = abs(raw_sens) / scl_factor
            alpha_curve[i,] = scl_factor * sum(scl_rs ** a / N) ** (1 / a) 
        if i > 1:
            if alpha_curve[i,] < alpha_curve[i-1,]:
                scl_factor /= 10
                scl_rs = abs(raw_sens) / scl_factor
                alpha_curve[i,] = scl_factor * sum(scl_rs ** a / N) ** (1 / a) 
    alpha_curve = float_array(alpha_curve)
    return alpha_curve

def interaction_invariant_mlp(
    wts,
    bias,
    actfunc,
    X: pd.core.frame.DataFrame,
    y: pd.core.frame.DataFrame,
    N1: int = 20,
    N2: int = 20,
    x1_limit = None,
    x2_limit = None,
    dev: str = "cpu",
    use_torch: bool = False,
    scaler = None,
    w0 = 0.0
    ):
    """Obtain invariant of a MLP model to analyze interaction

    Args:
        wts (list[float]): list of weight matrixes of the MLP layers.
        bias (list[float]): list of bias vectors of the MLP layers.
        actfunc (list[str]): list of names of the activation function of the MLP layers.
        X (pd.core.frame.DataFrame): data.frame with the input data
        y (pd.core.frame.DataFrame): data.frame with the output data
        x1_name (str): name of the first variable
        x2_name (str): name of the second variable
        N1 (int): number of points to create the grid of the first variable
        N2 (int): number of points to create the grid of the second variable
        dev (str, optional): device where calculations shall be performed ("gpu" or "cpu"). 
            Only available if pytorch installation could be found. Defaults to "cpu".
        use_torch (bool, optional): Flag to indicate if pytorch Tensor shall be used. Defaults to False.
        w0 (float, optional): Regularization value. If 0.0, function calculates the exact invariant on the grid. 
            Defaults to 0.0.

    Returns:
        DataFrame: Invariant value in each point of the 2D grid
    """
    # Calculate predictions of model for the grid
    # Import necessary modules based on processor, use torch when processor is gpu or use_torch is True
    if dev == "gpu" or use_torch:
        try:
            from torch import (
                eye as identity,
                ones,
                tensor,
                device,
                matmul,
                from_numpy,
                is_tensor,
                stack,
                exp,
                zeros
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
                array,
                ones,
                array,
                matmul,
                exp,
                zeros
            )

            def float_array(x, dev="cpu") -> array:
                return array(x.numpy())

            use_torch = False

    else:
        from numpy import (
            identity,
            array,
            ones,
            array,
            matmul,
            exp,
            zeros
        )

        def float_array(x, dev="cpu") -> array:
            return array(x)

        use_torch = False
        
    # Create grid of values to calculate the sensitivities on
    x1_min, x2_min = X.min(axis=0)
    x1_max, x2_max = X.max(axis=0)
    if x1_limit is not None:
        x1_min, x1_max = x1_limit
    if x2_limit is not None:
        x2_min, x2_max = x2_limit
    x1_grid = linspace(x1_min, x1_max, num=N1)
    x2_grid = linspace(x2_min, x2_max, num=N2)
    X1, X2 = meshgrid(x1_grid, x2_grid)
    X = pd.DataFrame(vstack([X1.ravel(), X2.ravel()]).T, columns=X.columns)
    
    # Calculate first, second and third derivatives
    jacmlp = jacobian_mlp(wts, bias, actfunc, X, y, dev=dev, use_torch=use_torch)
    hessmlp = hessian_mlp(wts, bias, actfunc, X, y, dev=dev, use_torch=use_torch)
    jerkmlp = jerkian_mlp(wts, bias, actfunc, X, y, dev=dev, use_torch=use_torch)
        
    # Structure of the mlp model
    mlpstr = [wts[0].shape[0]] + [lyr.shape[1] for lyr in wts]

    # Activation functions for each neuron layer
    actfunc = [activation_function(af, use_torch) for af in actfunc]

    # Weights of input layer
    W = [identity(X.shape[1])]

    # Input of input layer
    # inputs = [np.hstack((ones((len(X_train),1), dtype=int), X_train))]
    Z = [matmul(float_array(X, dev), float_array(W[0], dev))]

    # Output of input layer
    O = [actfunc[0](Z[0])]

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

    
    # Obtain raw derivatives
    jacobian = jacmlp.raw_sens[0]
    hessian  = hessmlp.raw_sens[0]
    jerkian  = jerkmlp.raw_sens[0]
    
    # Unscale variable if scaler is passed
    if scaler is not None:
        X = scaler.inverse_transform(vstack([x1_grid, x2_grid]).T)
        x1_grid = X[:, 0]
        x2_grid = X[:, 1]
    
    # Calculate invariant
    x = X.columns[0]
    y = X.columns[1]
    
    # Regularization
    reg_values = (jacobian[x]**2 * jacobian[y]**2).values.reshape(x1_grid.shape[0], x2_grid.shape[0])
    w = pd.DataFrame(0, index=round(x1_grid,1), columns=round(x2_grid,1))
    w = w0 * trapz(trapz(reg_values, axis=0, x=x1_grid), axis=0, x=x2_grid) / abs((x1_grid.max() - x1_grid.min()) * (x2_grid.max() - x2_grid.min()))
    reg = w * exp(-jacobian[x]**2 * jacobian[y]**2 / w)
    
    # Invariant calculates
    invariant_values =  abs(jacobian[x] * jacobian[y]**2 * jerkian[x, x, y] - 
                            jacobian[x]**2 * jacobian[y] * jerkian[x, y, y] - 
                            jacobian[y]**2 * hessian[x, y] * hessian[x, x] + 
                            jacobian[x]**2 * hessian[x, y] * hessian[y, y]) / (jacobian[x]**2 * jacobian[y]**2 + reg)
    invariant_values = nan_to_num(invariant_values.values.reshape(x1_grid.shape[0], x2_grid.shape[0]), 0)
    invariant_values = pd.DataFrame(invariant_values, index=round(x1_grid,1), columns=round(x2_grid,1))
    # integrate invariant
    inv = pd.DataFrame(None, index=round(x1_grid,1), columns=round(x2_grid,1))
    for i in range(inv.shape[0]-1):
        for j in range(inv.shape[1]-1):
            inv.iloc[i, j] = trapz(trapz(invariant_values.iloc[i:(i+2), j:(j+2)], x=x1_grid[i:(i+2)]), x=x2_grid[j:(j+2)])
    inv = inv.iloc[:-1, :-1]
            
    return Interaction_invariant(inv.fillna(0), 
                                O[-1].reshape(x1_grid.shape[0], x2_grid.shape[0])[:-1, :-1].reshape(-1), 
                                pd.DataFrame((jacobian[x]**2 * jacobian[y]**2 + reg).values.reshape(x1_grid.shape[0], x2_grid.shape[0]), index=round(x1_grid,1), columns=round(x2_grid,1)), 
                                O[-1].reshape(-1), 
                                jacmlp.input_name, jacmlp.output_name, mlpstr, 'Complete')

def interaction_invariant_mlp_sparse(
    wts,
    bias,
    actfunc,
    X: pd.core.frame.DataFrame,
    y: pd.core.frame.DataFrame,
    x1_name: str,
    x2_name: str,
    N1: int = 20,
    N2: int = 20,
    x1_limit = None,
    x2_limit = None,
    dev: str = "cpu",
    use_torch: bool = False,
    scaler = None,
    p = 2.0,
    w0 = 0.0
    ):
    """Obtain invariant of a MLP model to analyze interaction

    Args:
        wts (list[float]): list of weight matrixes of the MLP layers.
        bias (list[float]): list of bias vectors of the MLP layers.
        actfunc (list[str]): list of names of the activation function of the MLP layers.
        X (pd.core.frame.DataFrame): data.frame with the input data
        y (pd.core.frame.DataFrame): data.frame with the output data
        x1_name (str): name of the first variable
        x2_name (str): name of the second variable
        N1 (int): number of points to create the grid of the first variable
        N2 (int): number of points to create the grid of the second variable
        dev (str, optional): device where calculations shall be performed ("gpu" or "cpu"). 
            Only available if pytorch installation could be found. Defaults to "cpu".
        use_torch (bool, optional): Flag to indicate if pytorch Tensor shall be used. Defaults to False.
        p (float, optional): Choice of p for Lp-norm of the output space.
        w0 (float, optional): Regularization value. If 0.0, function calculates the exact invariant on the grid. 
            Defaults to 0.0.

    Returns:
        DataFrame: Invariant value in each point of the 2D grid
    """
    # Calculate predictions of model for the grid
    # Import necessary modules based on processor, use torch when processor is gpu or use_torch is True
    if dev == "gpu" or use_torch:
        try:
            from torch import (
                exp,
                zeros
            )

            use_torch = True

        except ImportError:
            
            from numpy import (
                exp,
                zeros
            )

            use_torch = False

    else:
        from numpy import (
            exp,
            zeros
        )

        use_torch = False
        
    # Structure of the mlp model
    mlpstr = [wts[0].shape[0]] + [lyr.shape[1] for lyr in wts]
    
    # Create grid of values to calculate the sensitivities on
    x1_min, x2_min = X[[x1_name, x2_name]].min(axis=0)
    x1_max, x2_max = X[[x1_name, x2_name]].max(axis=0) * 1.0001
    if x1_limit is not None:
        x1_min, x1_max = x1_limit
    if x2_limit is not None:
        x2_min, x2_max = x2_limit
    x1_grid = linspace(x1_min, x1_max, num=N1)
    x2_grid = linspace(x2_min, x2_max, num=N2)
    
    # Calculate first, second and third derivatives
    jacmlp = jacobian_mlp(wts, bias, actfunc, X, y, dev=dev, use_torch=use_torch)
    hessmlp = hessian_mlp(wts, bias, actfunc, X, y, dev=dev, use_torch=use_torch)
    jerkmlp = jerkian_mlp(wts, bias, actfunc, X, y, dev=dev, use_torch=use_torch)
    
    # Obtain raw derivatives
    jacobian = jacmlp.raw_sens
    hessian  = hessmlp.raw_sens
    jerkian  = jerkmlp.raw_sens
    
    # Calculate invariant
    x = x1_name
    y = x2_name
    
    # Iterates over each output  
    invariant_values = zeros(jacobian[0].shape[0])
    for jac, hes, jerk in zip(jacobian, hessian, jerkian):
        # Regularization
        reg_values = jac[x]**2 * jac[y]**2
        w = w0 * reg_values.mean()
        reg = w * exp(-jac[x]**2 * jac[y]**2 / w)
        
        # Invariant
        invariant_temp = (abs(jac[x] * jac[y]**2 * jerk[x, x, y] - 
                                jac[x]**2 * jac[y] * jerk[x, y, y] - 
                                jac[y]**2 * hes[x, y] * hes[x, x] + 
                                jac[x]**2 * hes[x, y] * hes[y, y]) / (jac[x]**2 * jac[y]**2 + reg)) ** p
        invariant_values += nan_to_num(invariant_temp, 0)
    invariant_values = invariant_values ** (1.0 / p)
    
    invariant = zeros((N1-1, N2-1))
    
    x1_pos = ((X[x1_name] - x1_min) / abs(x1_grid[1] - x1_grid[0])).astype(int).values
    x2_pos = ((X[x2_name] - x2_min) / abs(x2_grid[1] - x2_grid[0])).astype(int).values
    for i, j, I_L in zip(x1_pos, x2_pos, invariant_values):
        if I_L > invariant[i, j]:
            invariant[i, j] = I_L 
    cell_area = abs((x1_grid[1] - x1_grid[0]) * (x2_grid[1] - x2_grid[0]))
    invariant *= cell_area
    inv = pd.DataFrame(invariant, index=round(x1_grid[:-1],1), columns=round(x2_grid[:-1],1))
    return Interaction_invariant(inv, None, None, None, 
                                jacmlp.input_name, jacmlp.output_name, mlpstr, 'Sparse')

from numpy import log
import copy
from plotly.subplots import make_subplots
# Define self class
class Interaction_invariant:
    def __init__(
        self,
        inv,
        pred,
        jac,
        pred_jac,
        input_name,
        output_name,
        mlp_struct,
        invariant_type
    ):
        self.__inv = inv
        self.__pred = pred
        self.__jac = jac
        self.__pred_jac = pred_jac
        self.__input_name = input_name
        self.__output_name = output_name
        self.__mlp_struct = mlp_struct
        self.__invariant_type = invariant_type
    @property
    def inv(self):
        return self.__inv
    
    @property
    def values(self):
        return self.__inv.values
    
    @property
    def index(self):
        return self.__inv.index
    
    @property
    def columns(self):
        return self.__inv.columns

    @property
    def pred(self):
        return self.__pred
    
    @property
    def pred_jac(self):
        return self.__pred_jac

    @property
    def jac(self):
        return self.__jac

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
    
    @property
    def invariant_type(self):
        return self.__invariant_type
    
    def __repr__(self) -> str:
        return self.inv.__repr__()
    
    def __str__(self):
        print(f"{self.interaction_invariant} interaction invariant of {self.mlp_struct} MLP network.\n")
        return self.inv.__str__()
    
    def __copy__(self):
        return Interaction_invariant(
            self.__inv,
            self.__pred,
            self.__jac,
            self.__input_name,
            self.__output_name,
            self.__mlp_struct,
            self.__interaction_invariant
        )

    def sum(self):
        return self.inv.values.sum()
    
    def summary(self):
        print(f"{self.interaction_invariant} Interaction invariant of {self.mlp_struct} MLP network.\n")
        print("Sum of the invariant in the whole grid: ", self.inv.values.sum())
        

    def info(self):
        print(f"{self.interaction_invariant} Interaction invariant of {self.mlp_struct} MLP network.\n")
        print(self.inv)
    
    def plot(self, plot_type='3d', scale_inv='lineal', scale_jac='lineal', color_scale=None, text_auto=False, marker_opacity=0.3):
        
        ### Prepare data for 3d plot
        inv = self.inv
        # Create grid of values to calculate the sensitivities on
        X1, X2 = meshgrid(array(inv.index), array(inv.columns))
        if self.invariant_type == "Complete":
            jac = 1 / self.jac
            X1j, X2j = meshgrid(array(jac.index), array(jac.columns))
        if scale_inv == 'log':
            inv = pd.DataFrame(log(1 + inv.values), index=inv.index, columns=inv.columns)
        
        if scale_jac == 'log' and self.invariant_type == "Complete":
            jac = pd.DataFrame(log(1 + jac.values), index=jac.index, columns=jac.columns)
        
        # Set color scale as wanted
        figure1 = None
        figure4 = None
        figure2 = None
        if color_scale is None:
            if self.invariant_type == "Complete":
                figure1 = px.scatter_3d(x=X1.ravel(), 
                                        y=X2.ravel(), 
                                        z=self.pred, 
                                        color=inv.values.ravel(), 
                                        opacity=marker_opacity,
                                        labels=dict(
                                            x= self.input_name[0],
                                            y= self.input_name[1],
                                            z= self.output_name[0],
                                            color= 'Invariant'
                                        ),
                                        title=f'Invariant along the input space. Invariant = {inv.values.sum()}')
                
                figure4 = px.scatter_3d(x=X1j.ravel(), 
                                        y=X2j.ravel(), 
                                        z=self.pred_jac, 
                                        color=jac.values.ravel(), 
                                        opacity=marker_opacity,
                                        labels=dict(
                                            x= self.input_name[0],
                                            y= self.input_name[1],
                                            z= self.output_name[0],
                                            color= r'J_x \times J_y'
                                        ),
                                        title='Unstable points along input space')
                
                figure2 = px.imshow(jac, 
                                    x=jac.index,
                                    y=jac.columns,
                                    text_auto=text_auto, 
                                    labels=dict(
                                        x= self.input_name[0],
                                        y= self.input_name[1],
                                        color= 'J_x x J_y'
                                    )).update_layout(
                                        scene=dict(
                                            xaxis_title=self.input_name[0], 
                                            yaxis_title=self.input_name[1], 
                                        ),
                                        legend=dict(title=r'J_x \times J_y'),
                                        title='Unstable points along input space'
                                    )
                                    
            figure3 = px.imshow(inv, 
                                x=inv.index,
                                y=inv.columns,
                                text_auto=text_auto, 
                                labels=dict(
                                    x= self.input_name[0],
                                    y= self.input_name[1],
                                    color= 'Invariant'
                                )).update_layout(
                                    scene=dict(
                                        xaxis_title=self.input_name[0], 
                                        yaxis_title=self.input_name[1], 
                                    ),
                                    legend=dict(title='Invariant'),
                                    title='Invariant along input space'
                                )
            
        else:
            if self.invariant_type == "Complete":
                figure1 = px.scatter_3d(x=X1.ravel(), 
                                        y=X2.ravel(), 
                                        z=self.pred, 
                                        color=inv.values.ravel(), 
                                        opacity=marker_opacity,
                                        labels=dict(
                                            x= self.input_name[0],
                                            y= self.input_name[1],
                                            z= self.output_name[0],
                                            color= 'Invariant'
                                        ),
                                        title='Invariant along the input space',
                                        color_continuous_scale=color_scale)
                
                figure4 = px.scatter_3d(x=X1j.ravel(), 
                                        y=X2j.ravel(), 
                                        z=self.pred_jac, 
                                        color=jac.values.ravel(), 
                                        opacity=marker_opacity,
                                        labels=dict(
                                            x= self.input_name[0],
                                            y= self.input_name[1],
                                            z= self.output_name[0],
                                            color= r'J_x \times J_y'
                                        ),
                                        title='Unstable points along input space',
                                        color_continuous_scale=color_scale)
                
                figure2 = px.imshow(jac, 
                                    x=jac.index,
                                    y=jac.columns,
                                    text_auto=text_auto,
                                    color_continuous_scale=color_scale,
                                    labels=dict(
                                        x= self.input_name[0],
                                        y= self.input_name[1],
                                        color= 'J_x x J_y'
                                    )).update_layout(
                                        scene=dict(
                                            xaxis_title=self.input_name[0], 
                                            yaxis_title=self.input_name[1], 
                                        ),
                                        legend=dict(title=r'J_x \times J_y'),
                                        title='Unstable points along input space'
                                    )
            figure3 = px.imshow(inv, 
                                x=inv.index,
                                y=inv.columns,
                                text_auto=text_auto, 
                                labels=dict(
                                    x= self.input_name[0],
                                    y= self.input_name[1],
                                    color= 'Invariant',
                                    color_continuous_scale=color_scale
                                )).update_layout(
                                    scene=dict(
                                        xaxis_title=self.input_name[0], 
                                        yaxis_title=self.input_name[1], 
                                    ),
                                    legend=dict(title='Invariant'),
                                    title='Invariant along input space'
                                )
        
        if self.invariant_type == "Complete":
            figure1 = figure1.update_layout(
                scene = dict(
                    xaxis_title=self.input_name[0], 
                    yaxis_title=self.input_name[1],
                    zaxis_title=self.output_name[0]
                    ),
                legend = dict(title='Invariant')
            )
            
            figure4 = figure4.update_layout(
                scene = dict(
                    xaxis_title=self.input_name[0], 
                    yaxis_title=self.input_name[1],
                    zaxis_title=self.output_name[0]
                    ),
                legend = dict(title=r'J_x \times J_y')
            )
        
        if self.invariant_type == "Complete":
            if plot_type == '3d':
                return figure1
            elif plot_type == '2d':
                return figure3
            elif plot_type == 'jacobian':
                return figure2
            elif plot_type == 'jacobian3d':
                return figure4
            else:
                raise ValueError(f'Plot type {plot_type} is not accepted. Valid plot types are "3d", "2d", "jacobian" and "jacobian3d".')
        else:
            return figure3
