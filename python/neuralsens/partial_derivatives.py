import pandas as pd
from numpy import triu_indices, meshgrid
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import warnings
from neuralsens.activation_functions import (
    activation_function,
    der_activation_function,
    der_2_activation_function,
    der_3_activation_function,
)
from numpy import nanmax, inf

plt.style.use("ggplot")


def jacobian_mlp(
    wts: list[float],
    bias: list[float],
    actfunc: list[str],
    X: pd.core.frame.DataFrame,
    y: pd.core.frame.DataFrame,
    sens_origin_layer: int = 0,
    sens_end_layer: str = "last",
    sens_origin_input: bool = True,
    sens_end_input: bool = False,
    dev: str = "cpu",
    use_torch: bool = False,
):
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
            warnings.warn(
                "Pytorch installation could not be found, numpy would be used instead",
                ImportWarning,
            )
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
    meansquareSens = mean(square(D_accum[counter]), axis=0)

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
    def __init__(
        self,
        sens: list,
        raw_sens: list,
        mlp_struct: list[int],
        X: pd.core.frame.DataFrame,
        input_name: list[str],
        output_name: list[str],
    ):
        self.__sens = sens
        self.__raw_sens = raw_sens
        self.__mlp_struct = mlp_struct
        self.__X = X
        self.__input_names = input_name
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
        sensitivity_plots(self)

    def featurePlots(self):
        pass

    def timePlots(self):
        pass


def hessian_mlp(
    wts: list[float],
    bias: list[float],
    actfunc: list[str],
    X: pd.core.frame.DataFrame,
    y: pd.core.frame.DataFrame,
    sens_origin_layer: int = 0,
    sens_end_layer: str = "last",
    sens_origin_input: bool = True,
    sens_end_input: bool = False,
    dev: str = "cpu",
    use_torch: bool = False,
):
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
            warnings.warn(
                "Pytorch installation could not be found, numpy would be used instead",
                ImportWarning,
            )
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
    def __init__(
        self,
        sens: list,
        raw_sens: list,
        mlp_struct: list[int],
        X: pd.core.frame.DataFrame,
        input_name: list[str],
        output_name: list[str],
    ):
        self.__sens = sens
        self.__raw_sens = raw_sens
        self.__mlp_struct = mlp_struct
        self.__X = X
        self.__input_names = input_name
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
    wts: list[float],
    bias: list[float],
    actfunc: list[str],
    X: pd.core.frame.DataFrame,
    y: pd.core.frame.DataFrame,
    sens_origin_layer: int = 0,
    sens_end_layer: str = "last",
    sens_origin_input: bool = True,
    sens_end_input: bool = False,
    dev: str = "cpu",
    use_torch: bool = False,
):
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
            )
            from numpy import ndarray, asarray

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

            use_torch = True

        except ImportError:
            warnings.warn(
                "Pytorch installation could not be found, numpy would be used instead",
                ImportWarning,
            )
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
    Z = [matmul(float_array(X, device), W[0])]

    # Output of input layer
    O = [actfunc[0](Z[0])]

    # First Derivative of input layer
    D = [float_array([deractfunc[0](Z[0][irow,]) for irow in r_samples], dev=device)]

    # Second derivative of input layer
    D2 = [float_array([der2actfunc[0](Z[0][irow,]) for irow in r_samples], dev=device)]

    # Third derivative of input layer
    D3 = [float_array([der3actfunc[0](Z[0][irow,]) for irow in r_samples], dev=device)]

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
                [deractfunc[lyr](Z[lyr][irow,]) for irow in r_samples], dev=device
            )
        )

        # Calculate second derivative of each layer
        D2.append(
            float_array(
                [der2actfunc[lyr](Z[lyr][irow,]) for irow in r_samples], dev=device
            )
        )

        # Calculate third derivative of each layer
        D3.append(
            float_array(
                [der3actfunc[lyr](Z[lyr][irow,]) for irow in r_samples], dev=device
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

    M = [zeros((n_samples, n_inputs, n_inputs, n_inputs, n_inputs))]
    # N = [zeros((n_samples,n_inputs,n_inputs,n_inputs,n_inputs))]
    J = [D3[0]]

    counter = 0
    for layer in range(sens_origin_layer + 1, sens_end_layer):
        counter += 1

        # First derivatives
        z_ = D_[counter - 1] @ W[layer][1:, :]
        D_.append(z_ @ D[layer])

        # Second derivatives
        q = matmul(
            matmul(H[counter - 1], W[layer][1:, :]).swapaxes(0, 1), D[counter]
        ).swapaxes(0, 1)
        h = matmul(z_, matmul(z_, D2[layer].swapaxes(0, 1)).swapaxes(0, 2)).swapaxes(
            0, 1
        )
        Q.append(q)
        H.append(h + q)

        # Third derivatives
        m = matmul(
            matmul(J[counter - 1], W[layer][1:, :]).swapaxes(0, 2), D[counter]
        ).swapaxes(0, 2)
        n = matmul(
            z_,
            matmul(
                matmul(
                    np.broadcast_to(
                        np.expand_dims(q, axis=4), list(q.shape) + [q.shape[-1]]
                    )
                    .swapaxes(0, 2)
                    .swapaxes(0, 3),
                    D2[layer][:, :, 0, :],
                ).swapaxes(1, 3),
                D2[layer][:, 0, :, :],
            ).swapaxes(0, 3),
        ).swapaxes(0, 2)
        j = matmul(
            z_,
            matmul(z_, matmul(z_, D3[layer].swapaxes(0, 2)).swapaxes(0, 3)).swapaxes(
                1, 3
            ),
        ).swapaxes(0, 2)

        M.append(m)
        # N.append(n)
        J.append(
            j
            + m
            + n
            + n.swapaxes(2, 3).swapaxes(1, 2)
            + n.swapaxes(1, 2).swapaxes(2, 3)
        )

    if sens_end_input:
        raw_sens = M[counter]
    else:
        raw_sens = J[counter]

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
                dev=device,
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
    if D_[counter].shape[2] > 1:
        output_name = ["_".join([y.name, lev]) for lev in y.unique()]
    return Jerkian_mlp(sens, raw_sens, mlpstr, X, input_name, output_name)


# Define self class
class Jerkian_mlp:
    def __init__(
        self,
        sens: list,
        raw_sens: list,
        mlp_struct: list[int],
        X: pd.core.frame.DataFrame,
        input_name: list[str],
        output_name: list[str],
    ):
        self.__sens = sens
        self.__raw_sens = raw_sens
        self.__mlp_struct = mlp_struct
        self.__X = X
        self.__input_names = input_name
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


def sensitivity_plots(jacobian: Jacobian_mlp):
    fig, ax = plt.subplots(3, len(jacobian.raw_sens))
    fig.suptitle("Sensitivity plots")
    for out, raw_sens in enumerate(jacobian.raw_sens):
        if len(ax.shape) > 1:
            axes = [[out, 0], [out, 1], [out, 2]]
        else:
            axes = [0, 1, 2]
        sens = jacobian.sens[out]

        # Plot mean-std plot
        ax[axes[0]].set_xlim(
            [
                min(sens["mean"]) - 0.2 * max(abs(sens["mean"])),
                max(sens["mean"]) + 0.2 * max(abs(sens["mean"])),
            ]
        )
        ax[axes[0]].hlines(
            y=0,
            xmin=min(sens["mean"]) - 0.2 * max(abs(sens["mean"])),
            xmax=max(sens["mean"]) + 0.2 * max(abs(sens["mean"])),
            color="blue",
        )
        ax[axes[0]].vlines(x=0, ymin=0, ymax=1, color="blue")
        ax[axes[0]].scatter(x=0, y=0, s=150, c="blue")
        for i, txt in enumerate(sens.index.values.tolist()):
            ax[axes[0]].annotate(
                txt,
                xy=(sens["mean"][i], sens["std"][i]),
                xycoords="data",
                va="center",
                ha="center",
                fontsize="large",
                bbox=dict(boxstyle="round", fc="w", ec="gray"),
            )
        ax[axes[0]].set_xlabel("mean(Sens)")
        ax[axes[0]].set_ylabel("std(Sens)")
        ax[axes[0]].title.set_text("Sensitivity plots for output " + str(out))

        # Plot variable importance mean_sqaured
        colors = plt.cm.cmap_d["Blues_r"](
            sens["mean_squared"] * 0.5 / max(sens["mean_squared"])
        )
        sens.plot.bar(
            y="mean_squared", ax=ax[axes[1]], color=colors, legend=False, rot=0
        )
        ax[axes[1]].set_xlabel("Input variables")
        ax[axes[1]].set_ylabel("mean(Sens^2)")

        # Plot density of sensitivities
        raw_sens.plot.kde(ax=ax[axes[2]])
        ax[axes[2]].set_xlabel("Sens")
        ax[axes[2]].set_ylabel("density(Sens)")


def alpha_sens_curves(
    jacobian,
    tol: float = None,
    max_alpha: int = 100,
    alpha_step: int = 1,
    curve_equal_origin: bool = False,
    dev: str = "cpu",
    use_torch: bool = False,
):
    # Import necessary modules based on processor, use torch when processor is gpu or use_torch is True
    if dev == "gpu" or use_torch:
        try:
            from torch import arange

        except ImportError:
            warnings.warn(
                "Pytorch installation could not be found, numpy would be used instead",
                ImportWarning,
            )
            from numpy import (
                arange,
            )

    else:
        from numpy import (
            arange,
        )


    # Initialize variables
    raw_sens = jacobian.raw_sens
    input_name = jacobian.X.columns
    alphas = arange(1, max_alpha + 1, alpha_step)

    # Select the color map named rainbow
    cmap = cm.get_cmap(name="rainbow")

    # Go through each output although this analysis is meant only for regression
    fig, ax = plt.subplots(len(jacobian.raw_sens), 2)
    fig.suptitle("Alpha sensitivity curves")
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


def alpha_sens_curves(
    jacobian,
    tol: float = None,
    max_alpha: int = 100,
    alpha_step: int = 1,
    curve_equal_origin: bool = False,
    dev: str = "cpu",
    use_torch: bool = False,
    columns: list[str] = None 
):
    # Import necessary modules based on processor, use torch when processor is gpu or use_torch is True
    if dev == "gpu" or use_torch:
        try:
            from torch import arange

        except ImportError:
            warnings.warn(
                "Pytorch installation could not be found, numpy would be used instead",
                ImportWarning,
            )
            from numpy import (
                arange,
            )

    else:
        from numpy import (
            arange,
        )

    # Check if field in variable 
    if hasattr(jacobian, 'raw_sens'):
        # Initialize variables
        raw_sens = jacobian.raw_sens
        input_name = jacobian.X.columns
    else:
        input_name = columns
        if columns is None:
            input_name = ['X{}'.format(x) for x in (arange(0, jacobian.shape[1]))]
        raw_sens = [pd.DataFrame(jacobian, columns=input_name)]
    alphas = arange(1, max_alpha + 1, alpha_step)
    # Select the color map named rainbow
    cmap = cm.get_cmap(name="rainbow")

    # Go through each output although this analysis is meant only for regression
    fig, ax = plt.subplots(len(raw_sens), 2)
    fig.suptitle("Alpha sensitivity curves")
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


def alpha_curve(
    raw_sens: list,
    tol: float = None,
    max_alpha: int = 100,
    alpha_step: int = 1,
    dev: str = "cpu",
    use_torch: bool = False,
):
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
            warnings.warn(
                "Pytorch installation could not be found, numpy would be used instead",
                ImportWarning,
            )
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
    scl_factor = 10 ** (max(floor(log10(abs(raw_sens)))) + 1) 
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