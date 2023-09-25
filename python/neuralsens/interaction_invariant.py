import pandas as pd
from neuralsens.partial_derivatives import jacobian_mlp, hessian_mlp, jerkian_mlp
from neuralsens.activation_functions import activation_function
from numpy import meshgrid, array
import plotly.express as px
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
                tensor,
                device,
                matmul,
                from_numpy,
                is_tensor,
                stack,
                exp,
                linspace,
                meshgrid,
                vstack,
                trapz,
                nan_to_num,
                round
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
                linspace,
                meshgrid,
                vstack,
                trapz,
                nan_to_num,
                round
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
            linspace,
            meshgrid,
            vstack,
            trapz,
            nan_to_num,
            round
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
                zeros,
                linspace,
                nan_to_num
            )

            use_torch = True

        except ImportError:
            
            from numpy import (
                exp,
                zeros,
                linspace,
                nan_to_num
            )

            use_torch = False

    else:
        from numpy import (
            exp,
            zeros,
            linspace,
            nan_to_num
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
    
    # Regularization
    # Cambiar a calcular el maximo de reg_values por cada celda 
    # Ese vector tiene el tamaño de numero de celdas
    # reg es el promedio de ese vector.
    reg_values = jacobian[x]**2 * jacobian[y]**2
    # reg_values_grid = max...
    w = w0 * reg_values.mean()
    reg = w * exp(-jacobian[x]**2 * jacobian[y]**2 / w)
    
    # Invariant
    invariant_temp = (abs(jacobian[x] * jacobian[y]**2 * jerkian[x, x, y] - 
                            jacobian[x]**2 * jacobian[y] * jerkian[x, y, y] - 
                            jacobian[y]**2 * hessian[x, y] * hessian[x, x] + 
                            jacobian[x]**2 * hessian[x, y] * hessian[y, y]) / (jacobian[x]**2 * jacobian[y]**2 + reg)) ** p
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



