#!/usr/bin/env python

"""Tests for `neuralsens` package."""

import pytest

from neuralsens import activation_functions as ac_f

def test_fill_diagonal_torch():
    """pytest test to check if torch tensor diagonal can be customized."""
    try:
        import torch 
        # Create zero tensor and change its diagonal
        z = torch.zeros((3,3,3))
        diagonal = torch.Tensor([1, 2, -1])
        z = ac_f.fill_diagonal_torch(z, diagonal)
        
        # Create tensor to check above operation
        z_check = torch.Tensor([[[ 1.,  0.,  0.],
                                 [ 0.,  0.,  0.],
                                 [ 0.,  0.,  0.]],
                                [[ 0.,  0.,  0.],
                                 [ 0.,  2.,  0.],
                                 [ 0.,  0.,  0.]],
                                [[ 0.,  0.,  0.],
                                 [ 0.,  0.,  0.],
                                 [ 0.,  0., -1.]]])

        assert torch.equal(z, z_check), "Function to fill diagonal of 3D-Tensor not working as expected"
        
        # Create zero tensor and change its diagonal
        z = torch.zeros((4,4,4,4))
        diagonal = torch.Tensor([5, -7, 0, 2])
        z = ac_f.fill_diagonal_torch(z, diagonal)
        
        # Create tensor to check above operation
        z_check = torch.Tensor([[[[ 5.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]],
                                 [[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]],
                                 [[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]],
                                 [[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]]],
                                [[[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]],
                                 [[ 0.,  0.,  0.,  0.],
                                  [ 0., -7.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]],
                                 [[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]],
                                 [[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]]],
                                [[[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]],
                                 [[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]],
                                 [[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]],
                                 [[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]]],
                                [[[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]],
                                 [[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]],
                                 [[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.]],
                                 [[ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  0.],
                                  [ 0.,  0.,  0.,  2.]]]])

        # Check tensors are equal
        assert torch.equal(z, z_check), "Function to fill diagonal of 4D-Tensor not working as expected"

    except ImportError:
        print(
            "torch installation could not be found, test shall not be passed"
        )

def test_activation_function_torch():
    """ Test to check if activation function performs as expected for pytorch Tensor. """
    try:
        from torch import sigmoid, tanh, relu, rand, equal
        from random import randrange
        def identity(x):
            return x
        act_fct = {"logistic": sigmoid,
                    "sigmoid": sigmoid,
                    "expit": sigmoid,
                    "identity": identity,
                    "tanh": tanh,
                    "relu": relu}
        for af, fct in act_fct.items(): 
            random_size = randrange(5, 20)
            acf = ac_f.activation_function(af, use_torch=True)
            tns = rand(random_size)
            tns_acf = acf(tns)
            tns_check = tns
            for i in range(random_size):
                tns_check[i] = fct(tns[i])
            assert equal(tns_acf, tns_check), "Activation function of the package not working correctly"
        
    except ImportError:
        print(
            "torch installation could not be found, test shall not be passed"
        )

def test_activation_function_numpy():
    """ Test to check if activation function performs as expected for numpy vector. """
    from numpy import tanh, array_equal
    from numpy.random import rand
    from scipy.special import expit
    from random import randrange
    def identity(x):
        return x
    def relu(x):
        return x if x > 0 else 0
    act_fct = {"logistic": expit,
                "sigmoid": expit,
                "expit": expit,
                "identity": identity,
                "tanh": tanh,
                "relu": relu}
    for af, fct in act_fct.items(): 
        random_size = randrange(5, 20)
        acf = ac_f.activation_function(af, use_torch=False)
        tns = rand(random_size)
        tns_acf = acf(tns)
        tns_check = tns
        for i in range(random_size):
            tns_check[i] = fct(tns[i])
        assert array_equal(tns_acf, tns_check), "Activation function of the package not working correctly"
