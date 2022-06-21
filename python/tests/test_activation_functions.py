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
    