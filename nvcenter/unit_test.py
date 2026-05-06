import numpy as np
import pytensor as pt
import arviz as az
import pymc as pm
import pytest
import data_io
import odmr_analysis

# test scalar convolution
def test_zero_input():
    size = 10
    test_K = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    zero = np.zeros(size)
    assert np.array_equal(convolution.scalar(test_K, 0), zero)

def test_simulate_data():
    size = 10
    test_K = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    zero = np.zeros(size)
    assert np.array_equal(convolution.scalar(test_K, 0), zero)

def test_posterior():
    