import numpy as np
import pytest
import dipole_field
import dipole_field_pt
import mean_fluorescence
import mean_fluorescence_pt
import vonMise
# test scalar convolution
def test_zero_input():
    size = 10
    test_K = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    zero = np.zeros(size)
    assert np.array_equal(convolution.scalar(test_K, 0), zero)

def test_vonMise():
    k = [1,1,0]
    assert np.equal(vonMisesPosterior(k,1),0)