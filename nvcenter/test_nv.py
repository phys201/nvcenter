import numpy as np
import pytensor as pt
import arviz as az
import pymc as pm
import pytest
import nvcenter.data_io
import nvcenter.odmr_analysis

def test_floats():
    assert (0.1 + 0.2) == pytest.approx(0.3)
    
def test_arrays():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.9999, 2.0001, 3.0])
    assert a == pytest.approx(b)
    
def test_dipole_field():
    dipole_mag=0
    test_dir=[1,1,1]
    assert np.array_equal(nvcenter.odmr_analysis.dipole_field(test_dir[0], test_dir[1], test_dir[2],
                                                     0,0,0, dipole_mag), zero)

