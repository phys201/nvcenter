from numpy import random

def mean_fluorescence(
    X, Y, Z, 
    x_d, y_d, z_d, 
    m_vec, 
    n_vec, 
    beta0, 
    C, 
    Gamma, 
    delta0
):
    """
    Computes the fluorescence for the image (noiseless)

    inputs
    ------
    UPDATE

    Returns
    -------
    UPDATE
    """

    Bx, By, Bz = dipole_field(X, Y, Z, x_d, y_d, z_d, m_vec)

    nx, ny, nz = n_vec
    B_par = nx*Bx + ny*By + nz*Bz

    Delta = delta0 - B_par

    lorentzian = 1.0 / (1.0 + (Delta/Gamma)**2)
    mu = beta0 * (1.0 - C*lorentzian)

    return mu, B_par, Delta