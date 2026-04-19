from numpy import random

def mean_fluorescence_pt(X, Y, Z, x_d, y_d, z_d,
                         m_x, m_y, m_z,
                         beta0, C, Gamma, delta0, n_vec):
    
    Bx, By, Bz = dipole_field_pt(X, Y, Z, x_d, y_d, z_d, m_x, m_y, m_z)

    nx, ny, nz = n_vec
    B_par = nx * Bx + ny * By + nz * Bz

    Delta = delta0 - B_par
    L = 1.0 / (1.0 + (Delta / Gamma)**2)
    mu = beta0 * (1.0 - C * L)

    return mu, B_par, Delta