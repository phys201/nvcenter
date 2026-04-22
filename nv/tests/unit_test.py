import pytensor.tensor as pt

def dipole_field(X, Y, Z, x_d, y_d, z_d, m_vec):
    """
    Will compute the magnetic field from a dipole
    on a 2D grid

    Input
    -----
    X, Y, Z: 2D arrays representing the observation coordinates
    x_d, y_d, z_d: float: dipole position
    m_vec : array of length 3: dipole moment vector

    Returns
    -------
    Bx, By, Bz : 2D arrays, magnetic field components
    
    """
    mx, my, mz = m_vec

    Rx = X - x_d
    Ry = Y - y_d
    Rz = Z - z_d

    R2 = Rx**2 + Ry**2 + Rz**2
    R = np.sqrt(R2)

    # include a safety cutoff value to top R^3, R^5 terms from blowing up
    lim = 1e-12
    R = np.maximum(R, lim)

    m_dot_R = mx*Rx + my*Ry + mz*Rz

    Bx = 3.0 * m_dot_R * Rx / R**5 - mx / R**3
    By = 3.0 * m_dot_R * Ry / R**5 - my / R**3
    Bz = 3.0 * m_dot_R * Rz / R**5 - mz / R**3

    return Bx, By, Bz