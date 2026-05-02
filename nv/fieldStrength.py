import numpy as np
import pytensor.tensor as pt

def vacPerm(reduced = 'true'):
    """
    Returns value of relative permittivity of the vacuum
    """
    mu_0 = 1.25663706127e-6 # T m / A
    mu0_OVER_4PI = mu_0 / (4.0 * np.pi) # ~1e-7 T m / A
    if reduced = 'false':
        return mu_0
    else:
        return MU0_OVER_4PI

def cylindrical_near(X, Y, Z, M):
    """
    Will compute the near field magnetic field from a solenoid

    Input
    -----
    X, Y, Z: 2D arrays representing displacement from magnetic moment vector
    m_vec : array of length 3: magnetic moment vector

    Returns
    -------
    Bx, By, Bz : 2D arrays, magnetic field components
    
    """
    
def dipole(X, Y, Z, m_vec):
    """
    Will compute the magnetic field from a dipole
    on a 2D grid

    Input
    -----
    X, Y, Z: 2D arrays representing displacement from dipole
    m_vec : array of length 3: dipole moment vector

    Returns
    -------
    Bx, By, Bz : 2D arrays, magnetic field components
    
    """
    mx, my, mz = m_vec

    Rx = X 
    Ry = Y 
    Rz = Z 

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

def fieldStrength(X, Y, Z, moment, fieldShape):
    field_at_pt = fieldShape(X, Y, Z, moment)
    return field_at_pt