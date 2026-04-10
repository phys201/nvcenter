from numpy import random

def dipole_field_pt(X, Y, Z, x_d, y_d, z_d, m_x, m_y, m_z):
    Rx = X - x_d
    Ry = Y - y_d
    Rz = Z - z_d

    R2 = Rx**2 + Ry**2 + Rz**2
    R2 = pt.maximum(R2, 1e-12)
    R = pt.sqrt(R2)

    m_dot_R = m_x * Rx + m_y * Ry + m_z * Rz

    Bx = 3.0 * m_dot_R * Rx / R**5 - m_x / R**3
    By = 3.0 * m_dot_R * Ry / R**5 - m_y / R**3
    Bz = 3.0 * m_dot_R * Rz / R**5 - m_z / R**3

    return Bx, By, Bz