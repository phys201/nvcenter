import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

def vonMisesPosterior(xbar, ybar, zbar, kappa, mu_x = 0, mu_y = 0, mu_z = 1):
    r2 = np.sqrt(xbar^2 + ybar^2 + zbar^2)
    xbar = xbar / r2
    ybar = ybar / r2
    zbar = zbar / r2
    C3 = kappa / (4*np.pi*np.sinh(kappa))

    arg = kappa * (xbar*mu_x+ybar*mu_y+zbar*mu_z)
    posterior = C3*np.exp(arg)
    return posterior

def vonMises_3D(xbar, ybar,zbar, kappa):
    """
    Will sample from 3D von-Mises Fisher distribution on S2 support

    Input
    -----
    xbar: x component of mean direction
    ybar: y component of mean direction
    zbar: z component of mean direction
    kappa : concentration; 1/kappa analagous to variance

    Returns
    -------
    (x, y) sampled from 3D von-Mises Fisher distribution
    
    """

    r2 = np.sqrt(xbar^2 + ybar^2 + zbar^2)
    xbar = xbar / r2
    ybar = ybar / r2
    zbar = zbar / r2
    mubar = np.array([[xbar,0,0],[0,ybar,0],[0,0,zbar]])

    C3 = kappa / (4*np.pi*np.sinh(kappa))

    with pm.Model() as vonMises_model:

        theta = pm.Uniform(0,2*np.pi)
        phi = pm.Uniform(-np.pi,np.pi)
        r = np.array([np.sin(phi)*np.cos(theta),0,0],[0,np.sin(phi)*np.sin(theta),0],[0,0,np.cos(phi)])

        vector = C3 * np.exp(kappa* mubar.T * r)