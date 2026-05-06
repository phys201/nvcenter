import pytensor.tensor as pt
import numpy as np
import pymc as pm
import arviz as az

D_NV_MHZ = 2870.0
GAMMA_NV_MHZ_PER_T = 28024.951386
MU0 = 1.25663706127e-6
FOUR_PI = 4.0 * pt.pi
MU0_OVER_4PI = MU0 / (4 * pt.pi)

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
    lim = 1e-30
    R = np.maximum(R, lim)

    m_dot_R = mx*Rx + my*Ry + mz*Rz

    Bx = MU0_OVER_4PI * (3.0 * m_dot_R * Rx / R**5 - mx / R**3)
    By = MU0_OVER_4PI * (3.0 * m_dot_R * Ry / R**5 - my / R**3)
    Bz = MU0_OVER_4PI * (3.0 * m_dot_R * Rz / R**5 - mz / R**3)

    return Bx, By, Bz # will now be in Tesla

GAMMA_NV_MHZ_PER_T = 28024.95138616912 # MHz/T
D_NV_MHZ = 2870.0 # MHz

def mean_fluorescence_dipole(
    X, Y, Z, 
    x_d, y_d, z_d, 
    m_vec, 
    n_vec, 
    beta0, 
    C, 
    Gamma, 
    f_mw,
    branch_sign=-1
):
    """
    Computes the noiseless fluorescence image for a fixed microwave tone.

    Input
    ----------
    X, Y, Z : 2D ndarray
        Observation coordinates of the scan grid, in METERS

    x_d, y_d, z_d : float
        Cartesian coordinates of the dipole position, in METERS.

    m_vec : array of shape (3,)
        Dipole moment vector [m_x, m_y, m_z], in A m^2.

    n_vec : array-like of shape (3,)
        Unit vector giving the NV orientation axis.

    beta0 : float
        Background fluorescence level, in kc/s.

    C : float
        ODMR contrast parameter. Dimensionless, between 0, 1

    Gamma : float
        Lorentzian linewidth parameter, in MHz.

    f_mw : float
        Applied microwave frequency, in MHz.

    branch_sign : int, optional
        Chooses which ODMR branch is being driven.
        Use -1 for the lower-frequency branch and +1 for the upper-frequency
        branch.

    Returns
    -------
    mu : 2D ndarray
        Data for fluorescence image, in kc/s.

    B_par : 2D ndarray
        Magnetic field component projected along the NV axis, in TESLA.

    Delta : 2D ndarray
        Frequency detuning f_mw - f_nv at each pixel, in MHz.
    """

    Bx, By, Bz = dipole_field(X, Y, Z, x_d, y_d, z_d, m_vec)

    nx, ny, nz = n_vec
    B_par = nx*Bx + ny*By + nz*Bz

    f_nv = D_NV_MHZ + branch_sign * GAMMA_NV_MHZ_PER_T * B_par
    Delta = f_mw - f_nv

    lorentzian = 1.0 / (1.0 + (Delta/Gamma)**2)
    mu = beta0 * (1.0 - C*lorentzian)

    return mu, B_par, Delta
    
def rect_prism_field_zmag_pt(X, Y, Z,
                             x_c, y_c, z_c,
                             Lx, Ly, Lz,
                             Br,
                             eps=1e-30):
    """
    Magnetic field of a uniformly z-magnetized rectangular prism.
    Coordinates and lengths are in meters.
    Br is in Tesla.
    """

    ax = Lx / 2.0
    ay = Ly / 2.0
    az = Lz / 2.0

    xs = [X - (x_c - ax), X - (x_c + ax)]
    ys = [Y - (y_c - ay), Y - (y_c + ay)]
    zs = [Z - (z_c - az), Z - (z_c + az)]

    Bx_sum = 0.0
    By_sum = 0.0
    Bz_sum = 0.0

    for i in range(2):
        for j in range(2):
            for k in range(2):
                xi = xs[i]
                yj = ys[j]
                zk = zs[k]

                r = pt.sqrt(xi**2 + yj**2 + zk**2 + eps)
                s = (-1.0) ** (i + j + k)

                Bx_sum = Bx_sum + s * pt.log(r + yj + eps)
                By_sum = By_sum + s * pt.log(r + xi + eps)
                Bz_sum = Bz_sum + s * pt.arctan2(xi * yj, zk * r + eps)

    pref = Br / FOUR_PI

    Bx = -pref * Bx_sum
    By = -pref * By_sum
    Bz =  pref * Bz_sum

    return Bx, By, Bz

def mean_fluorescence_prism_pt(X, Y, Z,
                               x_c, y_c, z_c,
                               Lx, Ly, Lz,
                               Br,
                               beta0, C, Gamma, f_mw, n_vec,
                               branch_sign=-1):
    
    Bx, By, Bz = rect_prism_field_zmag_pt(
        X, Y, Z,
        x_c, y_c, z_c,
        Lx, Ly, Lz,
        Br
    )

    nx, ny, nz = n_vec
    B_par = nx * Bx + ny * By + nz * Bz

    f_nv = D_NV_MHZ + branch_sign * GAMMA_NV_MHZ_PER_T * B_par
    Delta = f_mw - f_nv

    L = 1.0 / (1.0 + (Delta / Gamma)**2)
    mu = beta0 * (1.0 - C * L)

    return mu, B_par, Delta

def simulate_data(x_d_true = 1e-6, y_d_true = 1e-6, z_d_true = 1e-6, m_dir = [0,1,1], m_mag_true = 3e-16, 
                  n_vec_true = [0,0,1], beta0_true = 145.0, C_true = 0.30, Gamma_true = 40.0, f_mw_true = 2770.0, 
                  branch_sign_true = -1, sigma_true = 2.5):

    # ---------------------------------------
    # Define some set parameters for the scan
    # ---------------------------------------
    
    n_pix = 45
    vx = np.linspace(30.0, 50.0, n_pix)
    vy = np.linspace(30.0, 50.0, n_pix)
    
    VX, VY = np.meshgrid(vx, vy)
    
    # ---------------------------------------
    # We have a known volt -> um conversion
    # ---------------------------------------
    volt_to_m = 0.02e-6
    X = VX * volt_to_m
    Y = VY * volt_to_m
    
    # ---------------------------------------
    # Our third dimension will be the image data
    # ---------------------------------------
    Z = np.zeros_like(X)
    
    m_dir = np.array(m_dir)
    m_dir=m_dir/np.linalg.norm(m_dir)
    n_vec_true = np.array(n_vec_true)
    m_vec_true = m_mag_true * m_dir

    mu_true, Bpar_true, Delta_true = mean_fluorescence_dipole(
    X, Y, Z,
    x_d_true, y_d_true, z_d_true,
    m_vec_true,
    n_vec_true,
    beta0_true,
    C_true,
    Gamma_true,
    f_mw_true,
    branch_sign_true)

    # And now draw noisy observations
    rng = np.random.default_rng(1)
    y_sim = rng.normal(loc=mu_true, scale=sigma_true, size=mu_true.shape)

    extents =[vx.min(), vx.max(), vy.max(), vy.min()]
    
    return y_sim, extents

def rectangular_prism_posterior(data, extents):
    
    Ny, Nx = data.shape
    x_min, x_max, y_min, y_max = extents[-4:]
    x_volt = np.linspace(x_min, x_max, Nx)
    y_volt = np.linspace(y_min, y_max, Ny)
    X_volt, Y_volt = np.meshgrid(x_volt, y_volt)
    
    volt_to_um = 0.067
    x_um = x_volt * volt_to_um
    y_um = y_volt * volt_to_um
    X_um = X_volt * volt_to_um
    Y_um = Y_volt * volt_to_um
    Z_um = np.zeros_like(X_um, dtype=float)
    
    with pm.Model() as full_prism_inference_model:

        X_shared = pm.Data("X_shared", (X_um * 1e-6).astype("float64"))
        Y_shared = pm.Data("Y_shared", (Y_um * 1e-6).astype("float64"))
        Z_shared = pm.Data("Z_shared", (Z_um * 1e-6).astype("float64"))
        y_shared = pm.Data("y_shared", data.astype("float64"))
    
        # Prism position
        x_c_um = pm.Normal("x_c_um", mu=0.0, sigma=0.25)
        y_c_um = pm.Normal("y_c_um", mu=0.0, sigma=0.25)
    
        x_c = x_c_um * 1e-6
        y_c = y_c_um * 1e-6
    
        # NV standoff
        log_h_top_um = pm.Normal(
            "log_h_top_um",
            mu=np.log(0.30),
            sigma=0.25,
        )
        h_top_um = pm.Deterministic("h_top_um", pt.exp(log_h_top_um))
        h_top = h_top_um * 1e-6
    
        z_c = -(h_top + Lz / 2.0)
    
        # Remanence magnitude, negative sign fixed
        log_Br_abs_T = pm.Normal(
            "log_Br_abs_T",
            mu=np.log(0.12),
            sigma=0.30,
        )
        Br_abs_T = pm.Deterministic("Br_abs_T", pt.exp(log_Br_abs_T))
        Br_T = pm.Deterministic("Br_T", -Br_abs_T)
    
        # Magnetic field
        Bx, By, Bz = rect_prism_field_zmag_pt(
            X_shared,
            Y_shared,
            Z_shared,
            x_c,
            y_c,
            z_c,
            Lx,
            Ly,
            Lz,
            Br_T,
        )
    
        nx, ny, nz = n_vec
        B_par = nx * Bx + ny * By + nz * Bz
    
        B_par_mT = pm.Deterministic("B_par_mT", 1e3 * B_par)
    
        f_nv = D_NV_MHZ + branch_sign_true * GAMMA_NV_MHZ_PER_T * B_par
        Delta = pm.Deterministic("Delta", f_mw_true - f_nv)
    
        # ODMR line shape
        Gamma = pm.TruncatedNormal(
            "Gamma",
            mu=8.0,
            sigma=2.0,
            lower=1.0,
            upper=20.0,
        )
    
        C = pm.Beta(
            "C",
            alpha=22,
            beta=80,
        )
    
        L = 1.0 / (1.0 + (Delta / Gamma)**2)
    
        # Simple normal prior on background fluorescence
        beta0 = pm.Normal(
            "beta0",
            mu=float(np.mean(counts_kcps)),
            sigma=10.0,
        )
    
        mu_model = pm.Deterministic(
            "mu_model",
            beta0 * (1.0 - C * L),
        )
    
        # Noise
        sigma = pm.HalfNormal("sigma", sigma=8.0)
    
        y_like = pm.StudentT(
            "y_like",
            nu=4,
            mu=mu_model,
            sigma=sigma,
            observed=y_shared,
        )
    return full_prism_inference_model

def dipole_posterior(data, extents):
    Ny, Nx = data.shape
    x_min, x_max, y_min, y_max = extents[-4:]
    x_volt = np.linspace(x_min, x_max, Nx)
    y_volt = np.linspace(y_min, y_max, Ny)
    X_volt, Y_volt = np.meshgrid(x_volt, y_volt)
    
    volt_to_um = 0.067
    x_um = x_volt * volt_to_um
    y_um = y_volt * volt_to_um
    X_um = X_volt * volt_to_um
    Y_um = Y_volt * volt_to_um
    Z_um = np.zeros_like(X_um, dtype=float)
    
    with pm.Model() as nv_dipole_model:

        X_data = pm.Data("X_data", X.astype("float64"))
        Y_data = pm.Data("Y_data", Y.astype("float64"))
    
        Z_data = pm.Data("Z_data", Z.astype("float64"))
    
        y_out = pm.Data("y_sim", y_sim.astype("float64"))
    
        # Giving the model something easy to work with initially...
        z_d_um = pm.Normal("z_d_um", mu=0.078, sigma=0.05)
        x_d_um = pm.Uniform("x_d_um", lower=0.55, upper=1.05)
        y_d_um = pm.Uniform("y_d_um", lower=0.55, upper=1.05)
    
        # and converting to um for physical meaning
        z_d = -z_d_um * 1e-6
        x_d = x_d_um * 1e-6
        y_d = y_d_um * 1e-6
    
        # Take average count from data set as guess
        beta0 = pm.Normal("beta0", mu=float(np.mean(y_sim)), sigma=10.0)
    
        # From 0-1, average of 0.3, weighted towards lower values
        C = pm.Beta("C", alpha=2.0, beta=5.0)
    
        # Relatively well known parameter from prior ESR scans
        Gamma = pm.Exponential("Gamma", lam=1/40.0)
    
        # implement the model using the known dipole parameters
        mu_model, B_par, Delta = mean_fluorescence_dipole_pt(
            X_data, Y_data, Z_data,
            x_d, y_d, z_d,
            m_vec_true[0], m_vec_true[1], m_vec_true[2],
            beta0, C, Gamma, f_mw_true,
            n_vec_true,
            branch_sign=branch_sign_true
        )
    
        pm.Normal("y_like", mu=mu_model, sigma=sigma_true, observed=y_out)

    return nv_dipole_model

def Posterior(data, extents, magnet_geometry="rectangular-prism"):
    if magnet_geometry=="rectangular-prism":
        model_Posterior = rectangular_prism_posterior(data, extents)
        
    elif magnet_geometry == "dipole":
        model_Posterior = dipole_posterior(data, extents)
        
    else:
        print("Magnet geometry not recognized. Defaulting to rectangular prism geometry.")
        model_Posterior = rectangular_prism_posterior(data, extents)
        
    return model_Posterior