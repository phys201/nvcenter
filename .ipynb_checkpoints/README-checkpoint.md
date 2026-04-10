# NVCENTER PROJECT


We will model the observed data as an image of flouresence from a single NV center measured at a fixed microwave frequency $f_{MW}$. Each data point is recorded at a pixel in a 2D array, such that our data is:
$$
D = \{y_{ij}\}
$$
where $y_{ij}$ is the measured flourescene rate in kCounts/s at pixel $(i, j)$. The goal, from this data, is to infer the parameters of a magnetic dipole from this image. 


For the purpose of this preliminary model, we will only consider the following parameters: 
- single magnetic dipole
- known NV axis $\mathbf{\hat{n}}$
- Lorentzian fluorescence dip as a function of detuning
- constant background floresence
- independent Gaussian pixel noise


**Parameters:** For the full parameter vector, consider:

$$
\theta = \left(x_d, y_d, z_d, m_x, m_y, m_z, \beta_0, C, \Gamma, \delta_0, \sigma\right).
$$

Where:

- $(x_d, y_d, z_d)$: dipole position
- $(m_x, m_y, m_z)$: dipole moment components
- $\beta_0, \beta_x, \beta_y$: background offset and linear gradients
- $C$: fluorescence dip contrast
- $\Gamma$: resonance linewidth parameter
- $\delta_0$: global detuning offset
- $\sigma$: standard deviation of the pixel noise


What we will have as input from the image data is the pixel coordinate, i.e. the relative position of the NV center with respect to the NV center. For the defined coordinates $(i, j)$, the full scan coordiante is then
$$
\mathbf{r_{ij}} = (x_{ij}, y_{ij}, z_{NV})
$$
Where $z_{NV}$ is the height of the imaging plane. 

**Dipole B-Field**: We model the magnet as a simple point dipole located at 
$$
\mathbf{r_d} = (x_d, y_d, z_d)
$$
with dipole moment

$$
\mathbf{m} = (m_x, m_y, m_z)
$$
This allows us to define
$$
\mathbf R_{ij} = \mathbf r_{ij} - \mathbf r_d,
\qquad
R_{ij} = \|\mathbf R_{ij}\|,
\qquad
\hat{\mathbf R}_{ij} = \frac{\mathbf R_{ij}}{R_{ij}}.
$$
As the parameter for distance and direction of the displacement of the NV from the dipole. Then, we know that for any given pixel $(i, j)$, the dipole field at that pixel is: 
$$
\mathbf B_{ij}(\theta) = \frac{\mu_0}{4\pi R_{ij}^3} \left[3(\mathbf m \cdot \hat{\mathbf R}_{ij}) \hat{\mathbf R}_{ij} - \mathbf m \right].
$$
Finally, if we assume that the NV orientation is known, then the relevant field component which will impact the spin transition of the NV center is: 
$$
B_{\parallel,ij}(\theta) = \mathbf B_{ij}(\theta) \cdot \hat{\mathbf n}.
$$



**Resonance model:** We are sending in a known, fixed microwave freqeuncy to the system. What we then care about is the detuning between this microwave field and where the NV resonance is. This can be modeled as: 
$$
\Delta_{ij}(\theta) = \delta_0 - \gamma B_{\parallel,ij}(\theta),
$$

where $\gamma$ is the known NV gyromagnetic factor and $\delta_0$ is some global term that encompasses uncertainty in the resonance position. 

The dip itself then can be modeled using a Lorentzian line shape. There will be some global background florecense, as well as a contrast parameter $C$ which is intrinsic to the NV center, and will be known. Typically, $C$ is in the range of 25%. Let us then define our signal as: 
$$
\mu_{ij}(\theta) = \beta_0\left[1 - C \frac{1}{1 + \left( \frac{\Delta_{ij}(\theta)}{\Gamma} \right)^2} \right]
$$
Which means the flourecence will be lowest when $\Delta_{ij}$ is appraoching zero, which will produce the dark line that we see on the contour!

**Likelyhood:** Now for the likelyhood, we model each observed pixel as independent gaussian random variables:
$$
y_{ij} \mid \theta \sim \mathcal N\!\left(\mu_{ij}(\theta), \sigma^2\right).
$$
such that the likelyhood for the entire image is
$$
p(D \mid \theta) = \prod_{i,j} \mathcal{N} \!\left(y_{ij} \mid \mu_{ij}(\theta), \sigma^2\right).
$$
Which, for right now, we aknowledge is an aproximation for the true counting statistics. But because each pixel is an averaged flouresence, the noise should be approximately gaussian. 

**Priors:** Now, in order to complete our model, we need to assign priors to each of the unknown parameters. 

For the dipole position, we say that the dipole may appear anywhere within the region of the image with equal probability: 
$$
x_d \sim \mathrm{Uniform}(x_{\min}, x_{\max}),
\qquad
y_d \sim \mathrm{Uniform}(y_{\min}, y_{\max}),
\qquad
z_d \sim \mathrm{Uniform}(z_{\min}, z_{\max}).
$$

For the dipole itself, it is typically magnetized experimentally by applying an external field approximately along the z-axis. Thus, we should use a prior that favors a dipole moment that points roughly along the z-axis, but allows for imperfect alignment in x and y: 
$$
m_x \sim \mathcal{N}(0, s_\perp^2), 
\qquad
m_y \sim \mathcal{N}(0, s_\perp^2), 
\qquad
m_z \sim \mathcal{N}(m_{z,0}, s_z^2)
$$

For the background, the baseline intensity has a rough expected scale, which can be determined emperically from the images or prior scans with the NV center. For this reason, we center the prior at this value with a normal distribution:
$$
\beta_0 \sim \mathcal{N}(\bar{y}, s_{\beta_0}^2)
$$

For the remaining parameters dip contrast, linewidth, and noise standard deviation, we know we need to enforce positivity when we have a generally known scale for the parameter. For this case, we use exponential parameters: 
$$
\Gamma \sim \text{Exponential}(\lambda_\Gamma), 
\qquad
\sigma \sim \text{Exponential}(\lambda_\sigma), 
$$

As well as the contrast $C$ for the system, which we know is a value bounded by $[0, 1]$. With this information we use a Beta prior:
$$
C \sim \text{Beta}(\alpha_C, \beta_C)
$$

Finally, we have some detuning offset which has a natural reference point of zero, but we shoudl allow for deviations on either side of zero, which is a natural choice for a normal prior:
$$
\delta_0 \sim \mathcal{N}(0, s_\delta^2)
$$


Now we may finally think about the joint distribution, which may be written as: 
$$
p(\theta, D) = p(\theta)\prod_{i,j} p(y_{ij}\mid \theta)
$$

with

$$
p(\theta) =
p(x_d)p(y_d)p(z_d)p(m_x)p(m_y)p(m_z)
p(\beta_0)p(C)p(\Gamma)p(\delta_0)p(\sigma),
$$

and 

$$
p(y_{ij}\mid \theta) = \mathcal N\!\left(y_{ij}\mid \mu_{ij}(\theta), \sigma^2\right)
$$

So that the posterior distribution is:

$$
p(\theta \mid D)
\propto
p(D\mid \theta)\,p(\theta).
$$