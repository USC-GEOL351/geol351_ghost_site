# LorenzODELCE.py:
#   Estimate the spectrum of Lyapunov Characteristic Exponents
#	  for the Lorenz ODEs, using the pull-back method.
#   Also, estimate the volume-contraction (dissipation) rate and the
#	   fractal dimenion (latter using the Kaplan-Yorke conjecture).
#   Plot out trajectory, for reference.
#
# Comment:
#	Notice how much more complicated the code has become, given
#		that we're writing out variables in component form.
#   This should be rewritten to use vectors, which will be
#	   much more compact and easier to debug. Equally important,
#      the code would generalize to any dimension system.

# Import plotting routines
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import colormaps as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider


# import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight')

# The Lorenz 3D ODEs
#	Original parameter values: (sigma,R,b) = (10,28,-8/3)
def LorenzXDot(sigma, R, b, x, y, z):
    return sigma * (-x + y)


def LorenzYDot(sigma, R, b, x, y, z):
    return R * x - x * z - y


def LorenzZDot(sigma, R, b, x, y, z):
    return b * z + x * y


# The tangent space (linearized) flow (aka co-tangent flow)
def LorenzDXDot(sigma, R, b, x, y, z, dx, dy, dz):
    return sigma * (-dx + dy)


def LorenzDYDot(sigma, R, b, x, y, z, dx, dy, dz):
    return (R - z) * dx - dy - x * dz


def LorenzDZDot(sigma, R, b, x, y, z, dx, dy, dz):
    return y * dx + x * dy + b * dz


# Volume contraction given by
#	 Trace(Jacobian(x,y,z)) = b - sigma - 1
def LorenzODETrJac(sigma, R, b, x, y, z):
    return b - sigma - 1


# As a check, we must have total contraction = Sum of LCEs
#	 Tr(J) = Sum_i LCEi
# Numerical check: at (sigma,R,b) = (10,28,-8/3)
#	 LCE0  ~   0.9058
#	 LCE1  ~   0.0000
#	 LCE2  ~ -14.572
#	 Tr(J) ~ -13.6666
# These use base-2 logs

# The fractal dimension from the LCEs (Kaplan-Yorke conjecture)
#   Assume these are ordered: LCE1 >= LCE2 >= LCE3
def FractalDimension3DODE(LCE1, LCE2, LCE3):
    # "Close" to zero ... we're estimating here
    Err = 0.01
    if LCE1 < -Err:  # Stable fixed point    (-,-,-)
        return 0.0
    elif abs(LCE1) <= Err:
        if LCE2 < -Err:  # Limit cycle	      (0,-,-)
            return 1.0
        else:  # Torus               (0,0,-)
            return 2.0
    else:  # Chaotic attractor   (+,0,-)
        return 2.0 + (LCE1 + LCE2) / abs(LCE3)


# 3D fourth-order Runge-Kutta integrator
def RKThreeD(a, b, c, x, y, z, f, g, h, dt):
    k1x = dt * f(a, b, c, x, y, z)
    k1y = dt * g(a, b, c, x, y, z)
    k1z = dt * h(a, b, c, x, y, z)
    k2x = dt * f(a, b, c, x + k1x / 2.0, y + k1y / 2.0, z + k1z / 2.0)
    k2y = dt * g(a, b, c, x + k1x / 2.0, y + k1y / 2.0, z + k1z / 2.0)
    k2z = dt * h(a, b, c, x + k1x / 2.0, y + k1y / 2.0, z + k1z / 2.0)
    k3x = dt * f(a, b, c, x + k2x / 2.0, y + k2y / 2.0, z + k2z / 2.0)
    k3y = dt * g(a, b, c, x + k2x / 2.0, y + k2y / 2.0, z + k2z / 2.0)
    k3z = dt * h(a, b, c, x + k2x / 2.0, y + k2y / 2.0, z + k2z / 2.0)
    k4x = dt * f(a, b, c, x + k3x, y + k3y, z + k3z)
    k4y = dt * g(a, b, c, x + k3x, y + k3y, z + k3z)
    k4z = dt * h(a, b, c, x + k3x, y + k3y, z + k3z)
    x += (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
    y += (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
    z += (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
    return x, y, z


# Tanget space flow (using fourth-order Runge-Kutta integrator)

# THIS IS HORRIBLE! Should
def TangentFlowRKThreeD(a, b, c, x, y, z, df, dg, dh, dx, dy, dz, dt):
    k1x = dt * df(a, b, c, x, y, z, dx, dy, dz)
    k1y = dt * dg(a, b, c, x, y, z, dx, dy, dz)
    k1z = dt * dh(a, b, c, x, y, z, dx, dy, dz)
    k2x = dt * df(a, b, c, x, y, z, dx + k1x / 2.0, dy + k1y / 2.0, dz + k1z / 2.0)
    k2y = dt * dg(a, b, c, x, y, z, dx + k1x / 2.0, dy + k1y / 2.0, dz + k1z / 2.0)
    k2z = dt * dh(a, b, c, x, y, z, dx + k1x / 2.0, dy + k1y / 2.0, dz + k1z / 2.0)
    k3x = dt * df(a, b, c, x, y, z, dx + k2x / 2.0, dy + k2y / 2.0, dz + k2z / 2.0)
    k3y = dt * dg(a, b, c, x, y, z, dx + k2x / 2.0, dy + k2y / 2.0, dz + k2z / 2.0)
    k3z = dt * dh(a, b, c, x, y, z, dx + k2x / 2.0, dy + k2y / 2.0, dz + k2z / 2.0)
    k4x = dt * df(a, b, c, x, y, z, dx + k3x, dy + k3y, dz + k3z)
    k4y = dt * dg(a, b, c, x, y, z, dx + k3x, dy + k3y, dz + k3z)
    k4z = dt * dh(a, b, c, x, y, z, dx + k3x, dy + k3y, dz + k3z)
    dx += (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
    dy += (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
    dz += (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
    return dx, dy, dz


def LorenzLCE(dt=0.01, pars=(10, 28, 8. / 3), nTransients=10, nIterates=1000):
    ''' Solves Lorenz system and plots xy projection, with the following  diagnostics:
      - Lyapunov Characteristic Exponents for x, y an z dimensions
      - Volume contraction rate
      - Attractor dimension (using the Kaplan-Yorke conjecture)

    Inputs:
      - dt:  Integration time step
      - pars: Vector of parameters (sigma, rho, beta)
      - nTransients: The number of iterations to throw away
      - nIterates: The number of time steps to integrate over


    '''

    # Simulation parameters
    # Integration time step
	# dt = 0.01
    #
    # Control parameters for the Lorenz ODEs:
    sigma = pars[0]
    R = pars[1]
    b = -pars[2]
    # The number of iterations to throw away
	# nTransients = 10
    # The number of time steps to integrate over
	# nIterates = 1000

    # The main loop that generates the orbit, storing the states
    xState = 5.0
    yState = 5.0
    zState = 5.0
    # Iterate for some number of transients, but don't use these states
    for n in range(0, nTransients):
        xState, yState, zState = RKThreeD(sigma, R, b, xState, yState, zState, LorenzXDot, LorenzYDot, LorenzZDot, dt)
    # Set up array of iterates and store the current state
    x = [xState]
    y = [yState]
    z = [zState]
    for n in range(0, nIterates):
        # at each time step calculate new (x,y,z)(t)
        xt, yt, zt = RKThreeD(sigma, R, b, x[n], y[n], z[n], LorenzXDot, LorenzYDot, LorenzZDot, dt)
        # and append to lists
        x.append(xt)
        y.append(yt)
        z.append(zt)

    # Estimate the LCEs
    # The number of iterations to throw away
    nTransients = 100
    # The number of iterations to over which to estimate
    #  This is really the number of pull-backs
    nIterates = 1000
    # The number of iterations per pull-back
    nItsPerPB = 10
    # Initial condition
    xState = 5.0
    yState = 5.0
    zState = 5.0
    # Initial tangent vectors
    e1x = 1.0
    e1y = 0.0
    e1z = 0.0
    e2x = 0.0
    e2y = 1.0
    e2z = 0.0
    e3x = 0.0
    e3y = 0.0
    e3z = 1.0
    # Iterate away transients and let the tangent vectors align
    #	with the global stable and unstable manifolds
    for n in range(0, nTransients):
        for i in range(nItsPerPB):
            xState, yState, zState = RKThreeD(sigma, R, b, xState, yState, zState, \
                                              LorenzXDot, LorenzYDot, LorenzZDot, dt)
            # Evolve tangent vector for maximum LCE (LCE1)
            e1x, e1y, e1z = TangentFlowRKThreeD(sigma, R, b, xState, yState, zState, \
                                                LorenzDXDot, LorenzDYDot, LorenzDZDot, e1x, e1y, e1z, dt)
            # Evolve tangent vector for next LCE (LCE2)
            e2x, e2y, e2z = TangentFlowRKThreeD(sigma, R, b, xState, yState, zState, \
                                                LorenzDXDot, LorenzDYDot, LorenzDZDot, e2x, e2y, e2z, dt)
            # Evolve tangent vector for last LCE
            e3x, e3y, e3z = TangentFlowRKThreeD(sigma, R, b, xState, yState, zState, \
                                                LorenzDXDot, LorenzDYDot, LorenzDZDot, e3x, e3y, e3z, dt)
        # Normalize the tangent vector
        d = np.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
        e1x /= d
        e1y /= d
        e1z /= d
        # Pull-back: Remove any e1 component from e2
        dote1e2 = e1x * e2x + e1y * e2y + e1z * e2z
        e2x -= dote1e2 * e1x
        e2y -= dote1e2 * e1y
        e2z -= dote1e2 * e1z
        # Normalize second tangent vector
        d = np.sqrt(e2x * e2x + e2y * e2y + e2z * e2z)
        e2x /= d
        e2y /= d
        e2z /= d
        # Pull-back: Remove any e1 and e2 components from e3
        dote1e3 = e1x * e3x + e1y * e3y + e1z * e3z
        dote2e3 = e2x * e3x + e2y * e3y + e2z * e3z
        e3x -= dote1e3 * e1x + dote2e3 * e2x
        e3y -= dote1e3 * e1y + dote2e3 * e2y
        e3z -= dote1e3 * e1z + dote2e3 * e2z
        # Normalize third tangent vector
        d = np.sqrt(e3x * e3x + e3y * e3y + e3z * e3z)
        e3x /= d
        e3y /= d
        e3z /= d

    # Okay, now we're ready to begin the estimation
    LCE1 = 0.0
    LCE2 = 0.0
    LCE3 = 0.0
    for n in range(0, nIterates):
        for i in range(nItsPerPB):
            xState, yState, zState = RKThreeD(sigma, R, b, xState, yState, zState, \
                                              LorenzXDot, LorenzYDot, LorenzZDot, dt)
            # Evolve tangent vector for maximum LCE (LCE1)
            e1x, e1y, e1z = TangentFlowRKThreeD(sigma, R, b, xState, yState, zState, \
                                                LorenzDXDot, LorenzDYDot, LorenzDZDot, e1x, e1y, e1z, dt)
            # Evolve tangent vector for next LCE (LCE2)
            e2x, e2y, e2z = TangentFlowRKThreeD(sigma, R, b, xState, yState, zState, \
                                                LorenzDXDot, LorenzDYDot, LorenzDZDot, e2x, e2y, e2z, dt)
            # Evolve tangent vector for last LCE
            e3x, e3y, e3z = TangentFlowRKThreeD(sigma, R, b, xState, yState, zState, \
                                                LorenzDXDot, LorenzDYDot, LorenzDZDot, e3x, e3y, e3z, dt)
        # Normalize the tangent vector
        d = np.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
        e1x /= d
        e1y /= d
        e1z /= d
        # Accumulate the first tangent vector's length change factor
        LCE1 += np.log(d)
        # Pull-back: Remove any e1 component from e2
        dote1e2 = e1x * e2x + e1y * e2y + e1z * e2z
        e2x -= dote1e2 * e1x
        e2y -= dote1e2 * e1y
        e2z -= dote1e2 * e1z
        # Normalize second tangent vector
        d = np.sqrt(e2x * e2x + e2y * e2y + e2z * e2z)
        e2x /= d
        e2y /= d
        e2z /= d
        # Accumulate the second tangent vector's length change factor
        LCE2 += np.log(d)
        # Pull-back: Remove any e1 and e2 components from e3
        dote1e3 = e1x * e3x + e1y * e3y + e1z * e3z
        dote2e3 = e2x * e3x + e2y * e3y + e2z * e3z
        e3x -= dote1e3 * e1x + dote2e3 * e2x
        e3y -= dote1e3 * e1y + dote2e3 * e2y
        e3z -= dote1e3 * e1z + dote2e3 * e2z
        # Normalize third tangent vector
        d = np.sqrt(e3x * e3x + e3y * e3y + e3z * e3z)
        e3x /= d
        e3y /= d
        e3z /= d
        # Accumulate the third tangent vector's length change factor
        LCE3 += np.log(d)

    # Convert to per-iterate, per-second LCEs and to base-2 logs
    IntegrationTime = dt * float(nItsPerPB) * float(nIterates)
    LCE1 = LCE1 / IntegrationTime
    LCE2 = LCE2 / IntegrationTime
    LCE3 = LCE3 / IntegrationTime
    LCE = np.array([LCE1, LCE2, LCE3])

    # Calculate contraction factor, for comparison.
    #	For Lorenz ODE, we know this is independent of (x,y,z).
    #	Otherwise, we'd have to estimate it along the trajectory, too.
    Contraction = LorenzODETrJac(sigma, R, b, 0.0, 0.0, 0.0)

    Dim = FractalDimension3DODE(LCE1, LCE2, LCE3)

    return LCE, Contraction, Dim

# LCE, Contraction, Dim = LorenzLCE()

# fig, ax = plt.subplots()
# # Choose a pair of coordinates from (x,y,z) to show
# # Setup the parametric plot:
# plt.xlabel(r'$x(t)$') # set x-axis label
# plt.ylabel(r'$y(t)$') # set y-axis label
# # Construct plot title
# LCEString = '(%g,%g,%g)' % (LCE1,LCE2,LCE3)
# PString = '($\sigma$,R,b) = (%g,%g,%g)' % (sigma,R,b)
# CString = 'Contraction = %g ' % Contraction
# FString   = 'Fractal dimension = %g' % FractalDimension3DODE(LCE1,LCE2,LCE3)
# plt.title('Lorenz system w/ ' + PString + ':\n LCEs = ' + LCEString + ', ' + CString + '\n ' + FString)
# #axis('equal')
# #plt.axis([-20.0,20.0,-20.0,20.0])
# # Plot the trajectory in the phase plane
# plt.plot(x,y,'b')
# plt.axhline(0.0,color = 'k')
# plt.axvline(0.0,color = 'k')

# Use command below to save figure
# savefig('LorenzODELCE', dpi=600)

# Display the plot in a window
# show()

def lorenz_ornaments(ax, ind, cols, LCE, Contraction, Dim, fig, force=None):
    # Adding a figure parameter to allow for figure-based annotations

    # Write Lyapunov Characteristic Exponents
    LCEString = r' $(\lambda_x,\lambda_y,\lambda_z)$'+'\n=({:1.4f},{:1.4f},{:3.2f})'.format(*LCE)
    # Write Contraction rate
    CString = r'$\nabla \cdot \vec f$ (contraction)'+'\n = {:3.2f}'.format(Contraction)
    # Write Dimension
    FString = 'd = {:1.4f}'.format(Dim)
    # title_copy = ax.get_title()
    # ax.set_title(title_copy+LCEString + '\n' + CString + '\n' + FString)

    if force is None:
      # # Determine subplot position (left, right, or below) based on cols and ind
      if cols == 2:
          if ind % cols == 0:  # Column 0, annotations to the left
              text_pos = 'left'
          else:  # Column 1, annotations to the right
              text_pos = 'right'
      else:
          text_pos = 'below'
    else:
      text_pos=force

    # Use figure text for annotations to place them outside the axes
    if text_pos == 'left':
        fig.text(0, 0.5, LCEString + '\n' + CString + '\n' + FString,
                 transform=ax.transAxes, ha='right', va='center')
    elif text_pos == 'right':
        fig.text(1, 0.5, LCEString + '\n' + CString + '\n' + FString,
                 transform=ax.transAxes, ha='left', va='center')
    elif text_pos == 'below':
        # Adjust the vertical position based on the subplot row
        row = ind // cols
        vertical_position = .1 - (row * 0.05)  # Example calculation, adjust as needed
        fig.text(.5, vertical_position, LCEString + '\n' + CString + '\n' + FString,
                 transform=ax.transAxes, ha='center', va='top')
                 

# def lorenz_ornaments(ax, LCE, Contraction, Dim, force=None):
#     # obtain axes limits
#     xlim = ax.get_xlim3d()
#     ylim = ax.get_ylim3d()
#     zlim = ax.get_zlim3d()

#     # Write Lyapunov Characteristic Exponents
#     LCEString = r' $(\lambda_x,\lambda_y,\lambda_z)=({:1.4f},{:1.4f},{:3.2f})$'.format(*LCE)
#     CString = r'$\nabla \cdot \vec f$ = {:3.2f}'.format(Contraction)
#     FString = 'd = {:1.4f}'.format(Dim)

#     if force is None:
#         ax.text3D(0.75 * xlim[0], 0.25 * ylim[0], 1.1 * zlim[1], LCEString)
#         # Write Contraction rate
#         # CString = 'Contraction = {:3.2f}'.format(Contraction)
#         ax.text3D(0.9 * xlim[0], 0.75 * ylim[0], 0.9 * zlim[0], CString)
#         # Write Dimension
#         ax.text3D(0.25 * xlim[1], 0. * ylim[1], 1.0 * zlim[0], FString)
#     elif force=='below':
#         xlabel = ax.get_xlabel()
#         xlabel = '\n'.join([xlabel, LCEString, CString, FString])
#         ax.set_xlabel(xlabel)

#     return ax


def lorenz_spaghetti(lorenz, pars=(10, 28, 8. / 3), T=4, N=20, u0=[10, 10, 10],
                     jitter=(-0.5, 0.5, 3), smpl_fac=50, ax=None, fs=12, include_lce=True):
    t = np.linspace(0, T, int(smpl_fac * T))
    clr = [plt.cm.get_cmap('inferno')(a) for a in np.linspace(0, 1, N)]
    rho = pars[1]  # extract Rayleigh number

    LCE = None
    if include_lce:
        LCE, _, _ = LorenzLCE(pars=pars)  # Contraction, Dim

    # solve and plot "true" trajectories
    truth = solve_ivp(lorenz, [0, T], u0, t_eval=t, args=pars)  # solve problem
    ax.plot(t, truth.y[0], linewidth=2, color='k', label='truth')  # plot solution

    # solve and plot trajectories
    for n in range(N):
        u1 = u0 + np.random.uniform(*jitter)  # define IC
        soln = solve_ivp(lorenz, [0, T], u1, t_eval=t, args=pars)  # solve problem
        ax.plot(t, soln.y[0], linewidth=1, alpha=0.3, color=clr[n])  # plot solution

    if include_lce and LCE is not None:
        ttl = r'Lorenz predictions, $\rho={:3.2f}, \lambda_1={:3.4f}$'.format(rho, LCE[0])
    else:
        ttl = r'Lorenz predictions, $\rho={:3.2f}$'.format(rho)
    ax.set_title(ttl, fontweight='bold', fontsize=fs)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x(t)$')

    return ax, soln


def lorenz_plot(lorenz, pars=(10, 28, 8. / 3), T=100, N=6, u0=[0, 1, 0],
                cmap='magma', jitter=(-0.5, 0.5, 3), smpl_fac=100, ax=None, fs=12):
    t = np.linspace(0, T, int(smpl_fac * T))
    clr = [cm.get_cmap(cmap)(a) for a in np.linspace(0, 1, N)]
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.get_cmap(cmap)(np.linspace(1,0,N)))
    rho = pars[1]  # extract Rayleigh number
    print('ax', ax)

    # solve and plot trajectories
    for n in range(N):
        u1 = u0 + np.random.uniform(*jitter)  # define IC
        ax.scatter(*u1, alpha=0.4, color=clr[n])  # plot IC
        soln = solve_ivp(lorenz, [0, T], u1, t_eval=t, args=pars)  # solve problem
        ax.plot(soln.y[0], soln.y[1], soln.y[2], linewidth=0.5, alpha=0.6, color=clr[n])  # plot solution

    # add fixed point
    ax.scatter(0, 0, 0, alpha=0.2, color='C0', s=40)

    if rho > 1:
        b = np.sqrt(pars[2] * (pars[1] - 1))
        Cp = (b, b, pars[1] - 1);
        Cm = (-b, -b, pars[1] - 1)
        ax.scatter(*Cp, alpha=0.6, color='C2', s=20)
        ax.scatter(*Cm, alpha=0.6, color='C2', s=20)

    # ttl = r'Lorenz system with $(\sigma,\rho,\beta)=({:3.2f},{:3.2f},{:3.2f})$'.format(*pars)
    ttl = r'Lorenz system with $\rho={:3.2f}$'.format(rho)
    ax.set_title(ttl, fontweight='bold', fontsize=fs)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    # ax.set_frame_on(False)
    return ax, soln

def compute_lce_grid(rho_grid, sigma=10.0, beta=8.0 / 3.0, dt=0.01, nTransients=100, nIterates=1000):
    """Compute Lorenz Lyapunov diagnostics on a rho grid."""
    rho_grid = np.array(rho_grid, dtype=float)
    LCE = np.empty((len(rho_grid), 3), dtype=float)
    contraction = np.empty(len(rho_grid), dtype=float)
    dim = np.empty(len(rho_grid), dtype=float)

    for i, rho in enumerate(rho_grid):
        vals = LorenzLCE(
            dt=dt,
            pars=(sigma, float(rho), beta),
            nTransients=nTransients,
            nIterates=nIterates,
        )
        LCE[i, :], contraction[i], dim[i] = vals

    return LCE, contraction, dim


def compute_forecast_horizon_grid(
    system,
    rho_values,
    sigma=10.0,
    beta=8.0 / 3.0,
    u0=(10.0, 10.0, 10.0),
    n_ens=10,
    T=40.0,
    dt=0.02,
    perturb_scale=1e-6,
    tolerance=5.0,
):
    """Estimate forecast horizon and uncertainty from ensemble perturbation growth."""
    rho_values = np.array(rho_values, dtype=float)
    u0 = np.array(u0, dtype=float)
    n_steps = max(1, int(np.floor(float(T) / float(dt))))
    t_eval = np.linspace(0.0, float(T), n_steps + 1)

    horizons = np.empty(len(rho_values), dtype=float)
    horizon_median = np.empty(len(rho_values), dtype=float)
    horizon_q25 = np.empty(len(rho_values), dtype=float)
    horizon_q75 = np.empty(len(rho_values), dtype=float)
    crossed_frac = np.empty(len(rho_values), dtype=float)
    spreads = []
    member_horizons = []

    for i, rho in enumerate(rho_values):
        pars = (sigma, float(rho), beta)
        truth = solve_ivp(system, [0.0, T], u0, t_eval=t_eval, args=pars)

        ens_dist = np.zeros((int(n_ens), len(t_eval)), dtype=float)
        h_members = np.full(int(n_ens), np.nan, dtype=float)
        for j in range(int(n_ens)):
            perturb = np.random.normal(0.0, perturb_scale, size=3)
            ens_sol = solve_ivp(system, [0.0, T], u0 + perturb, t_eval=t_eval, args=pars)
            ens_dist[j, :] = np.linalg.norm(ens_sol.y - truth.y, axis=0)
            c_j = np.where(ens_dist[j, :] >= tolerance)[0]
            if len(c_j) > 0:
                h_members[j] = t_eval[c_j[0]]

        spread = ens_dist.mean(axis=0)
        spreads.append(spread)
        member_horizons.append(h_members)

        crossed = np.where(spread >= tolerance)[0]
        horizons[i] = t_eval[crossed[0]] if len(crossed) > 0 else np.nan
        finite_h = h_members[np.isfinite(h_members)]
        crossed_frac[i] = float(len(finite_h)) / float(max(1, len(h_members)))
        if len(finite_h) > 0:
            horizon_median[i] = float(np.median(finite_h))
            horizon_q25[i] = float(np.percentile(finite_h, 25))
            horizon_q75[i] = float(np.percentile(finite_h, 75))
        else:
            horizon_median[i] = np.nan
            horizon_q25[i] = np.nan
            horizon_q75[i] = np.nan

    return {
        "rho": rho_values,
        "horizon": horizons,
        "horizon_member": np.array(member_horizons),
        "horizon_median": horizon_median,
        "horizon_q25": horizon_q25,
        "horizon_q75": horizon_q75,
        "crossed_fraction": crossed_frac,
        "t": t_eval,
        "spread": np.array(spreads),
    }


def lorenz(t, u, *pars):
    """Lorenz-63 RHS with optional (sigma, rho, beta) override."""
    sigma, rho, beta = (10.0, 28.0, 8.0 / 3.0) if len(pars) == 0 else pars
    x, y, z = u
    return [sigma * (y - x), rho * x - y - x * z, x * y - beta * z]


def _rk4_step(system, state, t, dt, pars):
    """Single RK4 step for a 3D state and RHS signature f(t, u, *pars)."""
    k1 = np.array(system(t, state, *pars), dtype=float)
    k2 = np.array(system(t + 0.5 * dt, state + 0.5 * dt * k1, *pars), dtype=float)
    k3 = np.array(system(t + 0.5 * dt, state + 0.5 * dt * k2, *pars), dtype=float)
    k4 = np.array(system(t + dt, state + dt * k3, *pars), dtype=float)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _poincare_crossings(traj_xyz, z_level=25.0, max_points=None):
    """Return x-y points where z crosses z_level upward using linear interpolation."""
    z = traj_xyz[:, 2]
    crossings = []
    for i in range(len(z) - 1):
        z0, z1 = z[i], z[i + 1]
        if z0 < z_level <= z1:
            frac = (z_level - z0) / (z1 - z0 + 1e-16)
            xy = traj_xyz[i, :2] + frac * (traj_xyz[i + 1, :2] - traj_xyz[i, :2])
            crossings.append(xy)
            if max_points is not None and len(crossings) >= int(max_points):
                break
    if len(crossings) == 0:
        return np.empty((0, 2))
    return np.array(crossings)


def local_stretch_demo(
    system=lorenz,
    pars=None,
    dt=0.01,
    spinup=20.0,
    sample_time=5.0,
    eps=1e-6,
    n_cloud=100,
    n_plot_cloud=None,
    seed=0,
    make_poincare=False,
    u_init=(5.0, 5.0, 5.0),
    display_time=None,
    show_time_slider=True,
    focus_projection_on_cloud=True,
    xz_zoom=0.35,
    show_zoom_slider=True,
    trace_alpha=0.25,
):
    """
    Visualize local stretching near a Lorenz attractor point.

    Students should notice:
    - early-time near-linear growth of log separation (approx largest Lyapunov slope),
    - later-time saturation because Lorenz is bounded,
    - cloud deformation from near-sphere to elongated set (local stretching).
    """
    if pars is None:
        # Keep backwards-compatible Lorenz defaults, but allow any 3D system when pars is supplied.
        if getattr(system, "__name__", "") == "lorenz":
            pars = (10.0, 28.0, 8.0 / 3.0)
        else:
            raise ValueError("For non-Lorenz systems, pass `pars=(...)` explicitly.")
    pars = tuple(pars)
    rng = np.random.default_rng(seed)

    n_spin = max(2, int(spinup / dt) + 1)
    n_samp = max(2, int(sample_time / dt) + 1)
    t_spin = np.linspace(0.0, spinup, n_spin)
    t = np.linspace(0.0, sample_time, n_samp)

    # Spin up a reference trajectory to land on the attractor neighborhood.
    u_init = np.array(u_init, dtype=float)
    if u_init.shape != (3,):
        raise ValueError("u_init must be length-3 for 3D systems.")
    spin_sol = solve_ivp(system, [0.0, spinup], u_init, t_eval=t_spin, args=pars)
    x0 = spin_sol.y[:, -1].copy()

    # Build an isotropic cloud with ||delta|| ~= eps.
    raw = rng.normal(0.0, 1.0, size=(int(n_cloud), 3))
    raw_norm = np.linalg.norm(raw, axis=1, keepdims=True)
    raw_norm = np.maximum(raw_norm, 1e-16)
    delta = eps * (raw / raw_norm)
    cloud0 = x0[None, :] + delta

    if n_plot_cloud is None:
        plot_idx = np.arange(int(n_cloud))
    else:
        n_plot_cloud = max(1, min(int(n_plot_cloud), int(n_cloud)))
        plot_idx = rng.choice(int(n_cloud), size=n_plot_cloud, replace=False)

    # Integrate reference and cloud forward with a shared fixed-step RK4.
    ref = np.empty((n_samp, 3), dtype=float)
    ref[0] = x0
    cloud = np.empty((int(n_cloud), n_samp, 3), dtype=float)
    cloud[:, 0, :] = cloud0

    for k in range(1, n_samp):
        tk = t[k - 1]
        ref[k] = _rk4_step(system, ref[k - 1], tk, dt, pars)
        for i in range(int(n_cloud)):
            cloud[i, k, :] = _rk4_step(system, cloud[i, k - 1, :], tk, dt, pars)

    # Distances to reference trajectory through time.
    dist = np.linalg.norm(cloud - ref[None, :, :], axis=2)
    dist = np.maximum(dist, 1e-16)
    logd = np.log(dist)
    med_logd = np.median(logd, axis=0)
    q25 = np.percentile(logd, 25, axis=0)
    q75 = np.percentile(logd, 75, axis=0)

    fit_end = max(4, int(0.3 * n_samp))
    fit_slice = slice(1, fit_end)  # skip t=0 to avoid numerical artifacts
    slope, intercept = np.polyfit(t[fit_slice], med_logd[fit_slice], deg=1)
    fit_line = slope * t + intercept

    # Compact 3-panel figure:
    # 1) 3D trajectory + cloud traces
    # 2) x-z projection (cloud-focused by default)
    # 3) log-separation growth with early-time slope fit
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    spec0 = axs[0].get_subplotspec()
    axs[0].remove()
    ax3d = fig.add_subplot(spec0, projection='3d')
    ax_xz = axs[1]
    ax_log = axs[2]

    ax3d.plot(ref[:, 0], ref[:, 1], ref[:, 2], lw=1.0, color="0.35", label="reference")
    ax3d.scatter(
        cloud0[plot_idx, 0], cloud0[plot_idx, 1], cloud0[plot_idx, 2],
        s=10, color="C0", alpha=0.8, label="initial cloud"
    )
    cloud_traces_3d = []
    for j in plot_idx:
        tr3, = ax3d.plot([], [], [], lw=0.6, color="C3", alpha=trace_alpha)
        cloud_traces_3d.append((j, tr3))
    cloud_now_3d, = ax3d.plot([], [], [], ls="", marker="o", ms=4, color="C3", alpha=0.9, label="cloud at t")

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("Local cloud deformation (3D)")
    ax3d.legend(loc="upper left", fontsize=8, frameon=False)

    ax_xz.scatter(cloud0[plot_idx, 0], cloud0[plot_idx, 2], s=14, color="C0", alpha=0.75, label="initial cloud")
    ax_xz.plot(ref[:, 0], ref[:, 2], lw=0.8, color="0.75", alpha=0.75)
    cloud_traces_xz = []
    for j in plot_idx:
        trxz, = ax_xz.plot([], [], lw=0.8, color="C3", alpha=trace_alpha)
        cloud_traces_xz.append((j, trxz))
    cloud_now_xz, = ax_xz.plot([], [], ls="", marker="o", ms=4, color="C3", alpha=0.9, label="cloud at t")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.set_title("x-z projection")
    ax_xz.legend(loc="best", fontsize=8, frameon=False)
    ax_xz.grid(True, alpha=0.25)

    ax_log.plot(t, med_logd, color="C2", lw=2, label="median log separation")
    ax_log.fill_between(t, q25, q75, color="C2", alpha=0.2, label="IQR")
    ax_log.plot(t, fit_line, color="k", ls="--", lw=1, label=f"early slope ~ {slope:.3f}")
    t_marker = ax_log.axvline(t[0], color="C3", ls=":", lw=1.2, label="display time")
    ax_log.set_xlabel("t")
    ax_log.set_ylabel("log separation")
    ax_log.set_title("log separation growth")
    ax_log.text(
        0.03,
        0.06,
        "Late-time flattening is expected:\nLorenz attractor is bounded.",
        transform=ax_log.transAxes,
        fontsize=8,
        va="bottom",
    )
    ax_log.legend(loc="best", fontsize=8, frameon=False)
    ax_log.grid(True, alpha=0.25)

    if show_time_slider:
        fig.subplots_adjust(bottom=0.18, wspace=0.35)
    else:
        fig.tight_layout()

    if display_time is None:
        display_idx = n_samp - 1
    else:
        display_idx = int(np.argmin(np.abs(t - float(display_time))))

    xz_focus_ref = None
    if focus_projection_on_cloud:
        all_x = np.r_[cloud0[plot_idx, 0], cloud[plot_idx][:, :, 0].ravel()]
        all_z = np.r_[cloud0[plot_idx, 2], cloud[plot_idx][:, :, 2].ravel()]
        cx = float(np.median(all_x))
        cz = float(np.median(all_z))
        sx = max(float(np.ptp(all_x)), 50.0 * eps)
        sz = max(float(np.ptp(all_z)), 50.0 * eps)
        xz_focus_ref = (cx, cz, sx, sz)

    zoom_state = {"val": float(max(0.05, xz_zoom))}

    def _apply_xz_zoom(zoom_val):
        if xz_focus_ref is None:
            return
        cx, cz, sx, sz = xz_focus_ref
        zv = float(max(0.05, zoom_val))
        pad_x = max(20.0 * eps, 0.5 * sx * zv)
        pad_z = max(20.0 * eps, 0.5 * sz * zv)
        ax_xz.set_xlim(cx - pad_x, cx + pad_x)
        ax_xz.set_ylim(cz - pad_z, cz + pad_z)

    def _update_display(idx):
        idx = int(max(0, min(n_samp - 1, idx)))
        x_now = cloud[plot_idx, idx, 0]
        y_now = cloud[plot_idx, idx, 1]
        z_now = cloud[plot_idx, idx, 2]

        for j, tr3 in cloud_traces_3d:
            tr3.set_data(cloud[j, :idx + 1, 0], cloud[j, :idx + 1, 1])
            tr3.set_3d_properties(cloud[j, :idx + 1, 2])
        cloud_now_3d.set_data(x_now, y_now)
        cloud_now_3d.set_3d_properties(z_now)
        ax3d.set_title(f"Local cloud deformation (3D), t={t[idx]:.2f}")

        for j, trxz in cloud_traces_xz:
            trxz.set_data(cloud[j, :idx + 1, 0], cloud[j, :idx + 1, 2])
        cloud_now_xz.set_data(x_now, z_now)
        _apply_xz_zoom(zoom_state["val"])
        ax_xz.set_title(f"x-z projection at t={t[idx]:.2f}")

        t_marker.set_xdata([t[idx], t[idx]])
        fig.canvas.draw_idle()

    _update_display(display_idx)

    t_slider = None
    z_slider = None
    if show_time_slider and n_samp > 2:
        slider_ax = fig.add_axes([0.15, 0.06, 0.70, 0.03])
        t_slider = Slider(slider_ax, "Display t", float(t[0]), float(t[-1]), valinit=float(t[display_idx]), valstep=float(dt))

        def _on_slide(val):
            i = int(np.argmin(np.abs(t - float(val))))
            _update_display(i)

        t_slider.on_changed(_on_slide)

    if focus_projection_on_cloud and show_zoom_slider:
        if show_time_slider and n_samp > 2:
            zoom_ax = fig.add_axes([0.15, 0.015, 0.70, 0.03])
            fig.subplots_adjust(bottom=0.24, wspace=0.35)
        else:
            zoom_ax = fig.add_axes([0.15, 0.06, 0.70, 0.03])
            fig.subplots_adjust(bottom=0.22, wspace=0.35)
        z_slider = Slider(zoom_ax, "x-z zoom", 0.05, 1.5, valinit=zoom_state["val"], valstep=0.01)

        def _on_zoom(val):
            zoom_state["val"] = float(val)
            _apply_xz_zoom(zoom_state["val"])
            fig.canvas.draw_idle()

        z_slider.on_changed(_on_zoom)

    result = {
        "t": t,
        "x0": x0,
        "delta": delta,
        "x_ref": ref,
        "cloud_initial": cloud0,
        "cloud_t": cloud,
        "cloud_final": cloud[:, -1, :],
        "plot_indices": plot_idx,
        "distance": dist,
        "median_log_distance": med_logd,
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "fig": fig,
        "axes": (ax3d, ax_xz, ax_log),
        "time_slider": t_slider,
        "zoom_slider": z_slider,
    }

    if make_poincare:
        z_level = 25.0
        kmax = 15
        ref_cross = _poincare_crossings(ref, z_level=z_level, max_points=kmax)
        cloud_cross = []
        for i in range(int(n_cloud)):
            c = _poincare_crossings(cloud[i], z_level=z_level, max_points=kmax)
            if len(c) > 0:
                cloud_cross.append(c)
        cloud_cross = np.vstack(cloud_cross) if len(cloud_cross) > 0 else np.empty((0, 2))
        result["poincare"] = {
            "z_level": z_level,
            "reference_xy": ref_cross,
            "cloud_xy": cloud_cross,
        }

    return result


def estimate_lce_spectrum(
    system,
    pars,
    u_init=(5.0, 5.0, 5.0),
    dt=0.01,
    spinup=20.0,
    n_steps=2000,
    fd_eps=1e-7,
):
    """
    Estimate full Lyapunov spectrum (lambda1, lambda2, lambda3) for a 3D ODE.

    Uses a finite-difference approximation of the one-step flow-map Jacobian
    and Benettin QR re-orthonormalization.
    """
    pars = tuple(pars)
    u = np.array(u_init, dtype=float)
    if u.shape != (3,):
        raise ValueError("u_init must be a length-3 iterable.")

    # Burn-in to settle on the attractor/regime.
    n_spin = max(1, int(spinup / dt))
    t = 0.0
    for _ in range(n_spin):
        u = _rk4_step(system, u, t, dt, pars)
        t += dt

    Q = np.eye(3, dtype=float)
    lsum = np.zeros(3, dtype=float)

    for _ in range(int(n_steps)):
        # Base one-step map
        u_next = _rk4_step(system, u, t, dt, pars)

        # Finite-difference Jacobian of one-step flow map Phi_dt(u)
        dphi = np.zeros((3, 3), dtype=float)
        for j in range(3):
            u_pert = u + fd_eps * Q[:, j]
            u_pert_next = _rk4_step(system, u_pert, t, dt, pars)
            dphi[:, j] = (u_pert_next - u_next) / fd_eps

        # Evolve basis and re-orthonormalize
        Y = dphi @ Q
        Q, R = np.linalg.qr(Y)
        diagR = np.maximum(np.abs(np.diag(R)), 1e-16)
        lsum += np.log(diagR)

        u = u_next
        t += dt

    lce = lsum / (float(n_steps) * dt)
    # Report in descending order (lambda1 >= lambda2 >= lambda3)
    return np.sort(lce)[::-1]


def compute_lce_spectrum_grid(
    system,
    parameter_list,
    u_init=(5.0, 5.0, 5.0),
    dt=0.01,
    spinup=20.0,
    n_steps=2000,
    fd_eps=1e-7,
):
    """Compute full 3-exponent Lyapunov spectrum for each parameter tuple."""
    parameter_list = list(parameter_list)
    out = np.empty((len(parameter_list), 3), dtype=float)
    for i, pars in enumerate(parameter_list):
        out[i, :] = estimate_lce_spectrum(
            system=system,
            pars=pars,
            u_init=u_init,
            dt=dt,
            spinup=spinup,
            n_steps=n_steps,
            fd_eps=fd_eps,
        )
    return out


def compute_lce_spectrum_sweep(
    system,
    base_pars,
    sweep_values,
    sweep_index=1,
    u_init=(5.0, 5.0, 5.0),
    dt=0.01,
    spinup=20.0,
    n_steps=2000,
    fd_eps=1e-7,
):
    """
    Convenience wrapper: sweep one parameter index and return spectra.

    Returns:
      sweep_values (np.ndarray), spectra (N,3)
    """
    sweep_values = np.array(sweep_values, dtype=float)
    base_pars = list(base_pars)
    parameter_list = []
    for v in sweep_values:
        p = base_pars.copy()
        p[int(sweep_index)] = float(v)
        parameter_list.append(tuple(p))

    spectra = compute_lce_spectrum_grid(
        system=system,
        parameter_list=parameter_list,
        u_init=u_init,
        dt=dt,
        spinup=spinup,
        n_steps=n_steps,
        fd_eps=fd_eps,
    )
    return sweep_values, spectra


def plot_lce_spectrum_sweep(
    system,
    base_pars,
    sweep_values,
    sweep_index=1,
    u_init=(5.0, 5.0, 5.0),
    dt=0.01,
    spinup=20.0,
    n_steps=2000,
    fd_eps=1e-7,
    semilogx=False,
    xlabel=None,
    ylabel="Lyapunov exponent",
    title="Lyapunov spectrum sweep",
    ax=None,
):
    """
    One-call helper: sweep one parameter, estimate (lambda1,lambda2,lambda3), and plot.

    Returns:
      fig, ax, sweep_values, spectra
    """
    sweep_values, spectra = compute_lce_spectrum_sweep(
        system=system,
        base_pars=base_pars,
        sweep_values=sweep_values,
        sweep_index=sweep_index,
        u_init=u_init,
        dt=dt,
        spinup=spinup,
        n_steps=n_steps,
        fd_eps=fd_eps,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    plot_fn = ax.semilogx if semilogx else ax.plot
    plot_fn(sweep_values, spectra[:, 0], label=r"$\lambda_1$")
    plot_fn(sweep_values, spectra[:, 1], label=r"$\lambda_2$")
    plot_fn(sweep_values, spectra[:, 2], label=r"$\lambda_3$")

    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel(xlabel if xlabel is not None else f"parameter index {int(sweep_index)}")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    return fig, ax, sweep_values, spectra
