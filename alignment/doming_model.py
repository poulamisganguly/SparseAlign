import autograd
import autograd.numpy as np
import scipy.optimize
import time
from tqdm.auto import tqdm, trange
import alignment.motion_model as mm

# define geometry

d=64 # discretisation of detector
eval_grid = np.linspace(-0.5, 0.5, d)

nt = 20 # no. of time frames
time_points = np.arange(0,nt)/(nt-1)

# tomography geometry
N_α = nt # number of proj angles
αs = np.linspace(-np.pi/3, np.pi/3, N_α, False)

N_p = 6 # number of velocity parameters

# forward model for one marker

def f(θx,θz,p1,p2):
    δx, δz = mm.deformation(p1,p2,θx,θz)
    r = (θz+δz*time_points)*np.sin(αs) + (θx+δx*time_points)*np.cos(αs)
    return np.stack(ρ for ρ in r)

# definition of linear operator Φ

def F(θxs, θzs, p1, p2):
    return np.array([f(θx,θz,p1,p2) for (θx,θz) in zip(θxs, θzs)]).T


# definition of convex loss

def loss(θxs, θzs, p1, p2, mark_pos_proj):
    if F(θxs, θzs, p1, p2).shape!=mark_pos_proj.shape:
        raise ValueError("Array shapes incorrect")
    return ((F(θxs, θzs, p1, p2)-mark_pos_proj)**2).sum()


def doming_model_opt(y, θxs, θzs, p1, p2, iters=35, only_est_motion=False):
    losses = []
    history = []
    # get initial guess for marker locations
    def initial_guess():
        n = len(θxs)
        fun = autograd.value_and_grad(lambda θs : loss(θs[:n], θs[n:], p1, p2, y))
        bnds = ([(-0.5, 0.5)]*n)+([(-0.5, 0.5)]*n)
        res = scipy.optimize.minimize(fun, np.concatenate((θxs, θzs)), jac=True, method='L-BFGS-B', 
                                      bounds=bnds, options={'disp': True})
        θxs1 = res['x'][:n]
        θzs1 = res['x'][n:]
        return θxs1, θzs1
    
    def opt_motion_params():
        fun_p = autograd.value_and_grad(lambda ps : loss(θxs, θzs, np.zeros(6), ps, y))
        num_ps = len(p2)
        bnds = ([(0.0, 2.0)]*num_ps)
        res = scipy.optimize.minimize(fun_p, p2, jac=True, method='L-BFGS-B', 
                                      bounds=bnds, options={'disp': True})
        pi1 = np.zeros(6)
        pi2 = res['x']
        return pi1, pi2
    
    for iter in range(iters):
        if not only_est_motion:
            θxs, θzs = initial_guess()
        
        p1, p2 = opt_motion_params()
        
        curr_loss = loss(θxs, θzs, p1, p2, y)
        losses.append(curr_loss)
        history.append((losses, θxs, θzs, p1, p2))
        print(iter, curr_loss)
        
    
    return history