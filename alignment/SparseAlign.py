import autograd
import autograd.numpy as np
import scipy.optimize
import time
from tqdm.auto import tqdm, trange
import alignment.motion_model as mm

# define geometry

d=64 # discretisation of detector
eval_grid = np.linspace(-0.5, 0.5, d)
σ = 0.05 # width of gaussian blur

nt = 20 # no. of time frames
time_points = np.arange(0,nt)/(nt-1)

# tomography geometry
N_α = nt # number of proj angles
αs = np.linspace(-np.pi/3, np.pi/3, N_α, False)

N_p = 6 # number of deformation parameters

# forward model for one marker

def f(θx,θz,p1,p2):
    δx, δz = mm.deformation(p1,p2,θx,θz)
    r = (θz+δz*time_points)*np.sin(αs) + (θx+δx*time_points)*np.cos(αs)
    return np.stack(np.exp(-((eval_grid - ρ)/σ)**2) for ρ in r)

# forward model for a set of markers - linear superposition of f's

def F(ws, θxs, θzs, p1, p2):
    return np.sum(np.array([w*f(θx,θz,p1,p2) for (w,θx,θz) in zip(ws, θxs, θzs)]),0)


# definition of convex loss

def loss(ws, θxs, θzs, p1, p2, data):
    return ((F(ws, θxs, θzs, p1, p2)-data)**2).sum()


# marker addition routines

num_grid_pts = 10 #number of grid points in coarse grid

grid_x = np.linspace(-0.5, 0.5,num_grid_pts)
grid_y = np.linspace(-0.5,0.5,num_grid_pts)

grid = np.meshgrid(grid_x, grid_y)
grid_pts = np.vstack([grid[0].ravel(), grid[1].ravel()])

def lmo(res, p1, p2):
    score0 = np.inf
    pbar = tqdm(zip(grid_pts[0], grid_pts[1]), total=len(grid_pts[0]))
    for (θx,θz) in pbar:
        pbar.set_description("Finding new location")
        score = f(θx,θz,p1,p2).flatten() @ res.flatten()
        if score<score0:
            cand_loc = θx,θz
            score0 = score
    return cand_loc


# block coordinate descent : (1) optimisation over weights, (2) weight pruning,
#                            (4) deformation parameter fitting (3) refinement of initial marker locations

def coordinate_descent(y, θxs, θzs, p1, p2, iters = 35, min_drop = 1E-4, all_at_once=True, num_mark_all_at_once=1):
    def min_ws():
        return scipy.optimize.lsq_linear(np.stack([f(θx,θz,p1,p2).flatten() for (θx,θz) in zip(θxs,θzs)]).T, y.flatten(), 
                                         bounds=(0.0,1.0))['x'] # solves a linear least-squares problem with bounds on the weights
    def min_θs():
        n = len(θxs)
        fun = autograd.value_and_grad(lambda θs : loss(ws, θs[:n], θs[n:n+n], p1, p2, y))
        bnds = ([(-0.5, 0.5)]*n)+([(-0.5, 0.5)]*n)
        res = scipy.optimize.minimize(fun, np.concatenate((θxs, θzs)), jac=True, method='L-BFGS-B', 
                                      bounds=bnds)
        
        θxs1 = res['x'][:n]
        θzs1 = res['x'][n:n+n]
        return θxs1, θzs1, res['fun']
    
    def fit_params():
        if all_at_once:
            # check if number of markers has reached the minimum set
            if len(θxs)>=num_mark_all_at_once:
                bnds_p = ([(0.0,2.0)]*N_p)
                fun_p = autograd.value_and_grad(lambda v : loss(ws, θxs, θzs, np.zeros(6), v, y))
                res_p = scipy.optimize.minimize(fun_p, p2, jac=True,method='L-BFGS-B', 
                                                bounds=bnds_p)


                π1 = np.zeros(6) 
                π2 = res_p['x'] 
            else:
                π1 = np.zeros(6) 
                π2 = np.zeros(6)
        else:
            if len(θxs)<=N_p:
                N_p_curr = len(θxs)
            else:
                N_p_curr = N_p
            rem = np.zeros(N_p-N_p_curr)
            bnds_p = ([(0.0,2.0)]*N_p_curr)
            fun_p = autograd.value_and_grad(lambda v : loss(ws, θxs, θzs, np.zeros(6), np.hstack((v,rem)), y))
            res_p = scipy.optimize.minimize(fun_p, p2[:N_p_curr], jac=True,method='L-BFGS-B', 
                                            bounds=bnds_p)


            π1 = np.zeros(6) 
            π2 = np.hstack((res_p['x'],rem)) 
        return π1, π2
    
    old_f_val = np.Inf
    for iter in trange(iters, desc='Local optimisation of locations and deformation params'):
        # optimise weights
        ws = min_ws()
        # prune zero weights
        if np.any(ws<=10**-5):
            print('Removing ', np.sum(ws<=10**-5), ' very small weights!')
            θxs = θxs[ws>=10**-5]
            θzs = θzs[ws>=10**-5]
            ws = ws[ws>=10**-5]
        # fit deformation parameters
        p1, p2 = fit_params()
        # improve support locally
        θxs, θzs, f_val = min_θs()
        θxs = np.array(θxs)
        θzs = np.array(θzs)
        # check if loss is stationary
        if old_f_val - f_val < min_drop:
            break
        old_f_val = f_val
    return ws, θxs, θzs, p1, p2

# definition of SparseAlign
def SparseAlign(y, max_iters, num_markers_known=False, num_markers=None, all_at_once=False, num_mark_all_at_once=1):
    theta_xs = np.zeros(0)
    theta_zs = np.zeros(0)
    pi1 = np.zeros(N_p)
    pi2 = np.zeros(N_p)
    output = np.zeros_like(y)
    losses = []
    history = [] 
    for i in trange(max_iters, desc='ADCG iteration'):
        residual = output - y
        curr_loss = ((residual**2).sum())
        print(i,curr_loss)
        losses.append((curr_loss))
        # find new source
        theta_x, theta_z = lmo(residual, pi1, pi2)
        theta_xs = np.append(theta_xs, theta_x)
        theta_zs = np.append(theta_zs, theta_z)
        # optimise locations and weights
        ws, theta_xs, theta_zs, pi1, pi2 = coordinate_descent(y, theta_xs, theta_zs, pi1, pi2, all_at_once=all_at_once,
                                                             num_mark_all_at_once=num_mark_all_at_once)
        # prune zero weights
        if np.any(ws<=10**-5):
            print('Removing ', np.sum(ws<=10**-5), ' very small weights!')
            theta_xs = theta_xs[ws>=10**-5]
            theta_zs = theta_zs[ws>=10**-5]
            ws = ws[ws>=10**-5]
        history.append((losses, theta_xs, theta_zs, ws, pi1, pi2))
        output = F(ws, theta_xs, theta_zs, pi1, pi2)
    return history