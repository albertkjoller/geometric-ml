import numpy as np
import torch

def get_grid_coords(ranges, resolutions, return_torch=True):
    u = np.linspace(ranges[0][0], ranges[0][1], resolutions[0])
    v = np.linspace(ranges[1][0], ranges[1][1], resolutions[1])
    u, v = np.meshgrid(u, v)
    coords = np.stack([u.flatten(), v.flatten()], axis=-1)
    return torch.tensor(coords) if return_torch else coords

def brownian_motion_ambient(x0, T, stepsize=1e-3, return_trajectories=True):
    stepsize = torch.tensor(stepsize)
    num_steps = int(T / stepsize)

    x_t = x0.clone()
    N, D = x0.shape

    noise_samples = torch.randn(num_steps, N, D)
    trajectories = torch.zeros((num_steps + 1, N, D))
    trajectories[0, :] = x_t
    for t in range(num_steps):
        x_t += torch.sqrt(stepsize) * noise_samples[t, :]
        trajectories[t + 1, :] = x_t
    return trajectories if return_trajectories else x_t

def matrix_sqrt(M):
    # Ensure float dtype
    M = M
    # Eigendecomposition (symmetric)
    eigenvals, eigenvecs = torch.linalg.eigh(M)  # eigenvals may be negative
    # Reconstruct the matrix square root
    return eigenvecs @ torch.diag(torch.sqrt(eigenvals)) @ eigenvecs.T

def rollout(f, init, xs):
    """A simple implementation of something similar to what torch.scan would be."""
    outputs = []
    carry = init
    for x in xs:
        carry = f(carry, x)
        outputs.append(carry)
    return carry, torch.stack(outputs)

def euler_integration(x0, velocity_field, t_span, dt):
    """
    x0: torch.Tensor [N,D] initial points
    velocity_field: function x -> dx/dt (returns [N,D])
    t_span: (t0, t_final)
    dt: time step
    """
    phi = x0.clone()
    t0, t_final = t_span
    n_steps = int((t_final - t0) / dt)
    
    for _ in range(n_steps):
        v = velocity_field(phi)
        phi = phi + dt * v  # Euler step

    return phi