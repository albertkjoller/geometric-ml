import torch
import numpy as np
from geometric_noise.manifolds.manifold import Manifold
from geometric_noise.methods.utils import get_grid_coords, brownian_motion_ambient
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class SwissRoll(Manifold):
    def __init__(self, return_torch=True):
        self.return_torch = return_torch

    def __call__(self, position):
        """Generate points on a Swiss roll given cylindrical coordinates."""
        u1, u2 = position[:, 0], position[:, 1]
        if type(u1) is np.ndarray:
            u1 = torch.tensor(u1)
            u2 = torch.tensor(u2)
        
        x = u1 * torch.cos(u1)
        y = u2
        z = u1 * torch.sin(u1)
        return torch.vstack([x, y, z]).T if self.return_torch else np.vstack([x, y, z]).T

    def __compute_brownian_update__(self, position, noise_sample, stepsize):
        N, D = position.shape
        u1, u2 = position[:, 0], position[:, 1]

        drift_term = torch.stack([torch.pow(u1, 2) / torch.pow(1 + torch.pow(u1, 2), 2), torch.zeros(N)]).T
        noise_term = torch.stack([1 / torch.sqrt(1 + torch.pow(u1, 2)), torch.ones(N)]).T
        return 0.5 * stepsize * drift_term + torch.sqrt(stepsize) * noise_term * noise_sample

    def __geodesic_equation__(self, position, velocity, **kwargs):
        u1, u2 = position.flatten()
        v1, v2 = velocity.flatten()
        acceleration = np.stack([
            - u1 / (1 + np.power(u1, 2)) * np.power(v1, 2),
            0,
        ]).reshape(-1,1)
        return acceleration

    def generate_data(self, N_train=50, train_ranges=[(1.5 * torch.pi, 4.5 * torch.pi), (0, 10)], test_ranges=[(0, 6*torch.pi), (-5, 15)], test_res=[200, 50]):
        # Sample training points uniformly in the intrinsic coordinate ranges from above
        local_coords_train = torch.rand(N_train, 2)
        local_coords_train[:, 0] = train_ranges[0][0] + (train_ranges[0][1] - train_ranges[0][0]) * local_coords_train[:, 0]
        local_coords_train[:, 1] = train_ranges[1][0] + (train_ranges[1][1] - train_ranges[1][0]) * local_coords_train[:, 1]
        # Map to manifold and define target attribute
        X_train = self(local_coords_train)
        y_train = local_coords_train[:, 0] # linear along the "roll"

        # Get evaluation coordinates
        local_coords_test = get_grid_coords(ranges=test_ranges, resolutions=test_res)
        X_test = self(local_coords_test)
        y_test = local_coords_test[:, 0]
        return X_train, local_coords_train, y_train, X_test, local_coords_test, y_test


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    # Set random seed for reproducibility    
    np.random.seed(0); torch.manual_seed(0)

    # Define the manifold
    manifold = SwissRoll(return_torch=True)

    # Set noise intensity
    noise_intensity = 5.0

    with torch.no_grad():
        # Generate data from the manifold
        X_train, U_train, y_train, X_test, U_test, y_test = manifold.generate_data(N_train=10)

        # Get geometric quantities at the starting points
        geometric_quantities = manifold.get_quantities(U_train)
        # Compute covariance matrix of the transformed tangent space noise
        transformed_covariances = noise_intensity * geometric_quantities['J_inv'] @ geometric_quantities['P'] @ geometric_quantities['J_inv'].transpose(-1,-2)
        # Compute Cholesky decomposition for sampling
        Ls = torch.linalg.cholesky(transformed_covariances)         

        # Sample initial velocities in the parameter space
        init_vs = torch.einsum('ijk,ik->ij', Ls, torch.randn_like(U_train))
        init_states = torch.hstack([U_train, init_vs])

    ### PLOTTING ###

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(223, projection='3d')
    ax2.view_init(elev=34, azim=151, roll=36)
    ax2.set_facecolor('white')
    ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c='lightgray', s=1, label='Deformed sphere', alpha=0.1)
    for i, init_state in enumerate(init_states):
        # Compute geodesic
        geodesics, intrinsic_curves = manifold.geodesic(init_state[:2], init_v=init_state[2:], geodesic_res=101)
        # Detach from autograd graph for plotting
        geodesics, intrinsic_curves = geodesics.detach(), intrinsic_curves.detach()
        # Plot the geodesics
        ax1.plot(intrinsic_curves[:, 0], intrinsic_curves[:, 1], alpha=0.7, label=i)
        ax2.plot(geodesics[:, 0], geodesics[:, 1], geodesics[:, 2], alpha=0.7)
    ax1.set_aspect('equal')
    ax1.set_title('Geodesics')
    ax2.set_aspect('equal')

    ax3 = fig.add_subplot(222)
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.view_init(elev=34, azim=151, roll=36)
    ax4.set_facecolor('white')
    ax4.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c='lightgrey', s=1, label='Deformed sphere')
    for i, (position, local_coord) in enumerate(zip(X_train, U_train)):
        # Simulate Brownian motion
        brownian_trajectory, intrinsic_trajectory = manifold.brownian_motion(local_coord[None, :], diffusion_time=noise_intensity, num_steps=200, return_trajectories=True)
        # Detach from autograd graph for plotting
        brownian_trajectory, intrinsic_trajectory = brownian_trajectory.squeeze(1).detach(), intrinsic_trajectory.squeeze(1).detach()
        
        # Plot the Brownian motion trajectory
        ax3.plot(intrinsic_trajectory[:, 0], intrinsic_trajectory[:, 1], alpha=0.7, label=i)
        ax3.plot(intrinsic_trajectory[0, 0], intrinsic_trajectory[0, 1], 'o', alpha=0.7, mec='k', color=f"C{i}")
        ax3.plot(intrinsic_trajectory[-1, 0], intrinsic_trajectory[-1, 1], 'X', alpha=0.7, mec='k', color=f"C{i}")
        ax4.plot(brownian_trajectory[:, 0], brownian_trajectory[:, 1], brownian_trajectory[:, 2], alpha=0.7)
        ax4.plot(position[0], position[1], position[2], 'o', alpha=0.7, mec='k', color=f"C{i}")
        ax4.plot(brownian_trajectory[-1, 0], brownian_trajectory[-1, 1], brownian_trajectory[-1, 2], 'X', alpha=0.7, mec='k', color=f"C{i}")

    ax3.set_aspect('equal')
    ax3.set_title('Brownian motion')
    ax4.set_aspect('equal')

    plt.show()