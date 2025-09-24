import torch
import numpy as np
from geometric_noise.manifolds.manifold import Manifold
from geometric_noise.methods.utils import get_grid_coords, brownian_motion_ambient
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class GeneralTorus(Manifold):
    def __init__(self, return_torch=True, a=1.0, c=1.0):
        self.return_torch = return_torch
        self.a = torch.tensor(a)
        self.c = torch.tensor(c)

    def __call__(self, position):
        """Generate points on a torus given intrinsic coordinates."""
        u1, u2 = position[:, 0], position[:, 1]
        if self.return_torch and not torch.is_tensor(u1):
            u1 = torch.tensor(u1)
            u2 = torch.tensor(u2)

        x = (self.a + self.c * torch.sin(u1)) * torch.sin(u2)
        y = (self.a + self.c * torch.sin(u1)) * torch.cos(u2)
        z = self.c * torch.cos(u1)
        return torch.vstack([x, y, z]).T if self.return_torch else np.vstack([x, y, z]).T
    
    def __compute_brownian_update__(self, position, noise_sample, stepsize):
        N, D = position.shape
        u1, u2 = position[:, 0], position[:, 1]

        drift_term = torch.stack([torch.cos(u1) / (self.c * (self.a + self.c * torch.sin(u1))), torch.zeros(N)]).T
        noise_term = torch.stack([torch.ones(N) / self.c, 1/(self.a + self.c * torch.sin(u1))]).T * noise_sample
        return 0.5 * stepsize * drift_term + torch.sqrt(stepsize) * noise_term

    def __geodesic_equation__(self, position, velocity, **kwargs):
        u1, u2 = position.flatten()
        v1, v2 = velocity.flatten()

        acceleration = np.stack([
            (self.a + self.c * np.sin(u1)) * np.cos(u1) / self.c * np.power(v2, 2),
            - 2 * self.c * np.cos(u1) / (self.a + self.c * np.sin(u1)) * v1 * v2,
        ]).reshape(-1,1)
        return acceleration

    def generate_data(self, N_train=40, train_ranges=[(0, 2*np.pi), (0, 2*np.pi)], test_ranges=[(0, 2*np.pi), (0, 2*np.pi)], test_res=[75, 175]):
        # Sample training points uniformly in the intrinsic coordinate ranges from above
        local_coords_train = torch.rand(N_train, 2)
        local_coords_train[:, 0] = train_ranges[0][0] + (train_ranges[0][1] - train_ranges[0][0]) * local_coords_train[:, 0]
        local_coords_train[:, 1] = train_ranges[1][0] + (train_ranges[1][1] - train_ranges[1][0]) * local_coords_train[:, 1]
        # Map to manifold and define target attribute
        X_train = self(local_coords_train)
        y_train = torch.sin(local_coords_train[:, 1]) * 100  # Scale the target to make it more interesting

        # Get evaluation coordinates
        local_coords_test = get_grid_coords(ranges=test_ranges, resolutions=test_res)
        X_test = self(local_coords_test)
        y_test = torch.sin(local_coords_test[:, 1]) * 100
        return X_train, local_coords_train, y_train, X_test, local_coords_test, y_test

class OnionRing(GeneralTorus):
    def __init__(self, return_torch=True,):
        super().__init__(return_torch=return_torch, a=1.0, c=0.1)

    def generate_data(self, N_train=40, train_ranges=[(0, 2*np.pi), (0, 2*np.pi)], test_ranges=[(0, 2*np.pi), (0, 2*np.pi)], test_res=[75, 175]):
        # Sample training points uniformly in the intrinsic coordinate ranges from above
        local_coords_train = torch.rand(N_train, 2)
        local_coords_train[:, 0] = train_ranges[0][0] + (train_ranges[0][1] - train_ranges[0][0]) * local_coords_train[:, 0]
        local_coords_train[:, 1] = train_ranges[1][0] + (train_ranges[1][1] - train_ranges[1][0]) * local_coords_train[:, 1]
        # Map to manifold and define target attribute
        X_train = self(local_coords_train)
        y_train = X_train[:, 2] * 100 # Use height as target

        # Get evaluation coordinates
        local_coords_test = get_grid_coords(ranges=test_ranges, resolutions=test_res)
        X_test = self(local_coords_test)
        y_test = X_test[:, 2] * 100
        return X_train, local_coords_train, y_train, X_test, local_coords_test, y_test

class Bead(GeneralTorus):
    def __init__(self, return_torch=True):
        super().__init__(return_torch=return_torch, a=1.0, c=0.8)

    def generate_data(self, N_train=40, train_ranges=[(0, 2*np.pi), (0, 2*np.pi)], test_ranges=[(0, 2*np.pi), (0, 2*np.pi)], test_res=[75, 175]):
        # Sample training points uniformly in the intrinsic coordinate ranges from above
        local_coords_train = torch.rand(N_train, 2)
        local_coords_train[:, 0] = train_ranges[0][0] + (train_ranges[0][1] - train_ranges[0][0]) * local_coords_train[:, 0]
        local_coords_train[:, 1] = train_ranges[1][0] + (train_ranges[1][1] - train_ranges[1][0]) * local_coords_train[:, 1]
        # Map to manifold and define target attribute
        X_train = self(local_coords_train)
        y_train = torch.sin(local_coords_train[:, 1])  # Scale the target to make it more interesting

        # Get evaluation coordinates
        local_coords_test = get_grid_coords(ranges=test_ranges, resolutions=test_res)
        X_test = self(local_coords_test)
        y_test = torch.sin(local_coords_test[:, 1])
        return X_train, local_coords_train, y_train, X_test, local_coords_test, y_test



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    # Set random seed for reproducibility    
    np.random.seed(0); torch.manual_seed(0)

    # Define the manifold
    manifold = Bead(return_torch=True)
    manifold = OnionRing(return_torch=True)

    # Set noise intensity
    noise_intensity = 0.05

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