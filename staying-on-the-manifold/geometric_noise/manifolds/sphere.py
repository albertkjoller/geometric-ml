import torch
import numpy as np
from geometric_noise.manifolds.manifold import Manifold
from geometric_noise.methods.utils import get_grid_coords, euler_integration
from geometric_noise.methods.velocity_field import VelocityField

torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15, floatmode='maxprec_equal')
np_dtype = np.float64

class Sphere(Manifold):
    def __init__(self, return_torch=True, a=1.0, c=1.0):
        self.return_torch = return_torch
        self.a = torch.tensor(a)
        self.c = torch.tensor(c)

    def __call__(self, position):
        """Generate points on a sphere given spherical coordinates."""
        u1, u2 = position[:, 0], position[:, 1]
        if type(u1) is np.ndarray:
            u1 = torch.tensor(u1)
            u2 = torch.tensor(u2)
            
        x = self.a * torch.sin(u1) * torch.cos(u2)
        y = self.a * torch.sin(u1) * torch.sin(u2)
        z = self.c * torch.cos(u1)
        return torch.vstack([x, y, z]).T if self.return_torch else np.vstack([x, y, z]).T
    
    def __compute_brownian_update__(self, position, noise_sample, stepsize):
        # The sphere has an analytical expression for the Brownian motion update in spherical coordinates
        N, D = position.shape
        u1, u2 = position[:, 0], position[:, 1]
        a2 = torch.pow(self.a, 2)
        c2 = torch.pow(self.c, 2)
        
        drift_term = torch.stack([
            a2 * torch.cos(u1) / (torch.sin(u1) * (a2 * torch.pow(torch.cos(u1), 2) + c2 * torch.pow(torch.sin(u1), 2))),
            torch.zeros(N),
        ]).T

        noise_term = torch.stack([
            1 / torch.sqrt(a2 * torch.pow(torch.cos(u1), 2) + c2 * torch.pow(torch.sin(u1), 2)),
            1 / (self.a * torch.sin(u1)),
        ]).T * noise_sample
        return 0.5 * stepsize * drift_term + torch.sqrt(stepsize) * noise_term

    def __geodesic_equation__(self, position, velocity, **kwargs):
        u1, u2 = position.flatten()
        v1, v2 = velocity.flatten()
        a2, c2 = torch.pow(self.a, 2), torch.pow(self.c, 2)

        acceleration = np.stack([
            (a2 * np.sin(u1) * np.cos(u1) * np.power(v2, 2) - (c2 - a2) * np.sin(u1) * np.cos(u1) * np.power(v1, 2)) / (a2 * np.power(np.cos(u1), 2) + c2 * np.power(np.sin(u1), 2)),
            - 2 * np.cos(u1) / np.sin(u1) * v1 * v2,
        ]).reshape(-1,1)
        return acceleration

    def generate_data(self, N_train=40, train_ranges=None, test_ranges=None, test_res=None):
        if train_ranges is None:
            train_ranges = [(0, torch.pi), (0, 2*torch.pi)]
        if test_ranges is None:
            test_ranges = [(0, torch.pi), (0, 2*torch.pi)]
        if test_res is None:
            test_res = [75, 175]

        # Sample training points uniformly in the intrinsic coordinate ranges from above
        local_coords_train = torch.rand(N_train, 2)
        local_coords_train[:, 0] = train_ranges[0][0] + (train_ranges[0][1] - train_ranges[0][0]) * local_coords_train[:, 0]
        local_coords_train[:, 1] = train_ranges[1][0] + (train_ranges[1][1] - train_ranges[1][0]) * local_coords_train[:, 1]
        # Map to manifold and define target attribute
        X_train = self(local_coords_train)
        # Construct a chosen target function
        y_train = local_coords_train[:, 0] 

        # Do the same for the test grid (the full sphere)
        local_coords_test = get_grid_coords(ranges=test_ranges, resolutions=test_res)
        X_test = self(local_coords_test)
        y_test = local_coords_test[:, 0]
        return X_train, local_coords_train, y_train, X_test, local_coords_test, y_test


class SqueezedSphere(Sphere):
    def __init__(self, return_torch=True):
        super().__init__(return_torch=return_torch, a=1.0, c=1/7)

    def generate_data(self, N_train=40, train_ranges=None, test_ranges=None, test_res=None):
        if train_ranges is None:
            train_ranges = [(0, torch.pi), (0, 2*np.pi)]
        if test_ranges is None:
            test_ranges = [(0, torch.pi), (0, 2*np.pi)]
        if test_res is None:
            test_res = [75, 175]

        # Sample training points uniformly in the intrinsic coordinate ranges from above
        local_coords_train = torch.rand(N_train, 2)
        local_coords_train[:, 0] = train_ranges[0][0] + (train_ranges[0][1] - train_ranges[0][0]) * local_coords_train[:, 0]
        local_coords_train[:, 1] = train_ranges[1][0] + (train_ranges[1][1] - train_ranges[1][0]) * local_coords_train[:, 1]
        # Map to manifold and define target attribute
        X_train = self(local_coords_train)
        # Construct a chosen target function
        y_train = local_coords_train[:, 0]

        # Do the same for the test grid (the full sphere)
        local_coords_test = get_grid_coords(ranges=test_ranges, resolutions=test_res)
        X_test = self(local_coords_test)
        y_test = local_coords_test[:, 0]
        return X_train, local_coords_train, y_train, X_test, local_coords_test, y_test


class DeformedSphere(Manifold):
    # def __init__(self, return_torch=True, alpha=10, hidden_dim=128, seed=42):
    def __init__(self, return_torch=True, alpha=18, hidden_dim=64, seed=3, base_manifold=Sphere, dt=1e-1):
        super().__init__(return_torch=return_torch)

        # Define the base manifold (here, the sphere)
        self.base_manifold = base_manifold(return_torch=return_torch)

        # Fix the "random" velocity field
        torch.manual_seed(seed)
        self.velocity_field = VelocityField(alpha=alpha, hidden_dim=hidden_dim)
        # Define the integration timestep
        self.dt = dt

    def __flow__(self, base_position, t=torch.tensor([0.0, 1.0])):
        # OBS: torchdiffeq version --- doesn't work with torch.func
        # deformed_points = odeint(lambda t, x: self.velocity_field(x), base_points, t)[-1]
        return euler_integration(base_position, self.velocity_field, t, dt=self.dt)

    def __call__(self, position, t=torch.tensor([0.0, 1.0])):
        base_position = self.base_manifold(position)
        return self.__flow__(base_position, t=t)

    def generate_data(self, N_train=40, t=torch.tensor([0.0, 1.0]), train_ranges=None, test_ranges=None, test_res=None):
        if train_ranges is None:
            train_ranges = [(0, torch.pi), (0, 2*np.pi)]
        if test_ranges is None:
            test_ranges = [(0, torch.pi), (0, 2*np.pi)]
        if test_res is None:
            test_res = [50, 75]
        # Sample training points uniformly in the intrinsic coordinate ranges from above
        local_coords_train = torch.rand(N_train, 2)
        local_coords_train[:, 0] = train_ranges[0][0] + (train_ranges[0][1] - train_ranges[0][0]) * local_coords_train[:, 0]
        local_coords_train[:, 1] = train_ranges[1][0] + (train_ranges[1][1] - train_ranges[1][0]) * local_coords_train[:, 1]
        # Map to manifold and define target attribute
        X_train = self(local_coords_train, t=t)
        # Construct a chosen target function
        y_train = local_coords_train[:, 0] 

        # Do the same for the test grid (the full deformed sphere)
        local_coords_test = get_grid_coords(ranges=test_ranges, resolutions=test_res)
        X_test = self(local_coords_test, t=t)
        y_test = local_coords_test[:, 0]
        return X_train, local_coords_train, y_train, X_test, local_coords_test, y_test

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    # Define the manifold
    manifold = DeformedSphere(return_torch=True)#, alpha=10, hidden_dim=128, seed=42)
    # manifold = Sphere(return_torch=True)

    # Set random seed for reproducibility    
    np.random.seed(0); torch.manual_seed(0)

    # Set noise intensity
    noise_intensity = 0.01

    with torch.no_grad():
        # Generate data from the manifold
        X_train, U_train, y_train, X_test, U_test, y_test = manifold.generate_data(N_train=10, test_res=[400, 200])

        # Get geometric quantities at the starting points
        geometric_quantities = manifold.get_quantities(U_train)
        # Compute covariance matrix of the transformed tangent space noise
        transformed_covariances = noise_intensity * geometric_quantities['J_inv'] @ geometric_quantities['P'] @ geometric_quantities['J_inv'].transpose(-1,-2)
        # Compute Cholesky decomposition for sampling
        Ls = torch.linalg.cholesky(transformed_covariances)         

        # Sample initial velocities in the parameter space
        init_vs = torch.einsum('ijk,ik->ij', Ls, torch.randn_like(U_train))
        init_states = torch.hstack([U_train, init_vs])

    ### PLOTTING DEFORMED SPHERE AT DIFFERENT TIMES ###
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        with torch.no_grad():
            # Generate data from the manifold
            X_train, U_train, y_train, X_test, U_test, y_test = manifold.generate_data(N_train=10, test_res=[400, 400], t=torch.tensor([0.0, t]))

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=-48, azim=-112, roll=-77)
        ax.view_init(elev=4, azim=116, roll=27)
        ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, s=10, cmap='RdBu_r')
        ax.set_aspect('equal')
        ax.set_title(f'$t={t:.2f}$', fontsize=40)
        ax.axis('off')
        ax.set_facecolor('white')
        fig.savefig(f'deformed_sphere_t{t:.2f}.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
    
    ### PLOTTING ###

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(223, projection='3d')
    ax2.view_init(elev=34, azim=151, roll=36)
    ax2.set_facecolor('white')
    ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, s=1, label='Deformed sphere')
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