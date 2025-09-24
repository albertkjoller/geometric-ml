import torch
import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from geometric_noise.methods.utils import matrix_sqrt, rollout
from geometric_noise.manifolds.utils import second2first_order, evaluate_solution

class Manifold:
    def __init__(self, return_torch=True):
        self.return_torch = return_torch

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def jacobian(self, points, *args, **kwargs):
        return torch.vmap(torch.func.jacfwd(lambda p: self.__call__(p.view(1,-1)).flatten()))(points)

    def generate_data(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def metric(self, u, jacobian=None):
        if jacobian is None:
            jacobian = self.jacobian(u.view(1,-1)) # N x 3 x 2
        return torch.einsum('aij,ajk->aik', jacobian.transpose(-1,-2), jacobian).squeeze(0) # N x 2 x 2
    
    def christoffel_symbols(self, point, *args, **kwargs):
        # Flatten points and require gradient
        point = point.view(point.shape[0], -1).requires_grad_(True)
        # Get the inverse of the metric
        g_inv = torch.inverse(self.metric(point)) # (n, n)

        # Compute the derivatives of the metric
        dg = torch.func.jacfwd(lambda u: self.metric(u.view(1,-1), *args, **kwargs))(point.flatten())
        # dg shape: (n, n, n) - order: (row of metric, col of metric, derivative dimension)

        # where dg[m, k, l] = \partial_{x^l} g_{mk}
        # build T[m,k,l] = dg[m,k,l] + dg[m,l,k] - dg[k,l,m]
        T = torch.einsum("mkl->mkl", dg) + torch.einsum("mlk->mkl", dg) - torch.einsum("klm->mkl", dg)
        
        # Returns the Christoffel symbols at the given points, computed with torch
        return 0.5 * torch.einsum('im,mkl->ikl', g_inv, T)  # (i,k,l)

    def get_quantities(self, points, *args, **kwargs):
        jacobians = self.jacobian(points, *args, **kwargs) # N x 3 x 2
        metrics = torch.einsum('aij,ajk->aik', jacobians.transpose(-1,-2), jacobians)
        pseudoinverses = torch.linalg.pinv(jacobians) # equivalent to g^{-1} @ J
        normals = torch.linalg.cross(jacobians[:, :, 0], jacobians[:, :, 1])
        normals = normals / torch.linalg.norm(normals, axis=1, keepdims=True)
        projection_matrices = torch.eye(3) - normals[:, :, None] @ normals[:, None, :]
        return {'J': jacobians, 'g': metrics, 'J_inv': pseudoinverses, 'n': normals, 'P': projection_matrices}

    def __geodesic_equation__(self, position, velocity, **kwargs):
        if type(position) is np.ndarray:
            position = torch.tensor(position)
            velocity = torch.tensor(velocity)

        Gamma = self.christoffel_symbols(position[None, :], **kwargs)  # 2 x 2 x 2
        # Compute the acceleration using the geodesic equation
        acceleration = -torch.einsum('kij,i,j->k', Gamma, velocity.flatten(), velocity.flatten())  # 2
        return acceleration.detach().view(-1,1).numpy()

    def __get_parameter_space_curve__(self, position, velocity, **kwargs):
        # Input: v, x (Dx1)
        position = position.reshape(-1, 1)
        velocity = velocity.reshape(-1, 1)

        # Define the ODE function for the geodesic equation
        ode_fun = lambda t, c_dc: second2first_order(self.__geodesic_equation__, c_dc, **kwargs).flatten()  # The solver needs this shape (D,)
        init_state = np.concatenate([position, velocity]).flatten()

        # Solve the ODE system using the backbone
        solution = solve_ivp(ode_fun, [0, 1], init_state, dense_output=True, atol = 1e-3, rtol= 1e-6)
        # Return a function that evaluates the solution at given time points
        curve = lambda tt: evaluate_solution(solution, tt, 1)  # curve is a function of time t
        return curve

    def tangent_projection_matrix(self, jacobian):
        """Project ambient noise sample onto the tangent space at loc."""
        normals = torch.linalg.cross(jacobian[:, 0], jacobian[:, 1])
        normals = normals / torch.linalg.norm(normals, keepdims=True)
        return torch.eye(3) - normals[:, None] @ normals[None, :]

    def geodesic(self, position, init_v, geodesic_res=101):
        """Compute the geodesic starting at u0 with initial velocity noise_samples, evaluated at times ts. z is a point in the normal direction to fix the degree of freedom."""
        # Evaluation timesteps        
        ts = torch.linspace(0, 1, geodesic_res)

        # Define a function to compute the geodesic for a single initial state
        parameter_space_curve = self.__get_parameter_space_curve__(position, init_v)(ts)[0].T 
        parameter_space_curve = torch.tensor(parameter_space_curve) if type(parameter_space_curve) is np.ndarray else parameter_space_curve.detach()

        # Compute the 3D coordinates of the parameter space curves (i.e. the geodesics)
        trajectories = self(parameter_space_curve)
        return trajectories, parameter_space_curve
        
    def __compute_brownian_update__(self, position, noise_sample, stepsize):
        # Compute the metric and its inverse at the points
        position = position.view(position.shape[0], -1)
        g = self.metric(position)
        g_inv = torch.inverse(g)
        det_g = torch.det(g)
        sqrt_g_inv = matrix_sqrt(g_inv)
        
        def q(points): # Quantity whose derivative we need
            g = self.metric(points.view(1,-1))
            g_inv = torch.inverse(g)
            return torch.sqrt(torch.det(g)) * g_inv

        dq = torch.func.jacrev(q)(position.flatten())

        sum_terms = torch.einsum("lkl->k", dq)
        drift_term = 1 / torch.sqrt(det_g) * sum_terms
        noise_term = (sqrt_g_inv @ noise_sample.view(-1,1)).view(1,-1)
        update = 0.5 * stepsize * drift_term + torch.sqrt(stepsize) * noise_term
        return update

    def brownian_motion(self, position, diffusion_time, num_steps=100, return_trajectories=True, noise_samples=None):
        """Simulate Brownian motion on the Swiss roll."""
        # Extract shapes of the inputs
        position = position.view(1, -1)
        N, D = position.shape
        assert N == 1, "Only single point Brownian motion is implemented. Use vmap for batching."
        
        # Define stepsize
        stepsize = diffusion_time / num_steps
        if self.return_torch:
            stepsize = torch.tensor(stepsize)

        if noise_samples is None:  # generate noise samples
            noise_samples = torch.randn(num_steps, 1, D)

        final_state, trajectory = rollout(
            f=lambda pos, noise: self.__compute_brownian_update__(pos, noise, stepsize) + pos,
            init=position,
            xs=noise_samples,
        )

        if return_trajectories:
            return torch.vmap(self.__call__)(trajectory), trajectory
        else:
            return self.__call__(final_state), final_state