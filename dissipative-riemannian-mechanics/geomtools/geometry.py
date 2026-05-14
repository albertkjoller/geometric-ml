import torch
import numpy as np
from torchdiffeq import odeint
from .utils import second2first_order, choose_subset


class EmbeddedLossManifold:
    def __init__(self, network: torch.nn.Module, weight_subset=None, solver='dopri5', rtol=1e-7, atol=1e-9, reduction='sum'):
        assert (weight_subset is None) or (type(weight_subset) == list), "weight_subset should be a list of parameter indices." 

        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.network = network
        self.param_names = [n for n, _ in network.named_parameters()]
        self.params_structure = list(network.parameters())
        self.reduction = reduction
        self.num_params = sum(p.numel() for p in network.parameters())

        # Choose the subset of parameters, if specified
        self.weight_subset = weight_subset if weight_subset is not None else list(range(self.num_params))
        self.map_estimate, self.remaining_params, self.order = choose_subset(torch.nn.utils.parameters_to_vector(network.parameters()).detach(), self.weight_subset)
        self.ignore_ordering = (self.order.cpu().tolist() == torch.arange(self.num_params).tolist()) # if ordering is the same as original, no need to reorder

    def predict(self, params, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def loss_fn(self, params, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __call__(self, params, **kwargs):
        assert len(self.weight_subset) == params.shape[0], "Input params size does not match the selected weight subset size."
        loss_value = self.loss_fn(params, **kwargs)
        return torch.hstack([params, loss_value]).view(1,-1)  # Return the loss as part of the output to compute Jacobian etc

    def compute_metric(self, X, y, selected_params_vector):
        """ Computes the metric wrt. parameters, with the data defined in update_model_info."""
        grad = self.compute_gradient(X, y, selected_params_vector).view(-1,1) # D x 1
        metric = torch.eye(grad.shape[0], device=grad.device) + grad @ grad.T # D x D
        return metric

    def compute_sqrt_inverse_metric(self, X, y, selected_params_vector, grad=None):
        """ Computes the square root of the inverse of the metric wrt. parameters, with the data defined in update_model_info."""
        if grad is None:
            grad = self.compute_gradient(X, y, selected_params_vector).view(-1,1) # D x 1
        grad_norm = grad.T @ grad # 1 x 1
        sqrt_inv_metric = torch.eye(grad.shape[0], device=grad.device) - (grad @ grad.T) / (1 + grad_norm + torch.sqrt(1 + grad_norm)) # D x D
        return sqrt_inv_metric

    def compute_gradient(self, X, y, selected_params_vector):
        """ Computes the gradient wrt. parameters, with the data defined in update_model_info."""
        return torch.func.jacrev(lambda p: self.loss_fn(p, X=X, y=y))(selected_params_vector.flatten())

    def compute_hessian(self, X, y, selected_params_vector):
        """ Computes the Hessian wrt. parameters, with the data defined in update_model_info."""
        hessian_fn = torch.func.jacfwd(torch.func.jacrev(lambda p: self.loss_fn(p, X=X, y=y)), randomness='same')
        return hessian_fn(selected_params_vector.flatten())

    def compute_hvp(self, X, y, selected_params_vector, v):
        """ Computes the Hessian-vector product wrt. parameters, with the data defined in update_model_info."""
        grad, hvp = torch.func.jvp(torch.func.grad(lambda p: self.loss_fn(p.clone(), X=X, y=y)), (selected_params_vector.flatten(),), (v.flatten(),))
        return grad.view(selected_params_vector.shape), hvp.view(selected_params_vector.shape)

    def __geodesic_equation__(self, position, velocity, **kwargs):
        if type(position) is np.ndarray:
            position = torch.tensor(position)
            velocity = torch.tensor(velocity)
        
        # Compute the velocity-Hessian product via Hessian-vector product (HVP) or directly using the Hessian
        if kwargs.get('parallel', "batch") == "joblib":
            grad, Hv = zip(*[
                self.compute_hvp(kwargs['X'], kwargs['y'], position[i], velocity[i])
                for i in range(position.shape[0])
            ])
            grad = torch.stack(grad)
            Hv = torch.stack(Hv)
        else:
            grad, Hv = torch.vmap(lambda p, v: self.compute_hvp(kwargs['X'], kwargs['y'], p, v))(position, velocity)
        
        # Compute the velocity-Hessian product v^T H v using the HVP result
        vHv = torch.einsum('bi,bi->b', velocity, Hv).unsqueeze(1)  # (B,1)

        # Compute the gradient and geodesic acceleration
        grad_prod = grad.norm(dim=1).pow(2).unsqueeze(1)  # (B,1)
        riemannian_grad = grad / (1 + grad_prod) # (B,D)
        acceleration = - riemannian_grad * vHv # (B,D)

        if kwargs.get('correction', False):
            # Get the correction coefficients from kwargs, with default values
            gravity_coeff = kwargs.get('gravity_coeff', 1.0)
            friction_coeff = kwargs.get('friction_coeff', 1.0)

            # Compute velocity norm and velocity-gradient product
            v_norm_squared = velocity.norm(dim=1).pow(2).unsqueeze(1)  # (B,1)
            v_grad_prod_squared = torch.einsum('bi,bi->b', velocity, grad).pow(2).unsqueeze(1)  # (B,1)
            kinetic_energy = 0.5 * (v_norm_squared + v_grad_prod_squared) # kinetic energy 1/2 v^T G v: simplifies to v^T v + (v^T grad)^2 due to the metric structure

            if kwargs.get('correction', None) == 'gravity':
                correction = - gravity_coeff * riemannian_grad # (B,D)            
            elif kwargs.get('correction', None) == 'friction':
                correction = - riemannian_grad - friction_coeff * velocity # (B,D)
            elif kwargs.get('correction', None) == 'kinetic_friction': # dynamics as proposed in the paper
                correction = - gravity_coeff * riemannian_grad - friction_coeff * torch.sqrt(kinetic_energy) * velocity # (B,D)

            # Update the acceleration with the correction term
            acceleration = acceleration + correction # (B,D)

        return acceleration

    def __get_parameter_space_curve__(self, position, velocity, **kwargs):
        # position: (D x 1)
        # velocity: (B x D)
        B, D = velocity.shape

        # Repeat position to match the batch size of velocity
        position = position.repeat(B, 1)  # D x B 

        # Define the ODE function for the geodesic equation
        ode_fun = lambda t, c_dc: second2first_order(self.__geodesic_equation__, c_dc, **kwargs) # (2D x B) -> (2D x B)
        init_state = torch.hstack([position, velocity]) # B x 2D, where the first D columns are position and the next D columns are velocity

        # Define a curve function as a function of time
        def curve(tt):
            solution = odeint(ode_fun, init_state, tt, method=self.solver, atol=self.atol, rtol=self.rtol) # T x B x 2D
            return (solution[:, :, :D], solution[:, :, D:]) # (T x B x D, T x B x D) for both position and velocity
        return curve
    
    def geodesic(self, position, init_vs, geodesic_res=101, **kwargs):
        """Compute the geodesic starting at u0 with initial velocity noise_samples, evaluated at times ts. z is a point in the normal direction to fix the degree of freedom."""
        if init_vs.ndim == 1:
            init_vs = init_vs.unsqueeze(0)

        # Evaluation timesteps
        ts = (torch.linspace(0, kwargs.get('t_run', 1.0), geodesic_res) if geodesic_res > 1 else torch.tensor([kwargs.get('t_run', 1.0)])).to(position.device)
        
        # Define a function to compute the geodesic for a single initial state
        parameter_space_curve, velocities = self.__get_parameter_space_curve__(position, init_vs, **kwargs)(ts)
        return parameter_space_curve, velocities


class RegressionManifold(EmbeddedLossManifold):
    def __init__(self, network, solver='dopri5', rtol=1e-7, atol=1e-9, reduction='mean', weight_subset=None, prior_precision=0.0, noise_variance=1.0):
        super().__init__(network=network, solver=solver, atol=atol, rtol=rtol, reduction=reduction, weight_subset=weight_subset)
        self.reduction = reduction
        self.network = network
        self.prior_precision = prior_precision
        self.noise_variance = noise_variance

    def predict(self, params, X):
        p = torch.concatenate([params.flatten(), self.remaining_params])
        if not self.ignore_ordering:
            p = torch.index_select(p, 0, self.order)

        # Build the network parameters without breaking graph
        param_dict = {name: chunk.view_as(w) for name, chunk, w in zip(self.param_names, p.split([w.numel() for w in self.params_structure]), self.params_structure)}
        return torch.func.functional_call(self.network, param_dict, (X,))

    def loss_fn(self, params, **kwargs):
        # Extract the data from kwargs
        X, y = kwargs['X'], kwargs['y']

        # Compute the loss value (negative log likelihood under Gaussian noise model)
        loss_value = (self.predict(params, X) - y).pow(2) / (2 * self.noise_variance)
        
        # Add L2 regularization term to the loss --> equivalent to a diagonal Gaussian prior on the parameters
        neg_log_prior = (0.5 * self.prior_precision * params.pow(2).sum()).squeeze() # sum over all parameters, then remove extra dimensions
        # Return the total loss value (negative log posterior)
        loss_value = (loss_value.mean() if self.reduction == 'mean' else loss_value.sum()) + neg_log_prior
        return loss_value


class CrossEntropyLossManifold(EmbeddedLossManifold):
    def __init__(self, network, solver='dopri5', rtol=1e-7, atol=1e-9, reduction='mean', weight_subset=None, prior_precision=0.0, label_smoothing=0.0):
        super().__init__(network=network, solver=solver, atol=atol, rtol=rtol, reduction=reduction, weight_subset=weight_subset)
        self.reduction = reduction
        self.network = network
        self.prior_precision = prior_precision
        self.label_smoothing = label_smoothing

    def loss_fn(self, params, **kwargs):
        # Extract the data from kwargs
        X, y = kwargs['X'], kwargs['y']
        p = torch.concatenate([params.flatten(), self.remaining_params])
        if not self.ignore_ordering:
            p = torch.index_select(p, 0, self.order)

        # Build the network parameters without breaking graph
        param_dict = {name: chunk.view_as(w) for name, chunk, w in zip(self.param_names, p.split([w.numel() for w in self.params_structure]), self.params_structure)}
        log_probs = torch.func.functional_call(self.network, param_dict, (X,))
        neg_log_likelihood = torch.nn.functional.nll_loss(log_probs, y.flatten(), reduction='none')

        # Smoothing term: mean of log-probs across all classes
        smooth_loss = -log_probs.mean(dim=-1)
        # Blend: (1 - ε) * nll + ε * smooth
        loss = (1 - self.label_smoothing) * neg_log_likelihood + self.label_smoothing * smooth_loss
        
        # Add L2 regularization term to the loss --> equivalent to a diagonal Gaussian prior on the parameters
        neg_log_prior = (0.5 * self.prior_precision * params.pow(2).sum()).squeeze() # sum over all parameters, then remove extra dimensions
        # Return the total loss value (negative log posterior)
        loss_value = (loss.mean() if self.reduction == 'mean' else loss.sum()) + neg_log_prior
        return loss_value