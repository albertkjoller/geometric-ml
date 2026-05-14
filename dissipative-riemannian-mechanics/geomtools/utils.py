
import torch

def second2first_order(equation, state, **kwargs):
    # state: (B x 2D) where the first D columns are position and the next D columns are velocity
    D = int(state.shape[1] / 2)
    if state.ndim == 1:
        state = state.unsqueeze(0)  # Add batch dimension if missing

    position = state[:, :D]  # D x N
    velocity = state[:, D:]  # D x N
    acceleration = equation(position, velocity, **kwargs)  # D x N
    return torch.stack((velocity, acceleration), dim=1).reshape(-1, 2 * D)  # Flatten back to (B x 2D)

def get_params_structure(vector, true_params):
    splits = [w.numel() for w in true_params]
    chunks = vector.split(splits)
    return tuple(chunk.view_as(w) for chunk, w in zip(chunks, true_params))

def choose_subset(params_vector, subset):
    n = params_vector.numel()
    # TODO: make a faster version without masking, sorting, etc.
    mask = torch.zeros(n, dtype=torch.bool, device=params_vector.device)
    mask[subset] = True
    theta = params_vector[mask]
    remaining_params = params_vector[~mask]
    order = torch.argsort(torch.tensor(subset + (~mask).nonzero(as_tuple=False).view(-1).tolist(), device=params_vector.device))
    return theta, remaining_params, order