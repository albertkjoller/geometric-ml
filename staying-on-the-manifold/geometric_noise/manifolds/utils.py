import torch
import numpy as np
from scipy.integrate import solve_ivp

def second2first_order(equation, state, **kwargs):
    # Input: state [c; dc] (2D x N), y=[dc; ddc]: (2D x N)
    D = int(state.shape[0] / 2)
    if state.ndim == 1:
        state = state.reshape(-1, 1)

    c = state[:D, :]  # D x N
    cm = state[D:, :]  # D x N
    cmm = equation(c, cm, **kwargs)  # D x N
    return np.concatenate((cm, cmm), axis=0)

def evaluate_solution(solution, t, t_scale):
    # Input: t (Tx0), t_scale is used from the Expmap to scale the curve in order to have correct length, solution is an object that solver_bvp() returns
    c_dc = solution.sol(t * t_scale)
    D = int(c_dc.shape[0] / 2)

    # TODO: Why the t_scale is used ONLY for the derivative component?
    if np.size(t) == 1:
        c = c_dc[:D].reshape(D, 1)
        dc = c_dc[D:].reshape(D, 1) * t_scale
    else:
        c = c_dc[:D, :]  # D x T
        dc = c_dc[D:, :] * t_scale  # D x T
    return c, dc
