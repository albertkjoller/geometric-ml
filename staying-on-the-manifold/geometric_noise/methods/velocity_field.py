import torch
import torch.nn as nn

# Neural ODE defining velocity field on the sphere
class VelocityField(nn.Module):
    def __init__(self, alpha=1.0, hidden_dim=16):
        super().__init__()

        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, x):
        v = self.net(x)
        return self.alpha * v