# %%
import torch
import torch.nn as nn
from torchdiffeq import odeint

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    seed = 3
    torch.manual_seed(seed)

    # Define velocity field
    velocity_field = VelocityField(alpha=18, hidden_dim=64)

    # from geometric_noise.manifolds import Sphere, DeformedSphere
    # manifold

    # Example usage
    batch_size = 30000
    sphere_points = torch.randn(batch_size, 3)
    sphere_points = sphere_points / sphere_points.norm(dim=1, keepdim=True)  # normalize to unit sphere
    targets = sphere_points[:, 2]

    # Function to integrate deformation
    def deform_sphere(points, velocity_field, t=torch.tensor([0.0, 1.0])):
        return odeint(lambda t, x: velocity_field(x), points, t)

    cmap = 'RdBu_r'

    # ts = torch.arange(0.0, 1.1, 0.25)
    ts = torch.tensor([0.0, 0.25, 0.5, 1.0])
    deformed_points = deform_sphere(sphere_points, velocity_field, t=ts).detach().numpy()
    # reproduced_points = deform_sphere(deformed_points, velocity_field, t=torch.tensor([1.0, 0.0])).detach().numpy()
    
    fig, axs = plt.subplots(1, len(ts), subplot_kw={'projection': '3d'}, figsize=(20, 4))
    for i, ax in enumerate(axs):
        ax.view_init(elev=34, azim=151, roll=36)
        ax.set_title(fr'$t={ts[i]:.2g}$')
        ax.scatter(deformed_points[i, :, 0], deformed_points[i, :, 1], deformed_points[i, :, 2], c=targets, cmap=cmap, s=10)
        ax.set_facecolor('white')
        ax.set_aspect('equal')
        ax.axis('off')
    fig.savefig('sphere_deformation_time.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

#%%