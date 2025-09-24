import time
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from geometric_noise.manifolds import SwissRoll, Sphere, SqueezedSphere, OnionRing, Bead, DeformedSphere
from joblib import Parallel, delayed

def get_manifold(args):
    if args.manifold_name == 'swiss-roll':
        return SwissRoll(return_torch=True)
    elif args.manifold_name == 'sphere':
        return Sphere(return_torch=True)
    elif args.manifold_name == 'squeezed-sphere':
        return SqueezedSphere(return_torch=True)
    elif args.manifold_name == 'deformed-sphere':
        return DeformedSphere(return_torch=True)
    elif args.manifold_name == 'onion-ring':
        return OnionRing(return_torch=True)
    elif args.manifold_name == 'bead':
        return Bead(return_torch=True)
    else:
        raise ValueError(f"Unknown manifold name: {args.manifold_name}")

if __name__ == '__main__':
    import os
    import argparse
    import matplotlib.pyplot as plt

    # Define arguments
    parser = argparse.ArgumentParser(description='Train a neural network on Swiss Roll data')
    parser.add_argument('--n-train', type=int, default=200, help='Number of training samples')
    parser.add_argument('--n-hidden', type=int, default=64, help='Number of hidden units in the neural network')
    parser.add_argument('--num-epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--noise-intensity', type=float, help='Noise intensity for the geometric noise')
    parser.add_argument('--manifold-name', type=str, help='Name of the manifold to use')
    parser.add_argument('--noise-type', type=str, help='Type of noise to use', choices=['none', 'ambient', 'brownian', 'tangent', 'geodesic'])
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility', default=42)
    parser.add_argument('--num-steps', type=int, help='Number of steps for the random walks', default=100)
    args = parser.parse_args()

    # Define the manifold
    manifold = get_manifold(args)
    noise_intensity = torch.tensor(args.noise_intensity)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    with torch.no_grad():
        # Generate data from the manifold
        X_train, U_train, y_train, X_test, U_test, y_test = manifold.generate_data(N_train=args.n_train)

        # Get geometric quantities at the starting points
        geometric_quantities = manifold.get_quantities(U_train)
        # Compute covariance matrix of the transformed tangent space noise
        transformed_covariances = geometric_quantities['J_inv'] @ geometric_quantities['P'] @ geometric_quantities['J_inv'].transpose(-1,-2)
        # Compute Cholesky decomposition for sampling
        Ls = torch.linalg.cholesky(transformed_covariances)                

    # Define the model
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], args.n_hidden),
        nn.ReLU(),
        nn.Linear(args.n_hidden, args.n_hidden),
        nn.ReLU(),
        nn.Linear(args.n_hidden, 1)
    )
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    storage = {'train': [], 'test': [], 'time': []}
    for epoch in tqdm(range(args.num_epochs), desc=f'{manifold.__class__.__name__} - {args.noise_type} - {args.noise_intensity} -- {args.seed}', colour='blue'):
        # Track time taken for the epoch
        start = time.time()
        model.train(); optimizer.zero_grad()

        # Add noise according to the specified type
        if args.noise_type == 'ambient':
            # Gaussian noise samples in R^D
            noise_samples = torch.randn_like(X_train) * torch.sqrt(noise_intensity) 
            X_train_mod = X_train + noise_samples

        elif args.noise_type == 'tangent':
            # Sample centered Gaussian noise in R^D
            noise_samples = torch.randn_like(X_train) * torch.sqrt(noise_intensity)
            # Project noise sample to the tangent space
            Ps = torch.vmap(manifold.tangent_projection_matrix)(geometric_quantities['J'])
            tangent_space_noise = (Ps @ noise_samples[:, :, None]).squeeze(-1)
            X_train_mod = X_train + tangent_space_noise

        elif args.noise_type == 'geodesic':
            # Sample standard normal vectors and transform them to the parameter space as initial velocities
            init_vs = torch.sqrt(noise_intensity) * torch.einsum('ijk,ik->ij', Ls, torch.randn_like(U_train))
            init_states = torch.hstack([U_train, init_vs])
            
            geodesics = Parallel(n_jobs=-1)(delayed(lambda state: manifold.geodesic(state[:2], state[2:], geodesic_res=101)[0])(init_state) for init_state in init_states)
            # Take the endpoint of the geodesics as noisy samples
            X_train_mod = torch.stack(geodesics)[:, -1, :].detach()

        elif args.noise_type == 'brownian':
            # Get Brownian motion endpoints for each training point
            BMs = torch.vmap(lambda u: manifold.brownian_motion(u[None, :], diffusion_time=args.noise_intensity, num_steps=args.num_steps)[0], randomness='different')(U_train)
            X_train_mod = BMs[:, -1, :].squeeze(1)

        else: # add no noise
            X_train_mod = X_train

        # Use the model to predict on noisy data
        y_pred = model(X_train_mod)

        # Compute loss, store it and backpropagate
        loss = criterion(y_pred.view(-1,1), y_train.view(-1,1))
        storage['train'].append(loss.item())
        loss.backward()
        # Optimize the model
        optimizer.step()

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            test_loss = criterion(y_pred.view(-1,1), y_test.view(-1,1))
            storage['test'].append(test_loss.item())

        # Track time taken for the epoch and store results
        storage['time'].append(time.time() - start)
    
    # Store the final model
    save_path = f"results/{manifold.__class__.__name__}/{args.noise_type}" + (f"/{args.noise_intensity}" if args.noise_type != 'none' else "/0") + f"/{args.seed}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), f"{save_path}/final.pth")
    torch.save(storage, f"{save_path}/training_stats.pth")

    # Make a plot of the training and test losses
    plt.figure(figsize=(10,5))
    plt.plot(storage['train'], label='Train Loss')
    plt.plot(storage['test'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Test Losses ({manifold.__class__.__name__}, {args.noise_type})')
    plt.legend()
    plt.savefig(f"{save_path}/loss_plot.png")
    plt.close()

    