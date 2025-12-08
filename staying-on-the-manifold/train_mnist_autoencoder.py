import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from copy import deepcopy

# Set floating point precision
torch.set_default_dtype(torch.float32)

def get_loaders(batch_size, **kwargs):
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor()])

    # Load full MNIST dataset
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    full_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Create masks for selected digits if specified
    selected_subset = kwargs.get('selected_subset', range(10))
    train_mask = torch.from_numpy(np.isin(full_train.targets.numpy(), selected_subset)).to(torch.bool)
    test_mask = torch.from_numpy(np.isin(full_test.targets.numpy(), selected_subset)).to(torch.bool)
    # Select indices based on masks
    train_indices = torch.where(train_mask)[0]
    test_indices = torch.where(test_mask)[0]

    # Subsample training data if specified
    p = kwargs.get('subsample_fraction', 1.0)
    num_samples = int(len(train_indices) * p)
    subsample_indices = train_indices[torch.randperm(len(train_indices))[:num_samples]]

    # Create subsets
    mnist_train = torch.utils.data.Subset(full_train, subsample_indices)
    mnist_test = torch.utils.data.Subset(full_test, test_indices)
    # Create data loaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=False)
    return train_loader, test_loader


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, latent_dim=16, activation_fn=lambda: nn.Softplus(beta=100), use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),                             # -> 24x24
            activation_fn(),
            nn.MaxPool2d(2),                                # -> 12x12
            nn.Conv2d(6, 16, 5),                            # -> 8x8
            activation_fn(),
            nn.MaxPool2d(2),                                # -> 4x4
            nn.Flatten(),                                   # -> 16*4*4
            nn.Linear(16*4*4, latent_dim),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16*4*4),
            nn.Unflatten(1, (16, 4, 4)),
            nn.Upsample(scale_factor=2, mode='nearest'),    # 4x4 -> 8x8
            nn.ConvTranspose2d(16, 6, 5),                   # -> 12x12
            activation_fn(),
            nn.Upsample(scale_factor=2, mode='nearest'),    # -> 24x24
            nn.ConvTranspose2d(6, 1, 5),                    # -> 28x28
            nn.Sigmoid() if self.use_sigmoid else nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        if not self.use_sigmoid:
            # Rescale to [0, 1]
            out = (out + 1) / 2  
        return out, z
    

if __name__ == "__main__":
    
    import os
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Train MNIST Autoencoder')
    parser.add_argument('--latent-dim', type=int, default=16, help='Dimensionality of the latent space')
    parser.add_argument('--subsample-fraction', type=float, default=1.0, help='Fraction of training data to use')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=3e-5, help='Weight decay for optimizer')
    parser.add_argument('--use-sigmoid', action='store_true', help='Use sigmoid activation in the decoder output layer')
    parser.add_argument('--save-folder', type=str, default='./results', help='Folder to save results')
    parser.add_argument('--device', type=str, default='system', help='Device to use for training (cuda, mps, cpu)')
    args = parser.parse_args()

    # Define save path
    SAVEPATH = f'{args.save_folder}/mnist_autoencoder/fraction={args.subsample_fraction}_sigmoid={args.use_sigmoid}/seed={args.seed}'
    os.makedirs(SAVEPATH, exist_ok=True)

    # Check if path exists
    if os.path.exists(f"{SAVEPATH}/best.pth"):
        print(f"Model already trained for seed {args.seed} at {SAVEPATH}. Exiting.")
        exit(0)

    # Set seed
    torch.manual_seed(args.seed)

    # Use all 10 digits for training
    selected_subset = range(10)  # Using all digits 0-9
    # Load full MNIST dataset
    train_loader, test_loader = get_loaders(
        args.batch_size,
        subsample_fraction=args.subsample_fraction,
        selected_subset=selected_subset,
    )

    # Create mappings between original labels and subset labels
    mapping = {k: v for k, v in enumerate(selected_subset)}
    inverse_mapping = {v: k for k, v in enumerate(selected_subset)}

    # Select device
    device = args.device if args.device != 'system' else torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate model
    autoencoder = ConvolutionalAutoencoder(latent_dim=args.latent_dim).to(device)
    print("Autoencoder training on device:", device)

    # Define loss and optimizer
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Number of optimization steps
    num_steps = len(train_loader) * args.epochs
    # Define a learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    # Prepare for training
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    eval_every = len(train_loader)  # evaluate every epoch

    # Training loop
    with tqdm(total=num_steps, desc='Training autoencoder') as pbar:
        for step in range(num_steps):
            epoch = step // len(train_loader)

            # Set model to training mode
            autoencoder.train()
            
            # Get a batch of training data and move to device
            imgs, _ = next(iter(train_loader))
            imgs = imgs.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            # En- and decode the images
            outputs, _ = autoencoder(imgs)
            # Compute reconstruction loss and backpropagate
            loss = criterion(outputs, imgs)
            loss.backward()
            
            # Update model parameters and learning rate scheduler
            optimizer.step()
            scheduler.step()

            # Log training loss
            train_losses.append(loss.item())

            # Evaluate on test set periodically
            if step % eval_every == 0:
                with torch.no_grad():
                    autoencoder.eval()
                    test_loss = 0
                    for test_imgs, _ in test_loader:
                        test_imgs = test_imgs.to(device)
                        recon_imgs, _ = autoencoder(test_imgs)
                        test_loss += criterion(recon_imgs, test_imgs).item()

                    test_loss /= len(test_loader.dataset)
                    test_losses.append(test_loss)

                    # Check if this is the best model so far
                    if test_loss < best_test_loss:
                        best_epoch = epoch + 1
                        best_test_loss = test_loss
                        torch.save(deepcopy(autoencoder).to('cpu').state_dict(), f"{SAVEPATH}/best.pth")
                        print(f"New best model saved at epoch {best_epoch} with test loss {best_test_loss:.4f}")

                pbar.set_description(f"Test Loss: {test_loss:.4f}")
            pbar.update(1)

    print("Best test loss and epoch:")
    print(best_test_loss, best_epoch)

    import matplotlib.pyplot as plt
    # Plot training loss and save figure
    plt.figure(figsize=(6,4))
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.savefig(f"{SAVEPATH}/loss.png")
    plt.close()