from geometric_noise.manifolds import manifold
from geometric_noise.manifolds.manifold import Manifold
from train_mnist_autoencoder import ConvolutionalAutoencoder, get_loaders

import numpy as np
import pandas as pd
import torch
from torch import nn
torch.set_default_dtype(torch.float32)


class ImageManifold(Manifold):
    def __init__(self, checkpoint, device, return_torch=True):
        super().__init__(return_torch=return_torch)

        # Define the autoencoder here
        self.autoencoder = ConvolutionalAutoencoder()
        self.autoencoder.load_state_dict(torch.load(checkpoint))
        self.autoencoder.to(device)
        self.autoencoder.eval()

    def __call__(self, latent_representation):
        with torch.no_grad():
            self.autoencoder.eval()
            # Decode the latent representation
            out = self.autoencoder.decoder(latent_representation)
            if not self.autoencoder.use_sigmoid:
                # Rescale to [0, 1]
                out = (out + 1) / 2  
            return out


# class ClassifierInformedImageManifold(Manifold):
#     def __init__(self, checkpoint, latent_dim, in_channels, activation_fn, pretraining_epoch=0, noise_intensity=0, device='cpu', return_torch=True):
#         super().__init__(return_torch=return_torch)

#         # Define the autoencoder here
#         self.autoencoder = ConvolutionalAutoencoder(in_channels=in_channels, latent_dim=latent_dim, activation_fn=activation_fn)
#         self.autoencoder.load_state_dict(torch.load(checkpoint))
#         self.autoencoder.to(device)
#         self.autoencoder.eval()

#         self.classifier = get_classifier(hidden_dim=hidden_dim, n_classes=n_classes, device=device)
#         if bm_noise != 0.0:
#             self.classifier.load_state_dict(torch.load(f'classifier_{method}={noise_intensity}_sigmoid={use_sigmoid}.pth'))
#         elif pretraining_epoch != 0:
#             self.classifier.load_state_dict(torch.load(f'classifier_epoch_{pretraining_epoch}.pth'))
#         else:
#             pass

#     def __call__(self, latent_representation):
#         self.autoencoder.eval()
#         with torch.no_grad():
#             flat_reconstruction = self.autoencoder.decoder(latent_representation).flatten(start_dim=1)
#             flat_reconstruction = (flat_reconstruction + 1) / 2  # Rescale to [0, 1]
#             classifier_outputs = self.classifier(flat_reconstruction)
#             return torch.hstack([flat_reconstruction, classifier_outputs])


def get_classifier(hidden_dim, n_classes, device):
    # Define the classifier
    return nn.Sequential(
        nn.Linear(28*28, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, n_classes),
    ).to(device)

def compute_accuracies(classifier, train_loader, test_loader, device):
    classifier.eval()

    # Collect all training predictions and labels
    all_train_preds, all_train_labels = [], []
    for imgs, labels in train_loader:
        outputs = classifier(imgs.view(imgs.size(0), -1).to(device))
        all_train_preds.append(torch.max(outputs, 1)[1])
        all_train_labels.append(labels)
        
    # Stack all training predictions and labels
    all_train_labels = torch.hstack(all_train_labels).cpu()
    all_train_preds = torch.hstack(all_train_preds).cpu()

    # Collect all test predictions and labels
    all_test_preds, all_test_labels = [], []
    for imgs, labels in test_loader:
        outputs = classifier(imgs.view(imgs.size(0), -1).to(device))
        all_test_preds.append(torch.max(outputs, 1)[1])
        all_test_labels.append(labels)

    # Stack all test predictions and labels
    all_test_preds = torch.hstack(all_test_preds).cpu()
    all_test_labels = torch.hstack(all_test_labels).cpu()

    # Compute accuracies
    test_accuracy = (all_test_labels == all_test_preds).float().mean()
    train_accuracy = (all_train_labels == all_train_preds).float().mean()
    return train_accuracy, test_accuracy

def save_results(savepath, subsample_fraction, seed, train_accuracy, test_accuracy, method, noise_intensity):
    # Construct a CSV-file with the results
    results_df = pd.DataFrame({
        'subsample_fraction': [subsample_fraction],
        'train_accuracy': [train_accuracy.item()],
        'test_accuracy': [test_accuracy.item()],
        'method': [method],
        'noise_intensity': [noise_intensity],
        'seed': [seed],
    })

    if os.path.exists(f'{savepath}/results.csv'):
        # Load existing results
        existing_df = pd.read_csv(f'{savepath}/results.csv')
        # Check if the current configuration already exists
        mask = ((existing_df['subsample_fraction'] == subsample_fraction) &
                (existing_df['method'] == method) &
                (existing_df['noise_intensity'] == noise_intensity) &
                (existing_df['seed'] == seed))
        # If it exists, skip saving
        if existing_df[mask].shape[0] > 0:
            print("Results for this configuration already exist. Skipping saving.")
        else:
            results_df = pd.concat([existing_df, results_df], ignore_index=True)
            results_df.to_csv(f'{savepath}/results.csv', index=False)
            print(f"Results saved to {savepath}/results.csv")
    else:
        results_df.to_csv(f'{savepath}/results.csv', index=False)


if __name__ == "__main__":

    import os
    import argparse
    from tqdm import tqdm
    from copy import deepcopy

    parser = argparse.ArgumentParser(description='Train MNIST classifier with geometric noise')
    parser.add_argument('--autoencoder-checkpoint', type=str, help='Path to the autoencoder checkpoint')
    parser.add_argument('--subsample-fraction', type=float, default=0.01, help='Fraction of training data to use')
    parser.add_argument('--batch-size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use-sigmoid', action='store_true', help='Use sigmoid activation in the decoder output layer')
    parser.add_argument('--save-folder', type=str, default='./results', help='Folder to save results')
    parser.add_argument('--device', type=str, default='system', help='Device to use for training (cuda, mps, cpu)')
    parser.add_argument('--method', type=str, help='Method used for training')
    parser.add_argument('--noise-intensity', type=float, help='Noise intensity used during training')
    args = parser.parse_args()

    # Define save path
    SAVEFOLDER = f'{args.save_folder}/{args.autoencoder_checkpoint}'
    SAVEPATH = f'{SAVEFOLDER}/seed={args.seed}/fraction={args.subsample_fraction}/{args.method}/noise_intensity={args.noise_intensity}'

    # Check if results already exist for this configuration
    if not os.path.exists(f'{SAVEFOLDER}/seed={args.seed}/best.pth'):
        print(f"Autoencoder does not exist for seed={args.seed}. Please train the autoencoder first.")
        exit()

    if os.path.exists(f'{SAVEFOLDER}/results.csv'):
        existing_results = pd.read_csv(f'{SAVEFOLDER}/results.csv')
        mask = ((existing_results['subsample_fraction'] == args.subsample_fraction) &
                (existing_results['method'] == args.method) &
                (existing_results['noise_intensity'] == args.noise_intensity) &
                (existing_results['seed'] == args.seed))
        
        if existing_results[mask].shape[0] > 0:
            print(f"Results for noise_intensity={args.noise_intensity} already exist. Skipping.")
            exit()

    # Create save directory
    os.makedirs(SAVEPATH, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)

    # Use all 10 digits for training
    selected_subset = range(10)  # Using all digits 0-9
    n_classes = len(selected_subset)

    # Get dataloaders
    train_loader, test_loader = get_loaders(args.batch_size, selected_subset=selected_subset, subsample_fraction=args.subsample_fraction)

    # Select device
    device = args.device if args.device != 'system' else torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the classifier, returns logits
    classifier = get_classifier(hidden_dim=args.hidden_dim, n_classes=n_classes, device=device)
    # Define loss, optimizer and learning rate scheduler
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Define the manifold
    manifold = ImageManifold(
        checkpoint=f'{SAVEFOLDER}/seed={args.seed}/best.pth',
        device=device,
    )

    # Training loop
    best_test_acc = 0.0
    losses = {'train': [], 'test': []}
    with tqdm(total=args.epochs, desc="Training Classifier") as pbar:
        for epoch in range(args.epochs):
            classifier.train()

            # Train on training set
            for imgs, labels in train_loader:
                # Flatten images and move to device
                imgs = imgs.to(device)
                labels = labels.to(device).to(torch.long)
                
                if args.method == 'reconstructed':
                    # Reconstruct with the autoencoder
                    imgs = manifold.autoencoder(imgs)[0].flatten(start_dim=1)
                    
                elif args.method == 'brownian':
                    # Encode to latent space
                    latent_reps = manifold.autoencoder.encoder(imgs)
                    # Add intrinsic Brownian motion noise in latent space and get trajectories in image space
                    img_trajectories, _ = torch.vmap(
                        lambda x: manifold.brownian_motion(
                            position=x, 
                            diffusion_time=args.noise_intensity, 
                            num_steps=10, 
                            return_trajectories=True
                        ), 
                    randomness='different')(latent_reps)
                    # Take the endpoints as the noisy samples
                    imgs = img_trajectories[:, -1, 0, :].flatten(start_dim=1)

                elif args.method == 'ambient':
                    # Add ambient Gaussian noise and clamp to [0, 1]
                    imgs = imgs.flatten(start_dim=1)
                    imgs = imgs + torch.randn_like(imgs) * np.sqrt(args.noise_intensity)
                    imgs = torch.clamp(imgs, 0., 1.)

                else:
                    # Flatten images
                    imgs = imgs.flatten(start_dim=1)
                
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass to get logits and log probabilities
                logits = classifier(imgs)
                log_probs = torch.log_softmax(logits, dim=1)

                # Compute loss, backpropagate and optimize
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                # Store training loss
                losses['train'].append(loss.item())

            # Step the learning rate scheduler every epoch
            scheduler.step()
            
            # Evaluate on test set every epoch
            with torch.no_grad():
                classifier.eval()

                # Evaluate on test set
                correct, batch_losses = 0, 0
                for imgs, labels in test_loader:
                    # Flatten images and move to device
                    imgs = imgs.flatten(start_dim=1).to(device)
                    labels = labels.to(device).to(torch.long)

                    # Forward pass to get logits and log probabilities
                    logits = classifier(imgs)
                    log_probs = torch.log_softmax(logits, dim=1)
                    # Get predictions
                    _, predicted = torch.max(log_probs, 1)

                    # Count correct predictions and accumulate loss
                    correct += (predicted == labels).sum().item()
                    batch_losses += criterion(log_probs, labels).item()

                # Store test loss and accuracy for this epoch
                losses['test'].append(batch_losses / len(test_loader))
                accuracy = correct / (len(test_loader.dataset))
                
                # Save the best model
                if accuracy > best_test_acc:
                    best_test_acc = accuracy
                    best_model_state = deepcopy(classifier.state_dict())
                    torch.save(deepcopy(classifier).to('cpu').state_dict(), f"{SAVEPATH}/classifier.pth")

                # Set progress bar description
                pbar.set_description(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {losses['train'][-1]:.4f} - Test Loss: {losses['test'][-1]:.4f} - Test Acc: {accuracy*100:.2f}%")
                pbar.update(1)

    # Compute the final test and train accuracies using the best model
    classifier.load_state_dict(best_model_state)
    train_accuracy, test_accuracy = compute_accuracies(classifier, train_loader, test_loader, device)

    # Print results to console
    print("Final Results:")
    print(f"Subsample fraction: {args.subsample_fraction}")
    print(f"Train Accuracy: {train_accuracy.item()*100:.2f}%")
    print(f"Test Accuracy: {test_accuracy.item()*100:.2f}%")

    # Save results
    save_results(
        method=args.method, 
        noise_intensity=args.noise_intensity,
        train_accuracy=train_accuracy, 
        test_accuracy=test_accuracy, 
        savepath=SAVEFOLDER,
        seed=args.seed, 
        subsample_fraction=args.subsample_fraction, 
    )

    # Plot training and test losses
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(torch.linspace(0, 1, len(losses['train'])), losses['train'], label='Train Loss')
    plt.plot(torch.linspace(0, 1, len(losses['test'])), losses['test'], label='Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Classifier Training Losses (Method: {args.method}, Noise Intensity: {args.noise_intensity})')
    plt.savefig(f"{SAVEPATH}/loss_plot.png")
    plt.show()
