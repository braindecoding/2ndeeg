#!/usr/bin/env python3
# eeg_to_image_reconstruction.py - Image reconstruction from EEG using Brain2Image dataset

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from brain2image_dataset import Brain2ImageDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EEGToImageVAE(nn.Module):
    """Variational Autoencoder for EEG to Image reconstruction"""
    
    def __init__(self, eeg_features_dim, image_size=32, latent_dim=128):
        super(EEGToImageVAE, self).__init__()
        
        self.eeg_features_dim = eeg_features_dim
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # EEG Encoder
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_features_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Image Decoder
        self.image_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_size * image_size),
            nn.Sigmoid()
        )
    
    def encode(self, eeg_features):
        h = self.eeg_encoder(eeg_features)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.image_decoder(z)
    
    def forward(self, eeg_features):
        mu, logvar = self.encode(eeg_features)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed.view(-1, 1, self.image_size, self.image_size), mu, logvar

class EEGToImageGAN(nn.Module):
    """Generative Adversarial Network for EEG to Image reconstruction"""
    
    def __init__(self, eeg_features_dim, image_size=32, latent_dim=128):
        super(EEGToImageGAN, self).__init__()
        
        self.eeg_features_dim = eeg_features_dim
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # EEG to latent mapping
        self.eeg_to_latent = nn.Sequential(
            nn.Linear(eeg_features_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim),
            nn.Tanh()
        )
        
        # Generator (latent to image)
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_size * image_size),
            nn.Tanh()
        )
    
    def forward(self, eeg_features):
        latent = self.eeg_to_latent(eeg_features)
        generated = self.generator(latent)
        return generated.view(-1, 1, self.image_size, self.image_size)

def vae_loss(reconstructed, original, mu, logvar, beta=0.001):
    """VAE loss function"""
    # Reconstruction loss
    recon_loss = F.mse_loss(reconstructed, original, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss

def train_eeg_to_image_model(model, train_loader, val_loader, num_epochs=100, model_type='vae'):
    """Train EEG to Image reconstruction model"""
    print(f"\n🚀 Training {model_type.upper()} model...")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for eeg_features, images in train_loader:
            eeg_features, images = eeg_features.to(device), images.to(device)
            
            optimizer.zero_grad()
            
            if model_type == 'vae':
                reconstructed, mu, logvar = model(eeg_features)
                loss = vae_loss(reconstructed, images, mu, logvar)
            else:  # GAN
                reconstructed = model(eeg_features)
                loss = F.mse_loss(reconstructed, images)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for eeg_features, images in val_loader:
                eeg_features, images = eeg_features.to(device), images.to(device)
                
                if model_type == 'vae':
                    reconstructed, mu, logvar = model(eeg_features)
                    loss = vae_loss(reconstructed, images, mu, logvar)
                else:  # GAN
                    reconstructed = model(eeg_features)
                    loss = F.mse_loss(reconstructed, images)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"  ⚠️ Early stopping at epoch {epoch+1}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title(f'{model_type.upper()} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_type}_training_history.png')
    print(f"  📊 Training history saved as '{model_type}_training_history.png'")
    
    return model

def evaluate_reconstruction(model, test_loader, model_type='vae', n_examples=8):
    """Evaluate image reconstruction quality"""
    print(f"\n📊 Evaluating {model_type.upper()} reconstruction...")
    
    model.eval()
    all_mse = []
    all_ssim = []
    
    # Collect examples for visualization
    example_originals = []
    example_reconstructed = []
    example_eeg = []
    
    with torch.no_grad():
        for i, (eeg_features, images) in enumerate(test_loader):
            eeg_features, images = eeg_features.to(device), images.to(device)
            
            if model_type == 'vae':
                reconstructed, _, _ = model(eeg_features)
            else:  # GAN
                reconstructed = model(eeg_features)
            
            # Calculate MSE
            mse = F.mse_loss(reconstructed, images, reduction='none').mean(dim=[1,2,3])
            all_mse.extend(mse.cpu().numpy())
            
            # Calculate SSIM (simplified version)
            for j in range(images.size(0)):
                orig = images[j].cpu().numpy().squeeze()
                recon = reconstructed[j].cpu().numpy().squeeze()
                
                # Simplified SSIM calculation
                mu1, mu2 = np.mean(orig), np.mean(recon)
                sigma1, sigma2 = np.std(orig), np.std(recon)
                sigma12 = np.mean((orig - mu1) * (recon - mu2))
                
                c1, c2 = 0.01**2, 0.03**2
                ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
                all_ssim.append(ssim)
            
            # Collect examples for visualization
            if len(example_originals) < n_examples:
                for j in range(min(n_examples - len(example_originals), images.size(0))):
                    example_originals.append(images[j].cpu().numpy().squeeze())
                    example_reconstructed.append(reconstructed[j].cpu().numpy().squeeze())
                    example_eeg.append(eeg_features[j].cpu().numpy())
    
    mean_mse = np.mean(all_mse)
    mean_ssim = np.mean(all_ssim)
    
    print(f"  📊 Mean MSE: {mean_mse:.6f}")
    print(f"  📊 Mean SSIM: {mean_ssim:.4f}")
    
    # Visualize reconstruction examples
    visualize_reconstruction_results(example_originals, example_reconstructed, 
                                   example_eeg, model_type, mean_mse, mean_ssim)
    
    return mean_mse, mean_ssim

def visualize_reconstruction_results(originals, reconstructed, eeg_features, 
                                   model_type, mean_mse, mean_ssim, n_examples=8):
    """Visualize reconstruction results"""
    print(f"  📊 Creating reconstruction visualization...")
    
    fig, axes = plt.subplots(3, n_examples, figsize=(20, 8))
    
    for i in range(min(n_examples, len(originals))):
        # Original image
        axes[0, i].imshow(originals[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed image
        axes[1, i].imshow(reconstructed[i], cmap='gray', vmin=0, vmax=1)
        mse = np.mean((originals[i] - reconstructed[i])**2)
        axes[1, i].set_title(f'Reconstructed\nMSE: {mse:.4f}')
        axes[1, i].axis('off')
        
        # EEG features (simplified visualization)
        axes[2, i].plot(eeg_features[i][:50])  # Plot first 50 features
        axes[2, i].set_title(f'EEG Features')
        axes[2, i].set_xlabel('Feature Index')
        axes[2, i].set_ylabel('Value')
    
    plt.suptitle(f'{model_type.upper()} Reconstruction Results\n'
                f'Mean MSE: {mean_mse:.6f}, Mean SSIM: {mean_ssim:.4f}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{model_type}_reconstruction_results.png', dpi=300, bbox_inches='tight')
    print(f"    ✅ Visualization saved as '{model_type}_reconstruction_results.png'")
    
    plt.close()

def cross_subject_reconstruction_validation():
    """Perform cross-subject validation for image reconstruction"""
    print("🚀 Cross-Subject Image Reconstruction Validation")
    print("=" * 60)
    
    # Initialize Brain2Image dataset
    dataset = Brain2ImageDataset()
    
    # Create demo data if not exists
    if not os.path.exists(dataset.data_path):
        print("📁 Creating demo Brain2Image dataset...")
        dataset.create_demo_brain2image_data()
    
    # Test reconstruction on each subject
    subjects_to_test = [1, 2, 3, 4]
    reconstruction_results = {}
    
    for subject_id in subjects_to_test:
        print(f"\n🧪 Testing Image Reconstruction for Subject {subject_id:02d}")
        print("-" * 60)
        
        # Load subject data
        eeg_data, labels, images = dataset.load_subject_data(subject_id)
        
        if eeg_data is None:
            print(f"  ⚠️ Skipping subject {subject_id} - data not available")
            continue
        
        # Extract features
        features = dataset.extract_brain2image_features(eeg_data)
        
        # Prepare data for reconstruction
        # Normalize images to [0, 1]
        images_normalized = (images - images.min()) / (images.max() - images.min())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, images_normalized, test_size=0.3, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val).unsqueeze(1)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.FloatTensor(y_test).unsqueeze(1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Train VAE model
        print("  🔧 Training VAE model...")
        vae_model = EEGToImageVAE(
            eeg_features_dim=features.shape[1],
            image_size=32,
            latent_dim=64
        ).to(device)
        
        vae_model = train_eeg_to_image_model(vae_model, train_loader, val_loader, 
                                           num_epochs=50, model_type='vae')
        
        # Evaluate VAE
        vae_mse, vae_ssim = evaluate_reconstruction(vae_model, test_loader, 'vae')
        
        # Save results
        reconstruction_results[subject_id] = {
            'vae_mse': vae_mse,
            'vae_ssim': vae_ssim,
            'n_test_samples': len(X_test)
        }
        
        # Save model
        torch.save(vae_model.state_dict(), f'vae_subject_{subject_id:02d}.pth')
        print(f"  💾 VAE model saved as 'vae_subject_{subject_id:02d}.pth'")
    
    # Summary of reconstruction results
    if len(reconstruction_results) > 0:
        print(f"\n📊 Cross-Subject Image Reconstruction Summary")
        print("=" * 60)
        
        vae_mses = [result['vae_mse'] for result in reconstruction_results.values()]
        vae_ssims = [result['vae_ssim'] for result in reconstruction_results.values()]
        
        mean_vae_mse = np.mean(vae_mses)
        mean_vae_ssim = np.mean(vae_ssims)
        
        print(f"VAE Results:")
        print(f"  Mean MSE: {mean_vae_mse:.6f} ± {np.std(vae_mses):.6f}")
        print(f"  Mean SSIM: {mean_vae_ssim:.4f} ± {np.std(vae_ssims):.4f}")
        
        # Detailed results per subject
        print(f"\nDetailed Results per Subject:")
        print("-" * 40)
        for subject_id, result in reconstruction_results.items():
            print(f"Subject {subject_id:02d}: MSE={result['vae_mse']:.6f}, "
                  f"SSIM={result['vae_ssim']:.4f} ({result['n_test_samples']} samples)")
        
        # Save results
        np.save('cross_subject_reconstruction_results.npy', reconstruction_results)
        print(f"\n📊 Results saved as 'cross_subject_reconstruction_results.npy'")
        
        return reconstruction_results
    
    else:
        print("❌ No subjects were successfully tested for reconstruction")
        return None

def main():
    """Main function"""
    print("🧠 EEG to Image Reconstruction with Brain2Image Dataset")
    print("=" * 60)
    
    # Run cross-subject reconstruction validation
    results = cross_subject_reconstruction_validation()
    
    if results:
        vae_mses = [result['vae_mse'] for result in results.values()]
        vae_ssims = [result['vae_ssim'] for result in results.values()]
        
        mean_mse = np.mean(vae_mses)
        mean_ssim = np.mean(vae_ssims)
        
        print("\n✅ Cross-subject image reconstruction completed!")
        print(f"📊 Mean reconstruction MSE: {mean_mse:.6f}")
        print(f"📊 Mean reconstruction SSIM: {mean_ssim:.4f}")
        
        # Interpretation
        print(f"\n🔍 Reconstruction Quality Analysis:")
        if mean_ssim > 0.7:
            print("  ✅ Excellent reconstruction quality")
        elif mean_ssim > 0.5:
            print("  ⚠️ Good reconstruction quality")
        else:
            print("  ❌ Limited reconstruction quality")
        
        print(f"\n📝 Recommendations:")
        print("  1. Experiment with different latent dimensions")
        print("  2. Try more sophisticated architectures (e.g., ConvVAE)")
        print("  3. Implement perceptual loss functions")
        print("  4. Use real Brain2Image dataset for validation")
        print("  5. Explore conditional generation based on EEG categories")
        
    else:
        print("\n❌ Cross-subject reconstruction validation failed!")

if __name__ == "__main__":
    main()
