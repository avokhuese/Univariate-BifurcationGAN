"""
Minimal working example for BifurcationGAN
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data_loader_univariate_robust import load_synthetic_dataset
from config_univariate_test import config

# Create a minimal generator for testing
class SimpleBifurcationGenerator(nn.Module):
    def __init__(self, latent_dim=32, seq_len=50, hidden_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Simple architecture
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, seq_len * hidden_dim)
        
        # Bifurcation layer (simplified)
        self.bifurcation = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, z):
        batch_size = z.shape[0]
        
        # FC layers
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        # Reshape to (batch, hidden_dim, seq_len)
        x = x.view(batch_size, self.hidden_dim, self.seq_len)
        
        # Bifurcation dynamics
        x = self.bifurcation(x)
        
        # Transpose to (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)
        
        # Output layer
        output = self.output(x)
        
        return output

# Simple discriminator
class SimpleDiscriminator(nn.Module):
    def __init__(self, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        
        self.model = nn.Sequential(
            nn.Linear(seq_len, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Flatten time series
        x = x.view(x.size(0), -1)
        return self.model(x)

def test_minimal_training():
    """Test minimal training loop"""
    print("Testing minimal BifurcationGAN training...")
    
    # Create synthetic data
    data_info = load_synthetic_dataset("Test", n_samples=100, seq_len=config.seq_len)
    data = torch.FloatTensor(data_info['data']).unsqueeze(-1)  # Add channel dimension
    
    print(f"Data shape: {data.shape}")
    
    # Create models
    generator = SimpleBifurcationGenerator(
        latent_dim=config.latent_dim,
        seq_len=config.seq_len,
        hidden_dim=config.generator_hidden
    )
    
    discriminator = SimpleDiscriminator(seq_len=config.seq_len)
    
    # Move to device
    device = config.device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    data = data.to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=config.generator_lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config.discriminator_lr, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training loop
    n_epochs = 10
    batch_size = config.batch_size
    
    for epoch in range(n_epochs):
        # Train discriminator
        d_optimizer.zero_grad()
        
        # Real samples
        real_labels = torch.ones(batch_size, 1).to(device)
        real_samples = data[:batch_size]
        real_output = discriminator(real_samples)
        d_real_loss = criterion(real_output, real_labels)
        
        # Fake samples
        z = torch.randn(batch_size, config.latent_dim).to(device)
        fake_samples = generator(z).detach()
        fake_labels = torch.zeros(batch_size, 1).to(device)
        fake_output = discriminator(fake_samples)
        d_fake_loss = criterion(fake_output, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # Train generator
        g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, config.latent_dim).to(device)
        fake_samples = generator(z)
        fake_output = discriminator(fake_samples)
        
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: D_loss = {d_loss.item():.4f}, G_loss = {g_loss.item():.4f}")
    
    # Test generation
    generator.eval()
    with torch.no_grad():
        z = torch.randn(5, config.latent_dim).to(device)
        generated = generator(z).cpu().numpy()
        
        # Plot results
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        
        # Plot real samples
        for i in range(3):
            axes[0, i].plot(data[i, :, 0].cpu().numpy())
            axes[0, i].set_title(f"Real Sample {i}")
            axes[0, i].grid(True, alpha=0.3)
        
        # Plot generated samples
        for i in range(3):
            axes[1, i].plot(generated[i, :, 0])
            axes[1, i].set_title(f"Generated Sample {i}")
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle("Minimal BifurcationGAN Test", fontsize=14)
        plt.tight_layout()
        plt.savefig("test_bifurcation_gan.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    print("\n✅ Minimal test completed successfully!")
    print(f"Generated samples shape: {generated.shape}")
    print("Plot saved as: test_bifurcation_gan.png")

if __name__ == "__main__":
    test_minimal_training()