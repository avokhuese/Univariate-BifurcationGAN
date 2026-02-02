"""
Advanced GAN framework for univariate time series with bifurcation dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
#import wandb
import warnings
warnings.filterwarnings('ignore')

from models_univariate import create_model
from baseline_models_univariate import create_baseline_model
from config_univariate import config

class AdvancedUnivariateGAN:
    """Advanced GAN framework supporting multiple architectures"""
    
    def __init__(self, model_type: str, config):
        self.config = config
        self.model_type = model_type
        self.device = config.device
        # Initialize ALL attributes to avoid AttributeError
        self.generator = None
        self.discriminator = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.scheduler_G = None
        self.scheduler_D = None
        self.criterion = None
        self.scaler = None
        self.is_baseline = False

        # Initialize models
        self._initialize_models()
        
        # Initialize optimizers
        self._initialize_optimizers()
        
        # Initialize loss functions
        self._initialize_loss_functions()
        
        # Initialize training statistics
        self.training_stats = {
            'g_losses': [],
            'd_losses': [],
            'gradient_penalties': [],
            'wasserstein_distances': []
        }
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp and torch.cuda.is_available() else None
        
        print(f"Initialized {model_type} on {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters())}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters())}")
    # ==================== Make the GAN callable ====================
    def __call__(self, z: torch.Tensor, seq_len: Optional[int] = None):
        """
        Allows the GAN object to be called directly like gan(z)
        Calls the generator internally.
        """
        self.generator.eval()
        with torch.no_grad():
            if seq_len is not None:
                return self.generator(z, seq_len)
            else:
                return self.generator(z)
                    
    # def _initialize_models(self):
    #     """Initialize generator and discriminator"""
        
    #     # For baseline models that have their own wrappers
    #     if self.model_type in ["vanilla_gan", "wgan"]:
    #         # These use simple wrappers from baseline_models_univariate
    #         gan_model = create_baseline_model(self.model_type, self.config)
    #         self.generator = gan_model.generator
    #         self.discriminator = gan_model.discriminator
    #         self.train_step = gan_model.train_step
    #         self.is_baseline = True
    #         return
        
    #     # For advanced models
    #     self.is_baseline = False
    #     self.generator, self.discriminator = create_model(
    #         self.model_type, self.config.latent_dim, self.config
    #     )
        
    #     # Apply spectral normalization if needed
    #     if self.config.use_spectral_norm and "wgan_gp" in self.model_type:
    #         self.generator = self._apply_spectral_norm(self.generator)
    #         self.discriminator = self._apply_spectral_norm(self.discriminator)
    #     # Move to device
    #     self.generator = self.generator.to(self.device)
    #     self.discriminator = self.discriminator.to(self.device)
    # Update the _initialize_models method:
    def _initialize_models(self):
        """Initialize generator and discriminator - FIXED VERSION"""
        
        # For baseline models that have their own wrappers
        baseline_models = ["vanilla_gan", "wgan"]
        
        if self.model_type in baseline_models:
            print(f"  Using baseline model wrapper for {self.model_type}")
            try:
                from baseline_models_univariate import create_baseline_model
                gan_model = create_baseline_model(self.model_type, self.config)
                
                # Extract models from the baseline wrapper
                if hasattr(gan_model, 'generator'):
                    self.generator = gan_model.generator
                if hasattr(gan_model, 'discriminator'):
                    self.discriminator = gan_model.discriminator
                
                # IMPORTANT FIX: Store a reference to the baseline model's train_step
                # but wrap it properly to avoid tuple issues
                if hasattr(gan_model, 'train_step'):
                    # Store the baseline model itself, not just the method
                    self.baseline_model = gan_model
                    self.is_baseline = True
                    print(f"  Successfully loaded baseline model")
                else:
                    print(f"  Warning: Baseline model missing train_step method")
                    self.is_baseline = False
                    self.baseline_model = None
                    
            except Exception as e:
                print(f"  Error loading baseline model: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to advanced model
                self.is_baseline = False
                self.baseline_model = None
                self.generator, self.discriminator = create_model(
                    self.model_type, self.config.latent_dim, self.config
                )
        else:
            # For advanced models
            self.is_baseline = False
            self.baseline_model = None
            self.generator, self.discriminator = create_model(
                self.model_type, self.config.latent_dim, self.config
            )
        
        # Apply spectral normalization if needed
        if self.config.use_spectral_norm and "wgan_gp" in self.model_type:
            self.generator = self._apply_spectral_norm(self.generator)
            self.discriminator = self._apply_spectral_norm(self.discriminator)
        
        # Move to device
        if self.generator:
            self.generator = self.generator.to(self.device)
        else:
            print(f"  Warning: Generator is None for model {self.model_type}")
            
        if self.discriminator:
            self.discriminator = self.discriminator.to(self.device)
        else:
            print(f"  Warning: Discriminator is None for model {self.model_type}")


    def _apply_spectral_norm(self, model: nn.Module) -> nn.Module:
        """Apply spectral normalization to linear and conv layers"""
        def apply_sn(module):
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                return nn.utils.spectral_norm(module)
            return module
        
        return model.apply(apply_sn)
    
    def _initialize_optimizers(self):
        """Initialize optimizers based on model type"""
        
        if self.is_baseline:
            return
        
        # Optimizer parameters
        beta1, beta2 = self.config.beta1, self.config.beta2
        
        # Generator optimizer
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.generator_lr,
            betas=(beta1, beta2),
            weight_decay=self.config.weight_decay
        )
        
        # Discriminator optimizer
        if "wgan" in self.model_type:
            # Use RMSprop for WGAN variants
            self.optimizer_D = torch.optim.RMSprop(
                self.discriminator.parameters(),
                lr=self.config.discriminator_lr
            )
        else:
            self.optimizer_D = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.config.discriminator_lr,
                betas=(beta1, beta2),
                weight_decay=self.config.weight_decay
            )
        
        # Learning rate schedulers
        self.scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_G, mode='min', patience=10, factor=0.5
        )
        self.scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_D, mode='min', patience=10, factor=0.5
        )
    
    def _initialize_loss_functions(self):
        """Initialize loss functions based on GAN type"""
        
        if self.is_baseline:
            return
        
        # Binary cross entropy for vanilla GANs
        if "vanilla" in self.model_type or "tts_gan" == self.model_type:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # WGAN uses wasserstein loss
        elif "wgan" in self.model_type:
            self.criterion = None  # WGAN loss is implemented separately
        
        else:
            # Default to least squares GAN loss
            self.criterion = lambda output, target: torch.mean((output - target) ** 2)
    
    def compute_gradient_penalty(self, real_samples: torch.Tensor, 
                                fake_samples: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP"""
        
        # Random weight term for interpolation
        alpha = torch.rand(real_samples.size(0), 1, 1, device=self.device)
        
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        # Get discriminator output for interpolates
        d_interpolates = self.discriminator(interpolates)
        
        # Get gradients w.r.t. interpolates
        fake = torch.ones(real_samples.size(0), 1, device=self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def compute_consistency_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss for generated samples"""
        
        with torch.no_grad():
            # Generate samples from similar latent codes
            gen1 = self.generator(z1)
            gen2 = self.generator(z2)
        
        # Compute feature consistency
        consistency_loss = F.mse_loss(gen1, gen2)
        
        return consistency_loss
    
    def compute_feature_matching_loss(self, real_samples: torch.Tensor, 
                                     fake_samples: torch.Tensor) -> torch.Tensor:
        """Compute feature matching loss"""
        
        # Extract features from discriminator if possible
        if hasattr(self.discriminator, 'feature_extractor'):
            # Get features for real and fake samples
            real_features = self.discriminator.feature_extractor(real_samples.transpose(1, 2))
            fake_features = self.discriminator.feature_extractor(fake_samples.transpose(1, 2))
            
            # Compute feature matching loss
            feature_loss = F.mse_loss(real_features.mean(dim=0), fake_features.mean(dim=0))
        else:
            # Fallback to simple statistical matching
            real_mean = real_samples.mean(dim=[0, 1])
            fake_mean = fake_samples.mean(dim=[0, 1])
            real_std = real_samples.std(dim=[0, 1])
            fake_std = fake_samples.std(dim=[0, 1])
            
            feature_loss = F.mse_loss(real_mean, fake_mean) + F.mse_loss(real_std, fake_std)
        
        return feature_loss
    
    def compute_diversity_loss(self, fake_samples: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss to prevent mode collapse"""
        
        # Compute pairwise distances
        batch_size = fake_samples.size(0)
        fake_flat = fake_samples.view(batch_size, -1)
        
        # Normalize
        fake_norm = F.normalize(fake_flat, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(fake_norm, fake_norm.t())
        
        # Exclude diagonal
        mask = torch.eye(batch_size, device=self.device).bool()
        similarity = similarity[~mask].view(batch_size, batch_size - 1)
        
        # Diversity loss encourages dissimilarity
        diversity_loss = torch.mean(similarity)
        
        return diversity_loss
    
    def compute_temporal_coherence_loss(self, generated_samples: torch.Tensor) -> torch.Tensor:
        """Compute temporal coherence loss"""
        
        # Compute differences between consecutive time steps
        diff = generated_samples[:, 1:, :] - generated_samples[:, :-1, :]
        
        # Compute variance of differences
        temporal_loss = -torch.var(diff)  # Negative because we want higher variance
        
        return temporal_loss
    
    # def train_step(self, real_batch: torch.Tensor) -> Dict[str, float]:
    #     """Single training step for advanced GANs"""
        
    #     if self.is_baseline:
    #         # For baseline models, use their train_step method
    #         return self.train_step(real_batch)
        
    #     real_batch = real_batch.to(self.device)
    #     batch_size = real_batch.size(0)
        
    #     # Prepare labels
    #     real_labels = torch.ones(batch_size, 1, device=self.device)
    #     fake_labels = torch.zeros(batch_size, 1, device=self.device)
    def train_step(self, real_batch: torch.Tensor) -> Dict[str, float]:
        """Single training step for ALL models - FIXED VERSION"""
        
        # Check if we have a baseline model
        if self.baseline_model is not None and hasattr(self.baseline_model, 'train_step'):
            try:
                # Call the baseline model's train_step method
                result = self.baseline_model.train_step(real_batch)
                
                # Ensure result is a dict (some baseline models return tuple)
                if isinstance(result, tuple) and len(result) >= 2:
                    # Convert tuple to dict
                    return {'g_loss': result[0], 'd_loss': result[1]}
                elif isinstance(result, dict):
                    return result
                else:
                    print(f"  Warning: train_step returned unexpected type: {type(result)}")
                    return {'g_loss': 0.0, 'd_loss': 0.0}
                    
            except Exception as e:
                print(f"  Baseline train_step failed: {e}")
                # Fall through to advanced training
        
        # Check if we have optimizers for advanced training
        if self.optimizer_G is None or self.optimizer_D is None:
            print("  Error: Optimizers not initialized for advanced training")
            return {'g_loss': 0.0, 'd_loss': 0.0}
        
        # Advanced training logic (keep existing code)
        real_batch = real_batch.to(self.device)
        batch_size = real_batch.size(0)
        
        # Prepare labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device) 
        # ==================== Train Discriminator ====================
        self.optimizer_D.zero_grad()
        
        # Generate fake samples
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake_batch = self.generator(z)
        
        # Compute discriminator losses
        if "wgan_gp" in self.model_type:
            # WGAN-GP loss
            real_validity = self.discriminator(real_batch)
            fake_validity = self.discriminator(fake_batch.detach())
            
            # Wasserstein loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(real_batch.data, fake_batch.data)
            d_loss = d_loss + self.config.lambda_gp * gradient_penalty
            
            # Drift penalty
            if self.config.lambda_drift > 0:
                drift_penalty = torch.mean(real_validity ** 2)
                d_loss = d_loss + self.config.lambda_drift * drift_penalty
        
        elif "wgan" in self.model_type:
            # WGAN loss
            real_validity = self.discriminator(real_batch)
            fake_validity = self.discriminator(fake_batch.detach())
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        
        else:
            # Standard GAN loss
            real_validity = self.discriminator(real_batch)
            fake_validity = self.discriminator(fake_batch.detach())
            
        # Use BCE or MSE loss
        if hasattr(self, 'criterion') and self.criterion:
            d_real_loss = self.criterion(real_validity, real_labels)
            d_fake_loss = self.criterion(fake_validity, fake_labels)
        else:
            # Default to MSE
            d_real_loss = F.mse_loss(real_validity, real_labels)
            d_fake_loss = F.mse_loss(fake_validity, fake_labels)
            
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        # Add feature matching loss if enabled
        if self.config.use_feature_matching:
            feature_loss = self.compute_feature_matching_loss(real_batch, fake_batch.detach())
            d_loss = d_loss + self.config.feature_matching_lambda * feature_loss
        
        # Backward and optimize
        d_loss.backward()
        self.optimizer_D.step()
        
        # ==================== Train Generator ====================
        self.optimizer_G.zero_grad()
        
        # Generate new fake samples
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake_batch = self.generator(z)
        
        # Compute generator loss
        if "wgan" in self.model_type:
            # WGAN generator loss
            fake_validity = self.discriminator(fake_batch)
            g_loss = -torch.mean(fake_validity)
        
        else:
            # Standard GAN generator loss
            fake_validity = self.discriminator(fake_batch)
            g_loss = self.criterion(fake_validity, real_labels)
        
        # Add additional losses
        total_g_loss = g_loss
        
        # Diversity loss
        if self.config.diversity_weight > 0:
            diversity_loss = self.compute_diversity_loss(fake_batch)
            total_g_loss = total_g_loss + self.config.diversity_weight * diversity_loss
        
        # Temporal coherence loss
        if self.config.temporal_coherence_weight > 0:
            temporal_loss = self.compute_temporal_coherence_loss(fake_batch)
            total_g_loss = total_g_loss + self.config.temporal_coherence_weight * temporal_loss
        
        # Spectral consistency loss
        if self.config.spectral_consistency_weight > 0:
            # Simple spectral loss (variance matching)
            real_spectrum = torch.fft.rfft(real_batch, dim=1).abs()
            fake_spectrum = torch.fft.rfft(fake_batch, dim=1).abs()
            spectral_loss = F.mse_loss(real_spectrum.mean(dim=0), fake_spectrum.mean(dim=0))
            total_g_loss = total_g_loss + self.config.spectral_consistency_weight * spectral_loss
        
        # Consistency regularization
        if self.config.use_consistency_regularization:
            z1 = torch.randn(batch_size, self.config.latent_dim, device=self.device)
            z2 = z1 + torch.randn_like(z1) * 0.1  # Slight perturbation
            consistency_loss = self.compute_consistency_loss(z1, z2)
            total_g_loss = total_g_loss + self.config.consistency_lambda * consistency_loss
        
        # Backward and optimize
        total_g_loss.backward()
        self.optimizer_G.step()
        
        # ==================== Collect Statistics ====================
        stats = {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'total_g_loss': total_g_loss.item()
        }
        
        if "wgan_gp" in self.model_type:
            stats['gradient_penalty'] = gradient_penalty.item() if 'gradient_penalty' in locals() else 0
        
        # Store in history
        self.training_stats['g_losses'].append(stats['g_loss'])
        self.training_stats['d_losses'].append(stats['d_loss'])
        
        # Update learning rates
        self.scheduler_G.step(stats['g_loss'])
        self.scheduler_D.step(stats['d_loss'])
        
        return stats
    
    def train_epoch(self, train_loader, epoch: int, log_interval: int = 100) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.generator.train()
        self.discriminator.train()
        
        epoch_stats = {
            'g_loss': 0.0,
            'd_loss': 0.0,
            'total_g_loss': 0.0
        }
        
        n_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            real_data = batch['data'].to(self.device)
            
            # Training step
            step_stats = self.train_step(real_data)
            
            # Accumulate statistics
            for key in epoch_stats:
                if key in step_stats:
                    epoch_stats[key] += step_stats[key]
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                print(f'Epoch {epoch} [{batch_idx+1}/{n_batches}] '
                      f'G Loss: {step_stats["g_loss"]:.4f} '
                      f'D Loss: {step_stats["d_loss"]:.4f}')
        
        # Average statistics
        for key in epoch_stats:
            epoch_stats[key] /= n_batches
        
        return epoch_stats
    
    def generate_samples(self, n_samples: int, seq_len: Optional[int] = None) -> torch.Tensor:
        """Generate synthetic samples"""
        self.generator.eval()
        
        with torch.no_grad():
            # Generate in batches
            samples = []
            batch_size = min(self.config.batch_size, n_samples)
            
            for i in range(0, n_samples, batch_size):
                current_batch = min(batch_size, n_samples - i)
                z = torch.randn(current_batch, self.config.latent_dim, device=self.device)
                
                if seq_len:
                    batch_samples = self.generator(z, seq_len)
                else:
                    batch_samples = self.generator(z)
                
                samples.append(batch_samples.cpu())
            
            samples = torch.cat(samples, dim=0)
        
        return samples
    
    def save_checkpoint(self, epoch: int, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            #'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            #'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            #'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            #'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        self.training_stats = checkpoint['training_stats']
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch']
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Generator loss
        axes[0, 0].plot(self.training_stats['g_losses'])
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Discriminator loss
        axes[0, 1].plot(self.training_stats['d_losses'])
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Combined loss
        axes[1, 0].plot(self.training_stats['g_losses'], label='Generator')
        axes[1, 0].plot(self.training_stats['d_losses'], label='Discriminator')
        axes[1, 0].set_title('Combined Loss')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss ratio
        if len(self.training_stats['g_losses']) > 0 and len(self.training_stats['d_losses']) > 0:
            ratio = np.array(self.training_stats['d_losses']) / (np.array(self.training_stats['g_losses']) + 1e-8)
            axes[1, 1].plot(ratio)
            axes[1, 1].set_title('D/G Loss Ratio')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Ratio')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.model_type} Training Curves', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        plt.show()

# def create_gan_framework(model_type: str, config) -> AdvancedUnivariateGAN:
#         """Factory function to create GAN framework"""
#         return AdvancedUnivariateGAN(model_type, config)
def create_gan_framework(model_type: str, config) -> AdvancedUnivariateGAN:
    """Factory function to create GAN framework - FIXED VERSION"""
    try:
        gan = AdvancedUnivariateGAN(model_type, config)
        
        # Verify the gan is properly initialized
        if gan.generator is None:
            print(f"Warning: Generator not created for {model_type}")
        if gan.discriminator is None:
            print(f"Warning: Discriminator not created for {model_type}")
        
        return gan
    except Exception as e:
        print(f"Error creating GAN framework for {model_type}: {e}")
        # Return a minimal working GAN as fallback
        return _create_minimal_gan_fallback(config)

def _create_minimal_gan_fallback(config):
    """Create a minimal GAN as fallback when other methods fail"""
    print("Creating minimal GAN fallback...")
    
    class MinimalGenerator(nn.Module):
        def __init__(self, latent_dim=32, seq_len=100):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, seq_len),
                nn.Tanh()
            )
        
        def forward(self, z):
            batch_size = z.size(0)
            output = self.model(z)
            return output.view(batch_size, -1, 1)
    
    class MinimalDiscriminator(nn.Module):
        def __init__(self, seq_len=100):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(seq_len, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.model(x)
    
    # Create minimal config
    class MinimalConfig:
        def __init__(self, config):
            self.latent_dim = config.latent_dim
            self.seq_len = config.seq_len
            self.generator_lr = config.generator_lr
            self.discriminator_lr = config.discriminator_lr
            self.beta1 = config.beta1
            self.beta2 = config.beta2
            self.weight_decay = config.weight_decay
            self.device = config.device
            self.batch_size = config.batch_size
    
    minimal_config = MinimalConfig(config)
    
    # Create a simple wrapper
class MinimalGAN:
    def __init__(self, config):
            self.config = config
            self.device = config.device
            
            self.generator = MinimalGenerator(config.latent_dim, config.seq_len).to(self.device)
            self.discriminator = MinimalDiscriminator(config.seq_len).to(self.device)
            
            self.optimizer_G = torch.optim.Adam(
                self.generator.parameters(),
                lr=config.generator_lr,
                betas=(config.beta1, config.beta2)
            )
            
            self.optimizer_D = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=config.discriminator_lr,
                betas=(config.beta1, config.beta2)
            )
            
            self.criterion = nn.BCELoss()
        
    def train_step(self, real_batch):
            real_batch = real_batch.to(self.device)
            batch_size = real_batch.size(0)
            
            # Real and fake labels
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)
            
            # Train Discriminator
            self.optimizer_D.zero_grad()
            
            # Real samples
            real_validity = self.discriminator(real_batch)
            d_real_loss = self.criterion(real_validity, real_labels)
            
            # Fake samples
            z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
            fake_batch = self.generator(z).detach()
            fake_validity = self.discriminator(fake_batch)
            d_fake_loss = self.criterion(fake_validity, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.optimizer_D.step()
            
            # Train Generator
            self.optimizer_G.zero_grad()
            
            z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
            fake_batch = self.generator(z)
            fake_validity = self.discriminator(fake_batch)
            
            g_loss = self.criterion(fake_validity, real_labels)
            g_loss.backward()
            self.optimizer_G.step()
            
            return {'g_loss': g_loss.item(), 'd_loss': d_loss.item()}
        
    def generate_samples(self, n_samples):
            self.generator.eval()
            with torch.no_grad():
                z = torch.randn(n_samples, self.config.latent_dim, device=self.device)
                return self.generator(z).cpu()
        

    def train_epoch(self, train_loader, epoch: int, log_interval: int = 100) -> Dict[str, float]:
            """Train for one epoch"""
            
            self.generator.train()
            self.discriminator.train()
            
            epoch_stats = {
                'g_loss': 0.0,
                'd_loss': 0.0,
                'total_g_loss': 0.0
            }
            
            n_batches = len(train_loader)
            
            for batch_idx, batch in enumerate(train_loader):
                real_data = batch['data'].to(self.device)
                
                # Training step
                step_stats = self.train_step(real_data)
                
                # Accumulate statistics
                for key in epoch_stats:
                    if key in step_stats:
                        epoch_stats[key] += step_stats[key]
                
                # Log progress
                if (batch_idx + 1) % log_interval == 0:
                    print(f'Epoch {epoch} [{batch_idx+1}/{n_batches}] '
                        f'G Loss: {step_stats["g_loss"]:.4f} '
                        f'D Loss: {step_stats["d_loss"]:.4f}')
            
            # Average statistics
            for key in epoch_stats:
                epoch_stats[key] /= n_batches
            
            return epoch_stats
    def save_checkpoint(self, epoch: int, path: str):
            """Save model checkpoint"""
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                #'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                #'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                #'scheduler_G_state_dict': self.scheduler_G.state_dict(),
                #'scheduler_D_state_dict': self.scheduler_D.state_dict(),
                #'training_stats': self.training_stats,
                #'config': self.config.to_dict()
            }
            
            torch.save(checkpoint, path)
            print(f"Checkpoint saved to {path}")

            return MinimalGAN(minimal_config)