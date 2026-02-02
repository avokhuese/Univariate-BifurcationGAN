import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from config_univariate import config

# ===================== SIGNATURE TRANSFORM UTILITIES =====================

class UnivariateSignatureTransform(nn.Module):
    """Signature transform for univariate time series"""
    
    def __init__(self, depth: int = 3):
        super().__init__()
        self.depth = depth
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute signature transform for univariate series"""
        batch_size, seq_len, _ = x.shape
        
        # Add time dimension
        time = torch.linspace(0, 1, seq_len, device=x.device).unsqueeze(0).unsqueeze(-1)
        time = time.expand(batch_size, seq_len, 1)
        x_augmented = torch.cat([time, x], dim=-1)  # (batch, seq_len, 2)
        
        signature_terms = []
        
        # Level 1: increments
        increments = x_augmented[:, 1:, :] - x_augmented[:, :-1, :]
        signature_terms.append(increments.mean(dim=1))
        
        # Level 2: second order terms
        if self.depth >= 2:
            # Compute iterated integrals
            for i in range(2):  # time and value dimensions
                for j in range(2):
                    if i <= j:
                        term = x_augmented[:, :, i] * x_augmented[:, :, j]
                        signature_terms.append(term.mean(dim=1, keepdim=True))
        
        # Combine terms
        signature = torch.cat(signature_terms, dim=-1)
        
        return signature

# ===================== BASELINE GENERATORS =====================

class VanillaGenerator(nn.Module):
    """Vanilla GAN Generator"""
    
    def __init__(self, latent_dim: int, config: Any):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = config.seq_len
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.seq_len),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        seq_len = seq_len or self.seq_len
        batch_size = z.size(0)
        
        output = self.model(z)
        output = output.view(batch_size, seq_len, 1)
        
        return output

class WGANGenerator(nn.Module):
    """WGAN Generator"""
    
    def __init__(self, latent_dim: int, config: Any):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = config.seq_len
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.seq_len),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for WGAN stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        seq_len = seq_len or self.seq_len
        batch_size = z.size(0)
        
        output = self.model(z)
        output = output.view(batch_size, seq_len, 1)
        
        return output

class WGANGPGenerator(WGANGenerator):
    """WGAN-GP Generator"""
    
    def __init__(self, latent_dim: int, config: Any):
        super().__init__(latent_dim, config)
    
    def _initialize_weights(self):
        """Initialize weights for WGAN-GP"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class TTSGenerator(nn.Module):
    """Time Series Synthesis GAN Generator"""
    
    def __init__(self, latent_dim: int, config: Any):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = config.seq_len
        self.hidden_dim = 256
        
        # LSTM-based generator
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        )
        
        # Initial state projections
        self.h0_projection = nn.Linear(latent_dim, self.hidden_dim)
        self.c0_projection = nn.Linear(latent_dim, self.hidden_dim)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        seq_len = seq_len or self.seq_len
        batch_size = z.size(0)
        
        # Create input sequence
        z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Initialize hidden states
        h0 = self.h0_projection(z).unsqueeze(0).repeat(3, 1, 1)
        c0 = self.c0_projection(z).unsqueeze(0).repeat(3, 1, 1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(z_expanded, (h0, c0))
        
        # Project to output
        output = self.output_projection(lstm_out)
        
        return output

class TTSWGANGPGenerator(TTSGenerator):
    """TTS-WGAN-GP Generator"""
    
    def __init__(self, latent_dim: int, config: Any):
        super().__init__(latent_dim, config)
        
        # Use spectral normalization for WGAN-GP
        if config.use_spectral_norm:
            self.output_projection = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(self.hidden_dim, 128)),
                nn.ReLU(),
                nn.utils.spectral_norm(nn.Linear(128, 1)),
                nn.Tanh()
            )

class SigWGANGenerator(nn.Module):
    """Signature WGAN Generator"""
    
    def __init__(self, latent_dim: int, config: Any):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = config.seq_len
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.seq_len),
        )
        
        # Signature parameters
        self.signature_depth = 3
    
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        seq_len = seq_len or self.seq_len
        batch_size = z.size(0)
        
        output = self.model(z)
        output = output.view(batch_size, seq_len, 1)
        
        return output

class SigCWGANGenerator(SigWGANGenerator):
    """Conditional Signature WGAN Generator"""
    
    def __init__(self, latent_dim: int, config: Any):
        super().__init__(latent_dim, config)
        
        # Conditional input
        self.condition_projection = nn.Linear(10, 64)
        
        # Update model
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.seq_len),
        )
    
    def forward(self, z: torch.Tensor, condition: Optional[torch.Tensor] = None,
                seq_len: Optional[int] = None) -> torch.Tensor:
        seq_len = seq_len or self.seq_len
        batch_size = z.size(0)
        
        # Process condition
        if condition is None:
            condition = torch.randn(batch_size, 10, device=z.device)
        
        condition_proj = self.condition_projection(condition)
        
        # Concatenate
        z_cond = torch.cat([z, condition_proj], dim=-1)
        
        # Generate
        output = self.model(z_cond)
        output = output.view(batch_size, seq_len, 1)
        
        return output

# ===================== BASELINE DISCRIMINATORS =====================

class VanillaDiscriminator(nn.Module):
    """Vanilla GAN Discriminator"""
    
    def __init__(self, config: Any):
        super().__init__()
        self.seq_len = config.seq_len
        
        self.model = nn.Sequential(
            nn.Linear(self.seq_len, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Flatten
        x_flat = x.view(batch_size, -1)
        
        validity = self.model(x_flat)
        
        return validity

class WGANDiscriminator(nn.Module):
    """WGAN Critic"""
    
    def __init__(self, config: Any):
        super().__init__()
        self.seq_len = config.seq_len
        
        self.model = nn.Sequential(
            nn.Linear(self.seq_len, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for WGAN stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x_flat = x.view(batch_size, -1)
        validity = self.model(x_flat)
        
        return validity

class WGANGPDiscriminator(WGANDiscriminator):
    """WGAN-GP Critic"""
    
    def __init__(self, config: Any):
        super().__init__(config)
    
    def _initialize_weights(self):
        """Initialize weights for WGAN-GP"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class TTSDiscriminator(nn.Module):
    """TTS GAN Discriminator"""
    
    def __init__(self, config: Any):
        super().__init__()
        self.hidden_dim = 256
        
        # LSTM-based discriminator
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Classify
        validity = self.output_layers(last_hidden)
        
        return validity

class TTSWGANGPDiscriminator(TTSDiscriminator):
    """TTS-WGAN-GP Critic"""
    
    def __init__(self, config: Any):
        super().__init__(config)
        
        # Linear output for WGAN-GP
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
        
        # Spectral normalization
        if config.use_spectral_norm:
            self.output_layers = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(self.hidden_dim, 128)),
                nn.LeakyReLU(0.2),
                nn.utils.spectral_norm(nn.Linear(128, 1))
            )

class SigWGANDiscriminator(nn.Module):
    """Signature WGAN Discriminator"""
    
    def __init__(self, config: Any):
        super().__init__()
        
        # Signature transform
        self.signature_transform = UnivariateSignatureTransform(depth=3)
        
        # Discriminator on signature features
        self.model = nn.Sequential(
            nn.Linear(8, 512),  # Approximate signature dimension
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute signature
        signature = self.signature_transform(x)
        
        # Discriminate
        validity = self.model(signature)
        
        return validity

class SigCWGANDiscriminator(nn.Module):
    """Conditional Signature WGAN Discriminator"""
    
    def __init__(self, config: Any):
        super().__init__()
        
        # Signature transform
        self.signature_transform = UnivariateSignatureTransform(depth=3)
        
        # Condition projection
        self.condition_projection = nn.Linear(10, 64)
        
        # Discriminator
        self.model = nn.Sequential(
            nn.Linear(8 + 64, 512),  # Signature + condition
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Compute signature
        signature = self.signature_transform(x)
        
        # Process condition
        if condition is None:
            condition = torch.randn(batch_size, 10, device=x.device)
        
        condition_proj = self.condition_projection(condition)
        
        # Combine
        combined = torch.cat([signature, condition_proj], dim=-1)
        
        # Discriminate
        validity = self.model(combined)
        
        return validity

# ===================== SIMPLE TRAINING WRAPPERS =====================

class SimpleGAN:
    """Simple GAN wrapper"""
    
    def __init__(self, config: Any):
        self.config = config
        self.device = config.device
        
        # Models
        self.generator = VanillaGenerator(
            latent_dim=config.latent_dim,
            config=config
        ).to(self.device)
        
        self.discriminator = VanillaDiscriminator(
            config=config
        ).to(self.device)
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.generator_lr,
            betas=(0.5, 0.999)
        )
        
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.discriminator_lr,
            betas=(0.5, 0.999)
        )
        
        # Loss
        self.loss = nn.BCELoss()
    
    def train_step(self, real_batch: torch.Tensor) -> Tuple[float, float]:
        real_batch = real_batch.to(self.device)
        batch_size = real_batch.size(0)
        
        # Labels
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # Train Discriminator
        self.optimizer_D.zero_grad()
        
        # Real samples
        real_validity = self.discriminator(real_batch)
        d_real_loss = self.loss(real_validity, real_labels)
        
        # Fake samples
        z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        fake_batch = self.generator(z)
        fake_validity = self.discriminator(fake_batch.detach())
        d_fake_loss = self.loss(fake_validity, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.optimizer_D.step()
        
        # Train Generator
        self.optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        fake_batch = self.generator(z)
        fake_validity = self.discriminator(fake_batch)
        
        g_loss = self.loss(fake_validity, real_labels)
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item(), d_loss.item()
    
    def generate(self, n_samples: int) -> torch.Tensor:
        """Generate samples"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.latent_dim).to(self.device)
            samples = self.generator(z)
        return samples.cpu()

class WGAN:
    """WGAN wrapper"""
    
    def __init__(self, config: Any):
        self.config = config
        self.device = config.device
        
        # Set defaults for missing attributes
        if not hasattr(config, 'clip_value'):
            config.clip_value = 0.01
        if not hasattr(config, 'n_critic'):
            config.n_critic = 5
        
        # Models
        self.generator = WGANGenerator(
            latent_dim=config.latent_dim,
            config=config
        ).to(self.device)
        
        self.discriminator = WGANDiscriminator(
            config=config
        ).to(self.device)
        
        # Optimizers
        self.optimizer_G = torch.optim.RMSprop(
            self.generator.parameters(),
            lr=config.generator_lr
        )
        
        self.optimizer_D = torch.optim.RMSprop(
            self.discriminator.parameters(),
            lr=config.discriminator_lr
        )
        
        # Gradient clipping
        self.clip_value = config.clip_value
    
    def train_step(self, real_batch: torch.Tensor) -> Tuple[float, float]:
        real_batch = real_batch.to(self.device)
        batch_size = real_batch.size(0)
        
        # Train Critic
        for _ in range(self.config.n_critic):
            self.optimizer_D.zero_grad()
            
            # Real samples
            real_validity = self.discriminator(real_batch)
            
            # Fake samples
            z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
            fake_batch = self.generator(z)
            fake_validity = self.discriminator(fake_batch.detach())
            
            # WGAN loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            d_loss.backward()
            self.optimizer_D.step()
            
            # Clip weights
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)
        
        # Train Generator
        self.optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        fake_batch = self.generator(z)
        fake_validity = self.discriminator(fake_batch)
        
        g_loss = -torch.mean(fake_validity)
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item(), d_loss.item()
    
    def generate(self, n_samples: int) -> torch.Tensor:
        """Generate samples"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.latent_dim).to(self.device)
            samples = self.generator(z)
        return samples.cpu()

"""
FIXED: Ensure baseline models have callable train_step methods
"""

# In the create_baseline_model function at the end of the file:

def create_baseline_model(model_type: str, config: Any):
    """Factory function to create baseline GAN models - FIXED VERSION"""
    
    if model_type == "vanilla_gan":
        gan = SimpleGAN(config)
        # Ensure the train_step is properly bound
        if hasattr(gan, 'train_step'):
            return gan
        else:
            # Create a wrapper with proper method
            return _wrap_baseline_gan(gan, config)
    
    elif model_type == "wgan":
        gan = WGAN(config)
        if hasattr(gan, 'train_step'):
            return gan
        else:
            return _wrap_baseline_gan(gan, config)
    
    elif model_type == "wgan_gp":
        from gan_framework_univariate import AdvancedUnivariateGAN
        return AdvancedUnivariateGAN("wgan_gp", config)
    
    elif model_type == "tts_gan":
        from gan_framework_univariate import AdvancedUnivariateGAN
        return AdvancedUnivariateGAN("tts_gan", config)
    
    elif model_type == "tts_wgan_gp":
        from gan_framework_univariate import AdvancedUnivariateGAN
        return AdvancedUnivariateGAN("tts_wgan_gp", config)
    
    elif model_type == "sig_wgan":
        from gan_framework_univariate import AdvancedUnivariateGAN
        return AdvancedUnivariateGAN("sig_wgan", config)
    
    elif model_type == "sig_cwgan":
        from gan_framework_univariate import AdvancedUnivariateGAN
        return AdvancedUnivariateGAN("sig_cwgan", config)
    
    elif model_type == "bifurcation_gan":
        from gan_framework_univariate import AdvancedUnivariateGAN
        return AdvancedUnivariateGAN("bifurcation_gan", config)
    
    elif model_type == "oscillatory_bifurcation_gan":
        from gan_framework_univariate import AdvancedUnivariateGAN
        return AdvancedUnivariateGAN("oscillatory_bifurcation_gan", config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def _wrap_baseline_gan(gan_instance, config):
    """Wrap a baseline GAN instance to ensure proper method access"""
    class WrappedGAN:
        def __init__(self, gan_instance):
            self.gan = gan_instance
            self.generator = gan_instance.generator
            self.discriminator = gan_instance.discriminator
            
            # Copy other attributes if they exist
            for attr in ['optimizer_G', 'optimizer_D', 'criterion', 'clip_value']:
                if hasattr(gan_instance, attr):
                    setattr(self, attr, getattr(gan_instance, attr))
        
        def train_step(self, real_batch):
            # Call the original train_step method
            return self.gan.train_step(real_batch)
        
        def generate(self, n_samples):
            if hasattr(self.gan, 'generate'):
                return self.gan.generate(n_samples)
            else:
                # Fallback generation
                self.generator.eval()
                with torch.no_grad():
                    z = torch.randn(n_samples, config.latent_dim).to(config.device)
                    samples = self.generator(z)
                return samples.cpu()
    
    return WrappedGAN(gan_instance)
# def create_baseline_model(model_type: str, config: Any):
#     """Factory function to create baseline GAN models"""
    
#     if model_type == "vanilla_gan":
#         return SimpleGAN(config)
    
#     elif model_type == "wgan":
#         return WGAN(config)

    
#     elif model_type == "wgan_gp":
#         from gan_framework_univariate import AdvancedUnivariateGAN
#         return AdvancedUnivariateGAN("wgan_gp", config)
    
#     elif model_type == "tts_gan":
#         from gan_framework_univariate import AdvancedUnivariateGAN
#         return AdvancedUnivariateGAN("tts_gan", config)
    
#     elif model_type == "tts_wgan_gp":
#         from gan_framework_univariate import AdvancedUnivariateGAN
#         return AdvancedUnivariateGAN("tts_wgan_gp", config)
    
#     elif model_type == "sig_wgan":
#         from gan_framework_univariate import AdvancedUnivariateGAN
#         return AdvancedUnivariateGAN("sig_wgan", config)
    
#     elif model_type == "sig_cwgan":
#         from gan_framework_univariate import AdvancedUnivariateGAN
#         return AdvancedUnivariateGAN("sig_cwgan", config)
    
#     elif model_type == "bifurcation_gan":
#         from gan_framework_univariate import AdvancedUnivariateGAN
#         return AdvancedUnivariateGAN("bifurcation_gan", config)
    
#     elif model_type == "oscillatory_bifurcation_gan":
#         from gan_framework_univariate import AdvancedUnivariateGAN
#         return AdvancedUnivariateGAN("oscillatory_bifurcation_gan", config)
    
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")

# def wrap_baseline_model(gan_model):
#     """Wrap a baseline model to ensure train_step returns a dict"""
#     if not hasattr(gan_model, 'train_step'):
#         return gan_model
    
#     original_train_step = gan_model.train_step
    
#     def wrapped_train_step(real_batch):
#         result = original_train_step(real_batch)
#         if isinstance(result, tuple) and len(result) >= 2:
#             return {'g_loss': result[0], 'd_loss': result[1]}
#         elif isinstance(result, dict):
#             return result
#         else:
#             print(f"  Warning: train_step returned {type(result)}, using defaults")
#             return {'g_loss': 0.0, 'd_loss': 0.0}
    
#     gan_model.train_step = wrapped_train_step
#     return gan_model