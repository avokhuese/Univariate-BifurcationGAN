import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from config_univariate import config

# ===================== BIFURCATION DYNAMICS LAYERS =====================

class UnivariateBifurcationLayer(nn.Module):
    """Bifurcation dynamics layer for univariate time series - FIXED VERSION"""
    
    def __init__(self, hidden_dim: int, bifurcation_type: str = "hopf"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bifurcation_type = bifurcation_type
        
        # Learnable bifurcation parameters
        self.mu = nn.Parameter(torch.tensor(config.hopf_mu))
        self.alpha = nn.Parameter(torch.tensor(config.hopf_alpha))
        self.beta = nn.Parameter(torch.tensor(config.hopf_beta))
        self.omega = nn.Parameter(torch.tensor(config.hopf_omega))
        
        # State transformation layers - FIXED: handle any hidden_dim
        if hidden_dim >= 2:
            # If we have enough dimensions, use normal transform
            self.state_transform = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.Tanh(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        else:
            # For small hidden_dim, use simpler transform
            self.state_transform = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.Tanh(),
                nn.Linear(hidden_dim * 2, max(2, hidden_dim))
            )
        
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply bifurcation dynamics to univariate sequence - FIXED VERSION
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            t: Time tensor of shape (batch, seq_len, 1) or None
            
        Returns:
            Transformed tensor with bifurcation dynamics
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Apply state transformation
        x_transformed = self.state_transform(x)
        
        # Apply bifurcation dynamics
        if self.bifurcation_type == "hopf":
            output = self._hopf_dynamics(x_transformed, t, hidden_dim)
        elif self.bifurcation_type == "pitchfork":
            output = self._pitchfork_dynamics(x_transformed, t)
        elif self.bifurcation_type == "saddle_node":
            output = self._saddle_node_dynamics(x_transformed, t)
        elif self.bifurcation_type == "transcritical":
            output = self._transcritical_dynamics(x_transformed, t)
        else:
            raise ValueError(f"Unknown bifurcation type: {self.bifurcation_type}")
        
        # Residual connection
        output = x + output * 0.1
        
        return output
    
    def _hopf_dynamics(self, x: torch.Tensor, t: Optional[torch.Tensor], hidden_dim: int) -> torch.Tensor:
        """Hopf bifurcation dynamics - FIXED for any hidden_dim"""
        batch_size, seq_len, _ = x.shape
        
        # For small hidden_dim, use simpler dynamics
        if hidden_dim < 2:
            # Simple oscillatory dynamics
            if t is None:
                t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
                t = t.view(1, seq_len, 1).repeat(batch_size, 1, 1)
            
            # Create phase
            phase = 2 * np.pi * self.omega * t / seq_len
            
            # Simple oscillation
            output = torch.sin(phase + x) * self.alpha
            
            # Scale to match input dimension
            if output.shape[-1] != hidden_dim:
                output = output[..., :hidden_dim]
            
            return output
        
        # Split into amplitude and phase components for higher dimensions
        # Use first two dimensions for amplitude/phase representation
        x_real = x[..., 0:1]  # Real part
        x_imag = x[..., 1:2]  # Imaginary part
        
        # Compute amplitude and phase
        amplitude = torch.sqrt(x_real**2 + x_imag**2 + 1e-8)
        
        # Hopf normal form
        d_amplitude = self.mu * amplitude - amplitude**3
        d_phase = self.omega
        
        # Update
        new_amplitude = amplitude + d_amplitude * self.alpha
        
        # Create new phase (simple rotation)
        if t is None:
            phase_increment = d_phase * self.beta / seq_len
            # Create a simple phase progression
            new_phase = torch.atan2(x_imag, x_real) + phase_increment
        else:
            new_phase = torch.atan2(x_imag, x_real) + d_phase * t * self.beta
        
        # Convert back
        output_real = new_amplitude * torch.cos(new_phase)
        output_imag = new_amplitude * torch.sin(new_phase)
        
        # Combine and ensure correct dimension
        if hidden_dim >= 2:
            output = torch.cat([output_real, output_imag], dim=-1)
            # Pad if needed
            if hidden_dim > 2:
                padding = x[..., 2:]
                output = torch.cat([output, padding], dim=-1)
        else:
            output = output_real
        
        # Trim to original dimension
        output = output[..., :hidden_dim]
        
        return output
    
    def _pitchfork_dynamics(self, x: torch.Tensor, t: Optional[torch.Tensor]) -> torch.Tensor:
        """Pitchfork bifurcation dynamics"""
        # Pitchfork normal form: dx/dt = μx - x³
        dx = self.mu * x - x**3
        dx = dx + self.alpha * torch.tanh(x)
        return dx * self.beta
    
    def _saddle_node_dynamics(self, x: torch.Tensor, t: Optional[torch.Tensor]) -> torch.Tensor:
        """Saddle-node bifurcation dynamics"""
        # Saddle-node normal form: dx/dt = μ - x²
        dx = self.mu - x**2
        dx = dx - self.alpha * x
        return dx * self.beta
    
    def _transcritical_dynamics(self, x: torch.Tensor, t: Optional[torch.Tensor]) -> torch.Tensor:
        """Transcritical bifurcation dynamics"""
        # Transcritical normal form: dx/dt = μx - x²
        dx = self.mu * x - x**2
        dx = dx - self.alpha * x**3
        return dx * self.beta

class UnivariateOscillatorLayer(nn.Module):
    """Coupled oscillator layer for univariate time series"""
    
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.n_oscillators = config.n_oscillators
        
        # Natural frequencies
        self.natural_freqs = nn.Parameter(
            torch.tensor(config.natural_frequencies[:self.n_oscillators], dtype=torch.float32)
        )
        
        # Coupling matrix
        self.coupling_matrix = nn.Parameter(
            torch.randn(self.n_oscillators, self.n_oscillators) * config.oscillator_coupling_strength
        )
        
        # Projections
        self.to_oscillator = nn.Linear(1, self.n_oscillators * 2)
        self.from_oscillator = nn.Linear(self.n_oscillators * 2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply coupled oscillator dynamics"""
        batch_size, seq_len, _ = x.shape
        
        # Project to oscillator space
        oscillator_state = self.to_oscillator(x)  # (batch, seq_len, n_oscillators*2)
        
        # Split into phase and amplitude
        phase = oscillator_state[:, :, :self.n_oscillators]
        amplitude = oscillator_state[:, :, self.n_oscillators:]
        
        # Apply Kuramoto-like dynamics
        for i in range(1, seq_len):
            # Phase coupling
            phase_diff = phase[:, i-1:i, :] - phase[:, i-1:i, :].transpose(1, 2)
            coupling = torch.matmul(torch.sin(phase_diff), self.coupling_matrix)
            
            dphase = self.natural_freqs + coupling.squeeze(1)
            
            if self.config.use_phase_noise:
                dphase = dphase + torch.randn_like(dphase) * self.config.phase_noise_std
            
            # Amplitude dynamics
            damp = -self.config.amplitude_decay * amplitude[:, i-1:i, :]
            damp = damp - amplitude[:, i-1:i, :]**3 * 0.1
            
            # Update
            phase[:, i:i+1, :] = phase[:, i-1:i, :] + dphase.unsqueeze(1) * 0.1
            amplitude[:, i:i+1, :] = amplitude[:, i-1:i, :] + damp * 0.1
            
            # Saturate amplitude
            amplitude[:, i:i+1, :] = torch.tanh(amplitude[:, i:i+1, :]) * self.config.amplitude_saturation
        
        # Combine and project back
        oscillator_output = torch.cat([phase, amplitude], dim=-1)
        output = self.from_oscillator(oscillator_output)
        
        # Residual connection
        return x + output * 0.1

# ===================== GENERATORS =====================

class BifurcationGenerator(nn.Module):
    """BifurcationGAN Generator for univariate time series - FIXED VERSION"""
    
    def __init__(self, latent_dim: int, config: Any):
        super().__init__()
        self.latent_dim = latent_dim
        self.config = config
        self.hidden_dim = config.generator_hidden
        self.seq_len = config.seq_len
        
        # Noise processing - FIXED: ensure proper dimension handling
        self.noise_processor = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim * 4),
            nn.LayerNorm(self.hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Positional encoding
        self.register_buffer('positional_encoding', None)
        
        # Temporal layers - FIXED: use proper conv1d dimensions
        self.temporal_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.InstanceNorm1d(self.hidden_dim),
                nn.LeakyReLU(0.2)
            ) for _ in range(3)
        ])
        
        # Bifurcation layers - FIXED: ensure hidden_dim >= 2 for bifurcation layers
        self.bifurcation_layers = nn.ModuleList([
            UnivariateBifurcationLayer(max(2, self.hidden_dim), config.bifurcation_type)
            for _ in range(min(config.n_scales, 2))
        ])
        
        # Projection to handle dimension changes
        if self.hidden_dim < 2:
            self.dim_projection = nn.Linear(self.hidden_dim, 2)
            self.dim_projection_back = nn.Linear(2, self.hidden_dim)
        else:
            self.dim_projection = nn.Identity()
            self.dim_projection_back = nn.Identity()
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _get_positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate positional encoding"""
        if self.positional_encoding is None or self.positional_encoding.size(1) != seq_len:
            position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.hidden_dim, 2, dtype=torch.float32, device=device) * 
                -(np.log(10000.0) / self.hidden_dim)
            )
            
            pe = torch.zeros(seq_len, self.hidden_dim, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.positional_encoding = pe.unsqueeze(0)
        
        return self.positional_encoding
    
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Generate univariate time series with stable outputs"""
        batch_size = z.size(0)
        seq_len = seq_len or self.seq_len
        
        # Process noise
        h = self.noise_processor(z)
        
        # Expand and add positional encoding
        h = h.unsqueeze(1).repeat(1, seq_len, 1)
        pe = self._get_positional_encoding(seq_len, z.device)
        h = h + pe
        
        # Apply transformations
        h_proj = self.dim_projection(h)
        h_proj_t = h_proj.transpose(1, 2)
        
        for layer in self.temporal_layers:
            h_proj_t = h_proj_t + layer(h_proj_t)
        
        h_proj = h_proj_t.transpose(1, 2)
        
        # Apply bifurcation dynamics
        t = torch.arange(seq_len, dtype=torch.float32, device=z.device).view(1, seq_len, 1)
        for bif_layer in self.bifurcation_layers:
            h_proj = bif_layer(h_proj, t)
        
        h = self.dim_projection_back(h_proj)
        
        # Generate output with clamping for stability
        output = self.output_projection(h)
        
        # Ensure finite outputs
        output = torch.clamp(output, -10.0, 10.0)  # Prevent extreme values
        
        return output

class OscillatoryBifurcationGenerator(nn.Module):
    """Oscillatory BifurcationGAN Generator"""
    
    def __init__(self, latent_dim: int, config: Any):
        super().__init__()
        self.latent_dim = latent_dim
        self.config = config
        
        # Base generator
        self.base_generator = BifurcationGenerator(latent_dim, config)
        
        # Oscillator layer
        self.oscillator_layer = UnivariateOscillatorLayer(config)
        
        # Oscillation modulation
        self.oscillation_modulation = nn.Sequential(
            nn.Linear(1, config.generator_hidden // 2),
            nn.Tanh(),
            nn.Linear(config.generator_hidden // 2, 1)
        )
        
    def forward(self, z: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Generate with oscillatory bifurcation dynamics"""
        # Generate base signal
        base_signal = self.base_generator(z, seq_len)
        
        # Apply oscillator dynamics
        oscillator_output = self.oscillator_layer(base_signal)
        
        # Add oscillation modulation
        time = torch.arange(base_signal.size(1), device=z.device, dtype=torch.float32)
        time = time.view(1, -1, 1).repeat(base_signal.size(0), 1, 1)
        
        # Create frequency modulation
        frequencies = torch.tensor(self.config.natural_frequencies[:self.config.n_oscillators], 
                                  device=z.device)
        oscillation = torch.zeros_like(base_signal)
        
        for i, freq in enumerate(frequencies):
            if i < base_signal.size(-1):  # Ensure we don't exceed dimensions
                oscillation_component = torch.sin(2 * np.pi * freq * time / base_signal.size(1))
                oscillation = oscillation + oscillation_component * 0.1
        
        # Modulate oscillation
        modulated_oscillation = self.oscillation_modulation(oscillation)
        
        # Combine
        output = oscillator_output + modulated_oscillation * 0.5
        
        return output

# ===================== DISCRIMINATORS =====================

class BifurcationDiscriminator(nn.Module):
    """Discriminator for BifurcationGAN"""
    
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.hidden_dim = config.discriminator_hidden
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, self.hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Temporal analysis
        self.temporal_analysis = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        if config.gradient_penalty_type == "wgan-gp":
            # Remove final activation for WGAN-GP
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(self.hidden_dim, 1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate real vs generated"""
        batch_size, seq_len, _ = x.shape
        
        # Feature extraction
        x_conv = x.transpose(1, 2)  # (batch, 1, seq_len)
        features_conv = self.feature_extractor(x_conv).squeeze(-1)  # (batch, hidden_dim)
        
        # Temporal analysis
        lstm_out, _ = self.temporal_analysis(x)  # (batch, seq_len, hidden_dim)
        features_lstm = lstm_out[:, -1, :]  # Last hidden state
        
        # Combine features
        combined = torch.cat([features_conv, features_lstm], dim=-1)
        
        # Classify
        validity = self.classifier(combined)
        
        return validity

class OscillatoryBifurcationDiscriminator(BifurcationDiscriminator):
    """Discriminator for Oscillatory BifurcationGAN"""
    
    def __init__(self, config: Any):
        super().__init__(config)
        
        # Additional oscillation analysis
        self.oscillation_analyzer = nn.Sequential(
            nn.Conv1d(1, self.hidden_dim // 4, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.hidden_dim // 4, self.hidden_dim // 2, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Update classifier input dimension
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate with oscillation analysis"""
        batch_size, seq_len, _ = x.shape
        
        # Get base features
        features_conv, features_lstm = self._extract_base_features(x)
        
        # Analyze oscillations
        x_conv = x.transpose(1, 2)
        oscillation_features = self.oscillation_analyzer(x_conv).squeeze(-1)
        
        # Combine all features
        combined = torch.cat([features_conv, features_lstm, oscillation_features], dim=-1)
        
        # Classify
        validity = self.classifier(combined)
        
        return validity
    
    def _extract_base_features(self, x: torch.Tensor):
        """Extract base features from parent class"""
        x_conv = x.transpose(1, 2)
        features_conv = self.feature_extractor(x_conv).squeeze(-1)
        
        lstm_out, _ = self.temporal_analysis(x)
        features_lstm = lstm_out[:, -1, :]
        
        return features_conv, features_lstm

# ===================== MODEL FACTORY =====================

def create_model(model_type: str, latent_dim: int, config: Any) -> Tuple[nn.Module, nn.Module]:
    """Factory function to create generator and discriminator"""
    
    if model_type == "bifurcation_gan":
        generator = BifurcationGenerator(latent_dim, config)
        discriminator = BifurcationDiscriminator(config)
    
    elif model_type == "oscillatory_bifurcation_gan":
        generator = OscillatoryBifurcationGenerator(latent_dim, config)
        discriminator = OscillatoryBifurcationDiscriminator(config)
    
    elif model_type == "vanilla_gan":
        from baseline_models_univariate import VanillaGenerator, VanillaDiscriminator
        generator = VanillaGenerator(latent_dim, config)
        discriminator = VanillaDiscriminator(config)
    
    elif model_type == "wgan":
        from baseline_models_univariate import WGANGenerator, WGANDiscriminator
        generator = WGANGenerator(latent_dim, config)
        discriminator = WGANDiscriminator(config)
    
    elif model_type == "wgan_gp":
        from baseline_models_univariate import WGANGPGenerator, WGANGPDiscriminator
        generator = WGANGPGenerator(latent_dim, config)
        discriminator = WGANGPDiscriminator(config)
    
    elif model_type == "tts_gan":
        from baseline_models_univariate import TTSGenerator, TTSDiscriminator
        generator = TTSGenerator(latent_dim, config)
        discriminator = TTSDiscriminator(config)
    
    elif model_type == "tts_wgan_gp":
        from baseline_models_univariate import TTSWGANGPGenerator, TTSWGANGPDiscriminator
        generator = TTSWGANGPGenerator(latent_dim, config)
        discriminator = TTSWGANGPDiscriminator(config)
    
    elif model_type == "sig_wgan":
        from baseline_models_univariate import SigWGANGenerator, SigWGANDiscriminator
        generator = SigWGANGenerator(latent_dim, config)
        discriminator = SigWGANDiscriminator(config)
    
    elif model_type == "sig_cwgan":
        from baseline_models_univariate import SigCWGANGenerator, SigCWGANDiscriminator
        generator = SigCWGANGenerator(latent_dim, config)
        discriminator = SigCWGANDiscriminator(config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return generator, discriminator

def init_weights(m):
    """Initialize model weights"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)