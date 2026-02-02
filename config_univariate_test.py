"""
Simplified test configuration for univariate time series
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class UnivariateTestConfig:
    """Simplified configuration for testing"""
    
    # Dataset parameters
    dataset_names: List[str] = field(default_factory=lambda: ['TestDataset'])
    
    # Model architecture - SIMPLIFIED FOR TESTING
    seq_len: int = 50  # Shorter for testing
    latent_dim: int = 32  # Smaller for testing
    generator_hidden: int = 64  # Smaller for testing
    discriminator_hidden: int = 64  # Smaller for testing
    num_layers: int = 2  # Fewer layers for testing
    
    # === BIFURCATION GAN PARAMETERS ===
    use_bifurcation: bool = True
    bifurcation_type: str = "hopf"
    
    # Hopf Bifurcation parameters
    hopf_mu: float = 0.1
    hopf_omega: float = 2.0
    hopf_alpha: float = 0.1
    hopf_beta: float = 1.0
    bifurcation_threshold: float = 0.5
    
    # === TRAINING PARAMETERS ===
    batch_size: int = 16  # Smaller for testing
    epochs: int = 50  # Fewer for testing
    generator_lr: float = 1e-4
    discriminator_lr: float = 1e-4
    critic_iterations: int = 2  # Fewer for testing
    
    # Optimizer
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 1e-5
    
    # Gradient penalties
    lambda_gp: float = 10.0
    lambda_drift: float = 0.001
    gradient_penalty_type: str = "wgan-gp"
    
    # === EVALUATION ===
    calculate_fid: bool = False  # Disable for testing
    calculate_prd: bool = False
    calculate_mmd: bool = False
    
    # === BENCHMARKING ===
    benchmark_models: List[str] = field(default_factory=lambda: [
        "vanilla_gan", "bifurcation_gan"
    ])
    
    n_runs_per_model: int = 1  # Single run for testing
    
    # === EXPERIMENTAL SETTINGS ===
    experiment_name: str = "univariate_test"
    use_early_stopping: bool = False  # Disable for testing
    
    # === RESOURCE MANAGEMENT ===
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 0  # Disable multiprocessing for testing
    pin_memory: bool = True
    
    # Mixed precision
    use_amp: bool = False  # Disable for testing
    
    # === PATHS ===
    data_dir: str = "./data/univariate_test"
    save_dir: str = "./saved_models_univariate_test"
    results_dir: str = "./results_univariate_test"
    logs_dir: str = "./logs_univariate_test"
    cache_dir: str = "./cache_univariate_test"
    
    def __post_init__(self):
        """Initialize after dataclass creation"""
        import os
        directories = [self.data_dir, self.save_dir, self.results_dir, 
                      self.logs_dir, self.cache_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_dataset_params(self, dataset_name: str) -> Dict[str, Any]:
        """Get parameters for a specific dataset"""
        return {'avg_length': self.seq_len, 'n_classes': 2}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        import dataclasses
        return dataclasses.asdict(self)

# Test configuration instance
config = UnivariateTestConfig()