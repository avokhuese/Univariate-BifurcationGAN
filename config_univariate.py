import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class UnivariateConfig:
    """Configuration for univariate time series augmentation with BifurcationGAN variants"""
    
    # Dataset parameters
    dataset_names: List[str] = None
    max_series_length: int = 1000
    min_series_length: int = 50
    
    # Model architecture
    seq_len: int = 100  # Fixed sequence length for training
    latent_dim: int = 128
    generator_hidden: int = 256
    discriminator_hidden: int = 256
    num_layers: int = 3
    
    # === BIFURCATION GAN PARAMETERS ===
    use_bifurcation: bool = True
    bifurcation_type: str = "hopf"  # "hopf", "pitchfork", "saddle_node", "transcritical"
    
    # Hopf Bifurcation parameters
    hopf_mu: float = 0.1  # Bifurcation parameter
    hopf_omega: float = 2.0  # Oscillation frequency
    hopf_alpha: float = 0.1  # Amplitude coefficient
    hopf_beta: float = 1.0  # Nonlinear coefficient
    bifurcation_threshold: float = 0.5
    
    # === OSCILLATORY BIFURCATION GAN PARAMETERS ===
    use_oscillatory_dynamics: bool = True
    n_oscillators: int = 3
    oscillator_coupling_strength: float = 0.2
    natural_frequencies: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    
    # Phase dynamics
    use_phase_noise: bool = True
    phase_noise_std: float = 0.1
    
    # Amplitude dynamics
    amplitude_decay: float = 0.05
    amplitude_saturation: float = 1.0
    
    # === ADVANCED AUGMENTATION PARAMETERS ===
    use_multiscale: bool = True
    n_scales: int = 3
    scale_factors: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    
    # Temporal coherence
    temporal_coherence_weight: float = 0.1
    spectral_consistency_weight: float = 0.05
    
    # Diversity enhancement
    diversity_weight: float = 0.2
    mode_seeking_weight: float = 0.1
    
    # === ADVANCED TRAINING PARAMETERS ===
    batch_size: int = 64
    epochs: int = 500
    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    critic_iterations: int = 5
    
    # Optimizer
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 1e-5
    
    # Gradient penalties
    lambda_gp: float = 10.0
    lambda_drift: float = 0.001
    gradient_penalty_type: str = "wgan-gp"
    
    # Spectral normalization
    use_spectral_norm: bool = True
    spectral_norm_power: int = 1
    
    # === NOISE PARAMETERS ===
    noise_type: str = "hierarchical"
    noise_layers: int = 3
    noise_scale_range: Tuple[float, float] = (0.1, 2.0)
    
    # Temporal noise correlation
    temporal_correlation: float = 0.8
    
    # === REGULARIZATION PARAMETERS ===
    use_gradient_penalty: bool = True
    use_consistency_regularization: bool = True
    consistency_lambda: float = 10.0
    
    # Feature matching
    use_feature_matching: bool = True
    feature_matching_lambda: float = 0.1
    
    # === EVALUATION METRICS ===
    # Diversity metrics
    calculate_fid: bool = True
    calculate_prd: bool = True
    calculate_mmd: bool = True
    
    # Quality metrics
    calculate_wasserstein: bool = True
    calculate_jsd: bool = True
    calculate_ks_test: bool = True
    
    # Temporal metrics
    calculate_acf_similarity: bool = True
    calculate_psd_similarity: bool = True
    
    # === BENCHMARKING PARAMETERS ===
    benchmark_models: List[str] = field(default_factory=lambda: [
        "vanilla_gan", "wgan", "wgan_gp", "tts_gan", "tts_wgan_gp", 
        "sig_wgan", "sig_cwgan", "bifurcation_gan", "oscillatory_bifurcation_gan"
    ])
    
    n_runs_per_model: int = 5
    confidence_level: float = 0.95
    
    # === EXPERIMENTAL SETTINGS ===
    experiment_name: str = "univariate_bifurcation_gan_experiment"
    use_early_stopping: bool = True
    patience: int = 20
    min_delta: float = 0.001
    
    # Visualization
    visualize_dynamics: bool = True
    plot_phase_portraits: bool = True
    
    # === RESOURCE MANAGEMENT ===
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 4
    pin_memory: bool = True
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"
    
    # === PATHS ===
    data_dir: str = "./data/univariate"
    save_dir: str = "./saved_models_univariate"
    results_dir: str = "./results_univariate"
    logs_dir: str = "./logs_univariate"
    cache_dir: str = "./cache_univariate"
    
    # === DATASET SPECIFIC PARAMETERS ===
    dataset_params: Dict[str, Any] = field(default_factory=dict)
    
    # === SYNTHETIC DATA GENERATION ===
    n_synthetic_samples: int = 1000
    test_split: float = 0.2
    validation_split: float = 0.1
    
    # Length variation
    synthetic_length_variation: bool = True
    min_synthetic_length: int = 50
    max_synthetic_length: int = 500
    
    # === MODEL SAVING ===
    save_checkpoint_freq: int = 10
    save_best_model: bool = True
    model_save_format: str = "pytorch"
    
    def __post_init__(self):
        """Initialize after dataclass creation"""
        # Default univariate datasets from aeon
        if self.dataset_names is None:
            self.dataset_names = [
                'ECG5000', 'FordB', 'CBF', 'ScreenType', 'StrawBerry',
                'Yoga', 'EOGHorizonSignal', 'Fungi', 'GestureMidAirD1',
                'InsectEPGRegularTrain', 'MelbournePedestrian', 'PigCVP',
                'PowerCons', 'SemgHandMovement', 'GunPointAgeSpan'
            ]
        
        # Initialize dataset parameters
        self.dataset_params = {
            'ECG5000': {'avg_length': 140, 'n_classes': 5},
            'FordB': {'avg_length': 500, 'n_classes': 2},
            'CBF': {'avg_length': 128, 'n_classes': 3},
            'ScreenType': {'avg_length': 720, 'n_classes': 3},
            'StrawBerry': {'avg_length': 235, 'n_classes': 2},
            'Yoga': {'avg_length': 426, 'n_classes': 2},
            'EOGHorizonSignal': {'avg_length': 1250, 'n_classes': 12},
            'Fungi': {'avg_length': 201, 'n_classes': 18},
            'GestureMidAirD1': {'avg_length': 360, 'n_classes': 13},
            'InsectEPGRegularTrain': {'avg_length': 601, 'n_classes': 3},
            'MelbournePedestrian': {'avg_length': 24, 'n_classes': 10},
            'PigCVP': {'avg_length': 2000, 'n_classes': 52},
            'PowerCons': {'avg_length': 144, 'n_classes': 2},
            'SemgHandMovement': {'avg_length': 1500, 'n_classes': 6},
            'GunPointAgeSpan': {'avg_length': 150, 'n_classes': 2}
        }
        
        # Set natural frequencies
        if len(self.natural_frequencies) < self.n_oscillators:
            base_freq = 0.5
            self.natural_frequencies = [base_freq * (2 ** i) for i in range(self.n_oscillators)]
        
        # Create directories
        import os
        directories = [self.data_dir, self.save_dir, self.results_dir, 
                      self.logs_dir, self.cache_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Validate parameters
        self._validate_parameters()
    # Set num_workers based on platform and available cores
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if self.num_workers is None:
            # Use fewer workers on macOS/Linux to avoid semaphore issues
            self.num_workers = min(2, cpu_count // 2) if cpu_count > 1 else 0
    def _validate_parameters(self):
        """Validate configuration parameters"""
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.generator_hidden > 0, "generator_hidden must be positive"
        assert self.discriminator_hidden > 0, "discriminator_hidden must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0 < self.test_split < 1, "test_split must be between 0 and 1"
        assert 0 < self.validation_split < 1, "validation_split must be between 0 and 1"
        assert self.epochs > 0, "epochs must be positive"
        assert self.n_oscillators > 0, "n_oscillators must be positive"
        assert self.bifurcation_type in ["hopf", "pitchfork", "saddle_node", "transcritical"], \
            f"Invalid bifurcation_type: {self.bifurcation_type}"
    
    def get_dataset_params(self, dataset_name: str) -> Dict[str, Any]:
        """Get parameters for a specific dataset"""
        if dataset_name in self.dataset_params:
            return self.dataset_params[dataset_name]
        else:
            return {'avg_length': self.seq_len, 'n_classes': 2}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        import dataclasses
        return dataclasses.asdict(self)

# Global configuration instance
config = UnivariateConfig()