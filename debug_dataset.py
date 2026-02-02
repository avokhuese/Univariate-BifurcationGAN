"""
Debug utilities for univariate time series dataset analysis
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')
from data_loader_univariate_fixed import load_datasets_for_pipeline, analyze_fixed_dataset_shapes
#from data_loader_univariate_fixed import load_and_reshape_aeon_datasets_robust, analyze_fixed_dataset_shapes
from config_univariate import config

# Set plotting style
#plt.style.use('whitegrid')
sns.set_style("whitegrid")
sns.set_palette("husl")

class DatasetDebugger:
    """Debug and analyze time series datasets"""
    
    def __init__(self, config):
        self.config = config
        self.datasets = {}
        
    def load_and_analyze_all(self):
        """Load all datasets and perform comprehensive analysis"""
        print("=" * 80)
        print("DATASET DEBUGGER - COMPREHENSIVE ANALYSIS")
        print("=" * 80)
        
        # Load datasets
        #self.datasets = load_and_reshape_aeon_datasets(self.config)
        self.datasets = load_datasets_for_pipeline(self.config)

        # Analyze shapes
        summary = analyze_fixed_dataset_shapes(self.datasets)
        
        # Perform detailed analysis
        self._analyze_statistics()
        self._visualize_samples()
        self._analyze_temporal_properties()
        self._check_normalization()
        
        return self.datasets, summary
    
    def _analyze_statistics(self):
        """Analyze statistical properties of each dataset"""
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS")
        print("=" * 80)
        
        stats_data = []
        
        for name, info in self.datasets.items():
            if 'reshaped_data' in info:
                data = info['reshaped_data']
                
                # Calculate statistics
                mean_val = np.mean(data)
                std_val = np.std(data)
                min_val = np.min(data)
                max_val = np.max(data)
                skewness = pd.Series(data.flatten()).skew()
                kurtosis = pd.Series(data.flatten()).kurtosis()
                
                # Dynamic range
                dynamic_range = max_val - min_val
                
                # Stationarity (ADF test approximation)
                try:
                    from statsmodels.tsa.stattools import adfuller
                    p_value = adfuller(data.flatten())[1]
                    is_stationary = p_value < 0.05
                except:
                    p_value = np.nan
                    is_stationary = False
                
                stats_data.append({
                    'Dataset': name,
                    'Samples': data.shape[0],
                    'Mean': f"{mean_val:.3f}",
                    'Std': f"{std_val:.3f}",
                    'Min': f"{min_val:.3f}",
                    'Max': f"{max_val:.3f}",
                    'Range': f"{dynamic_range:.3f}",
                    'Skew': f"{skewness:.3f}",
                    'Kurtosis': f"{kurtosis:.3f}",
                    'Stationary': 'Yes' if is_stationary else 'No'
                })
        
        # Create and display DataFrame
        if stats_data:
            df_stats = pd.DataFrame(stats_data)
            print("\nDetailed Statistics:")
            print(df_stats.to_string(index=False))
            
            # Save to CSV
            csv_path = f"{self.config.results_dir}/dataset_statistics.csv"
            df_stats.to_csv(csv_path, index=False)
            print(f"\nStatistics saved to: {csv_path}")
    
    def _visualize_samples(self, n_datasets: int = 5, n_samples: int = 3):
        """Visualize sample time series from each dataset"""
        print("\n" + "=" * 80)
        print("VISUALIZING DATASET SAMPLES")
        print("=" * 80)
        
        # Select datasets to visualize
        dataset_names = list(self.datasets.keys())[:n_datasets]
        
        # Create figure
        fig, axes = plt.subplots(n_datasets, n_samples, 
                                 figsize=(4 * n_samples, 3 * n_datasets),
                                 squeeze=False)
        
        for i, dataset_name in enumerate(dataset_names):
            info = self.datasets[dataset_name]
            data = info.get('reshaped_data')
            
            if data is not None:
                # Select random samples
                sample_indices = np.random.choice(len(data), n_samples, replace=False)
                
                for j, idx in enumerate(sample_indices):
                    sample = data[idx]
                    ax = axes[i, j]
                    
                    # Plot time series
                    ax.plot(sample, linewidth=1.5)
                    ax.set_title(f"{dataset_name}\nSample {idx}")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Value")
                    
                    # Add statistics
                    stats_text = f"μ={np.mean(sample):.2f}\nσ={np.std(sample):.2f}"
                    ax.text(0.05, 0.95, stats_text,
                            transform=ax.transAxes,
                            fontsize=8,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{self.config.results_dir}/dataset_samples.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def _analyze_temporal_properties(self):
        """Analyze temporal properties like autocorrelation and periodicity"""
        print("\n" + "=" * 80)
        print("TEMPORAL PROPERTY ANALYSIS")
        print("=" * 80)
        
        temporal_data = []
        
        for name, info in self.datasets.items():
            if 'reshaped_data' in info:
                data = info['reshaped_data']
                
                # Analyze a subset of samples
                n_samples_to_analyze = min(10, len(data))
                sample_indices = np.random.choice(len(data), n_samples_to_analyze, replace=False)
                
                autocorrs = []
                periodicity_scores = []
                
                for idx in sample_indices:
                    sample = data[idx]
                    
                    # Calculate autocorrelation
                    autocorr = np.correlate(sample, sample, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    autocorr = autocorr / autocorr[0]
                    
                    # First zero crossing (rough period estimate)
                    zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
                    if len(zero_crossings) > 0:
                        first_zero = zero_crossings[0] if zero_crossings[0] > 0 else 1
                    else:
                        first_zero = len(sample)
                    
                    autocorrs.append(autocorr[1])  # Lag-1 autocorrelation
                    periodicity_scores.append(1.0 / first_zero)
                
                # Average scores
                avg_autocorr = np.mean(autocorrs)
                avg_periodicity = np.mean(periodicity_scores)
                
                temporal_data.append({
                    'Dataset': name,
                    'Lag-1 Autocorr': f"{avg_autocorr:.3f}",
                    'Periodicity': f"{avg_periodicity:.3f}",
                    'Samples Analyzed': n_samples_to_analyze
                })
        
        # Display temporal analysis
        if temporal_data:
            df_temporal = pd.DataFrame(temporal_data)
            print("\nTemporal Properties:")
            print(df_temporal.to_string(index=False))
    
    def _check_normalization(self):
        """Check normalization requirements for each dataset"""
        print("\n" + "=" * 80)
        print("NORMALIZATION CHECK")
        print("=" * 80)
        
        for name, info in self.datasets.items():
            if 'reshaped_data' in info:
                data = info['reshaped_data']
                
                # Check scaling requirements
                mean_val = np.mean(data)
                std_val = np.std(data)
                min_val = np.min(data)
                max_val = np.max(data)
                
                needs_normalization = (
                    abs(mean_val) > 1.0 or 
                    std_val > 2.0 or 
                    abs(min_val) > 3.0 or 
                    abs(max_val) > 3.0
                )
                
                status = "NEEDS NORMALIZATION" if needs_normalization else "OK"
                
                print(f"{name:30s} | "
                      f"Mean: {mean_val:7.3f} | "
                      f"Std: {std_val:7.3f} | "
                      f"Range: [{min_val:7.3f}, {max_val:7.3f}] | "
                      f"{status}")
    
    def validate_data_loader(self, dataset_name: str, batch_size: int = 4):
        """Validate DataLoader output for a specific dataset"""
        print("\n" + "=" * 80)
        print(f"DATA LOADER VALIDATION: {dataset_name}")
        print("=" * 80)
        
        from data_loader_univariate_fixed import prepare_dataset_with_fixed_size, safe_prepare_dataset
        
        if dataset_name not in self.datasets:
            print(f"Dataset '{dataset_name}' not found!")
            return
        
        dataset_info = self.datasets[dataset_name]
        
        try:
            # Prepare dataset
            train_loader, val_loader, test_loader, scaler = safe_prepare_dataset(
                dataset_info, self.config
            )
            
            # Get a batch
            batch = next(iter(train_loader))
            
            print(f"\nDataLoader Validation Successful!")
            print(f"Batch keys: {list(batch.keys())}")
            
            # Check shapes
            data = batch['data']
            print(f"Data shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
            
            # Check for NaN or Inf
            has_nan = torch.isnan(data).any()
            has_inf = torch.isinf(data).any()
            print(f"Has NaN: {has_nan.item()}")
            print(f"Has Inf: {has_inf.item()}")
            
            # Plot batch samples
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            for i in range(min(4, batch_size)):
                sample = data[i].cpu().numpy().flatten()
                axes[i].plot(sample, linewidth=2)
                axes[i].set_title(f"Batch Sample {i}")
                axes[i].set_xlabel("Time")
                axes[i].set_ylabel("Value")
                axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f"DataLoader Output: {dataset_name}", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.config.results_dir}/dataloader_validation_{dataset_name}.png", 
                       dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"DataLoader validation failed: {e}")
            import traceback
            traceback.print_exc()

def analyze_dataset_complexity(dataset_name: str, data: np.ndarray) -> Dict:
    """Analyze dataset complexity for bifurcation GAN"""
    
    n_samples, seq_len = data.shape
    
    complexity_metrics = {
        'dataset': dataset_name,
        'n_samples': n_samples,
        'seq_len': seq_len,
        'average_power': np.mean(data ** 2),
        'entropy_estimate': estimate_entropy(data.flatten()),
        'fractal_dimension': estimate_fractal_dimension(data),
        'nonlinearity_score': estimate_nonlinearity(data),
        'suggested_bifurcation_params': suggest_bifurcation_parameters(data)
    }
    
    return complexity_metrics

def estimate_entropy(data: np.ndarray, bins: int = 50) -> float:
    """Estimate entropy of the data"""
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))
    return entropy

def estimate_fractal_dimension(data: np.ndarray, n_samples: int = 100) -> float:
    """Estimate fractal dimension using box-counting method"""
    try:
        # Sample subset
        if len(data) > n_samples:
            indices = np.random.choice(len(data), n_samples, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data
        
        # Simple box-counting approximation
        scales = np.logspace(0.5, 2, 10)
        counts = []
        
        for scale in scales:
            # Box count at this scale
            scaled = sample_data / scale
            rounded = np.round(scaled).astype(int)
            unique_boxes = len(np.unique(rounded, axis=0))
            counts.append(unique_boxes)
        
        # Fit line in log-log space
        log_scales = np.log(scales)
        log_counts = np.log(counts)
        
        if len(log_scales) > 1 and len(log_counts) > 1:
            coeffs = np.polyfit(log_scales, log_counts, 1)
            fractal_dim = -coeffs[0]
        else:
            fractal_dim = 1.0
            
        return min(max(fractal_dim, 1.0), 2.0)
        
    except:
        return 1.0

def estimate_nonlinearity(data: np.ndarray) -> float:
    """Estimate nonlinearity score"""
    try:
        # Simple proxy: ratio of variance to linear prediction error
        if len(data) < 10:
            return 0.5
        
        # Make linear prediction
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.flatten()
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X[:len(y)], y)
        y_pred = model.predict(X[:len(y)])
        
        # Calculate nonlinearity score
        mse = np.mean((y - y_pred) ** 2)
        var = np.var(y)
        
        if var > 0:
            nonlinearity = mse / var
        else:
            nonlinearity = 0.0
        
        return min(nonlinearity, 1.0)
    except:
        return 0.5

def suggest_bifurcation_parameters(data: np.ndarray) -> Dict:
    """Suggest bifurcation parameters based on data characteristics"""
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    max_val = np.max(np.abs(data))
    
    # Scale parameters based on data statistics
    scale_factor = max_val if max_val > 0 else 1.0
    
    suggestions = {
        'hopf_mu': 0.1 * scale_factor,
        'hopf_omega': 2.0 * (std_val / scale_factor) if scale_factor > 0 else 2.0,
        'hopf_alpha': 0.1,
        'hopf_beta': 1.0,
        'bifurcation_threshold': 0.5 * scale_factor,
        'suggested_noise_std': std_val * 0.1
    }
    
    return suggestions

if __name__ == "__main__":
    """Main debug execution"""
    
    debugger = DatasetDebugger(config)
    
    # Run comprehensive analysis
    datasets, summary = debugger.load_and_analyze_all()
    
    # Validate DataLoader for a specific dataset
    if datasets:
        sample_dataset = list(datasets.keys())[0]
        debugger.validate_data_loader(sample_dataset)
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)