"""
Updated data loader with better error handling and synthetic fallback
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from typing import Tuple, List, Dict, Optional, Any
import os
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FixedSizeUnivariateDataset(Dataset):
    """Dataset that ensures fixed size for univariate time series"""
    
    def __init__(self, data: np.ndarray, target_seq_len: int = 100, 
                 config: Optional[Any] = None):
        """
        Args:
            data: numpy array of shape (n_samples, n_timesteps)
            target_seq_len: target sequence length for all samples
            config: configuration object
        """
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim == 3:
            # Try to squeeze, but handle different cases
            if data.shape[-1] == 1:
                data = data.squeeze(2)
            elif data.shape[1] == 1:
                data = data.squeeze(1)
            else:
                # Take first feature for multivariate
                data = data[:, :, 0]
        
        self.target_seq_len = target_seq_len
        self.config = config
        
        # Reshape data to consistent dimensions
        self.data, self.padding_info = self._reshape_to_fixed_size_with_info(data)
        self.n_samples, self.n_timesteps = self.data.shape
        
        print(f"FixedSizeDataset: {self.n_samples} samples, "
              f"{self.n_timesteps} timesteps")
    
    def _reshape_to_fixed_size_with_info(self, data: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Reshape data to fixed dimensions with padding information"""
        n_samples, n_timesteps = data.shape
        
        # Initialize output array
        reshaped_data = np.zeros((n_samples, self.target_seq_len))
        padding_info = []
        
        for i in range(n_samples):
            sample = data[i]
            info = {'was_padded': False, 'was_truncated': False}
            
            # Handle sequence length
            if n_timesteps > self.target_seq_len:
                info['was_truncated'] = True
                
                # For bifurcation analysis, preserve important dynamics
                if self.config and self.config.use_multiscale:
                    # Take central segment
                    start_idx = (n_timesteps - self.target_seq_len) // 2
                    reshaped_sample = sample[start_idx:start_idx + self.target_seq_len]
                    info['truncation_method'] = 'central_segment'
                else:
                    # Take random segment
                    start_idx = np.random.randint(0, n_timesteps - self.target_seq_len)
                    reshaped_sample = sample[start_idx:start_idx + self.target_seq_len]
                    info['truncation_method'] = 'random_segment'
                    
            elif n_timesteps < self.target_seq_len:
                info['was_padded'] = True
                pad_len = self.target_seq_len - n_timesteps
                
                if self.config and self.config.use_phase_noise:
                    # Use reflection padding
                    reshaped_sample = np.pad(sample, (0, pad_len), mode='reflect')
                    info['padding_method'] = 'reflection'
                else:
                    # Use edge padding
                    reshaped_sample = np.pad(sample, (0, pad_len), mode='edge')
                    info['padding_method'] = 'edge'
                    
                info['pad_len'] = pad_len
            else:
                reshaped_sample = sample
            
            reshaped_data[i] = reshaped_sample
            padding_info.append(info)
        
        return reshaped_data, padding_info
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset"""
        sample = self.data[idx]
        info = self.padding_info[idx]
        
        # Add small noise if configured
        if self.config and self.config.use_phase_noise:
            noise_level = self.config.phase_noise_std * np.std(sample)
            sample = sample + np.random.normal(0, noise_level, sample.shape)
        
        # Reshape to (seq_len, 1)
        sample = sample.reshape(-1, 1)
        
        # Create return dictionary
        item = {
            'data': torch.FloatTensor(sample),
            'original_length': torch.tensor(len(self.data[idx])),
            'sample_idx': torch.tensor(idx),
            'was_padded': torch.tensor(float(info['was_padded'])),
            'was_truncated': torch.tensor(float(info['was_truncated']))
        }
        
        return item
    
    def get_padding_stats(self) -> Dict[str, int]:
        """Get statistics about padding/truncation"""
        stats = {
            'n_padded': sum(1 for info in self.padding_info if info['was_padded']),
            'n_truncated': sum(1 for info in self.padding_info if info['was_truncated']),
        }
        return stats

def create_synthetic_dataset(dataset_name: str, n_samples: int = 1000, seq_len: int = 100):
    """Create synthetic datasets when real datasets fail to load"""
    np.random.seed(42)
    
    if dataset_name == "ECG5000":
        # ECG-like signals
        t = np.linspace(0, 10 * np.pi, seq_len)
        data = []
        for _ in range(n_samples):
            # Base ECG signal
            signal = np.sin(t) + 0.5 * np.sin(2*t) + 0.2 * np.sin(3*t)
            
            # Add P wave, QRS complex, T wave
            p_wave = 0.1 * np.exp(-((t - 2) ** 2) / 0.5)
            qrs = 0.3 * np.exp(-((t - 5) ** 2) / 0.2)
            t_wave = 0.15 * np.exp(-((t - 8) ** 2) / 0.8)
            
            signal = signal + p_wave + qrs + t_wave
            
            # Add noise and variations
            signal += np.random.normal(0, 0.05, seq_len)
            signal *= np.random.uniform(0.8, 1.2)
            
            data.append(signal)
    
    elif dataset_name == "FordB":
        # Automotive sensor-like signals
        data = []
        for _ in range(n_samples):
            # Random walk with drift
            signal = np.cumsum(np.random.normal(0, 0.1, seq_len))
            
            # Add periodic components
            t = np.linspace(0, 4*np.pi, seq_len)
            signal += 0.5 * np.sin(t) + 0.2 * np.sin(3*t)
            
            # Add occasional spikes
            if np.random.random() > 0.8:
                spike_pos = np.random.randint(20, seq_len-20)
                signal[spike_pos:spike_pos+5] += np.random.uniform(0.5, 1.5)
            
            data.append(signal)
    
    elif dataset_name == "CBF":
        # Cylinder, Bell, Funnel dataset
        data = []
        for _ in range(n_samples):
            # Randomly choose one of three patterns
            pattern = np.random.choice(['cylinder', 'bell', 'funnel'])
            
            if pattern == 'cylinder':
                # Cylinder shape
                signal = np.ones(seq_len) * np.random.uniform(0.5, 1.5)
                
            elif pattern == 'bell':
                # Bell shape (Gaussian)
                center = np.random.uniform(0.3, 0.7) * seq_len
                width = np.random.uniform(10, 30)
                x = np.arange(seq_len)
                signal = np.exp(-((x - center) ** 2) / (2 * width ** 2))
                signal = signal * np.random.uniform(0.5, 2)
                
            else:  # funnel
                # Funnel shape
                slope = np.random.uniform(0.01, 0.05)
                signal = slope * np.arange(seq_len) + np.random.uniform(0, 0.5)
                noise = np.random.normal(0, 0.05, seq_len)
                signal = signal * (1 + noise)
            
            data.append(signal)
    
    else:
        # Generic oscillatory signal
        data = []
        for _ in range(n_samples):
            t = np.linspace(0, 4*np.pi, seq_len)
            
            # Multiple frequency components
            base_freq = np.random.uniform(0.5, 2.0)
            signal = np.sin(base_freq * t)
            signal += 0.3 * np.sin(3 * base_freq * t)
            signal += 0.1 * np.sin(5 * base_freq * t)
            
            # Add noise
            signal += np.random.normal(0, 0.1, seq_len)
            
            # Random amplitude
            signal *= np.random.uniform(0.5, 1.5)
            
            data.append(signal)
    
    data = np.array(data)
    labels = np.random.randint(0, 3, n_samples)  # 3 classes for testing
    
    return {
        'data': data,
        'labels': labels,
        'n_samples': n_samples,
        'n_timesteps': seq_len,
        'is_synthetic': True
    }

def load_and_reshape_aeon_datasets_robust(config, use_synthetic_fallback: bool = True) -> Dict[str, Dict]:
    """Load multiple datasets from aeon with robust error handling"""
    datasets = {}
    failed_datasets = []
    
    print(f"Loading {len(config.dataset_names)} datasets with robust handler...")
    print("=" * 60)
    
    for dataset_name in tqdm(config.dataset_names, desc="Processing datasets"):
        try:
            # Try to load from aeon
            try:
                from aeon.datasets import load_classification
                X, y = load_classification(dataset_name)
                data_type = "classification"
            except:
                try:
                    from aeon.datasets import load_regression
                    X, y = load_regression(dataset_name)
                    data_type = "regression"
                except Exception as e:
                    if use_synthetic_fallback:
                        print(f"  {dataset_name}: Using synthetic fallback")
                        dataset_info = create_synthetic_dataset(dataset_name, 
                                                               n_samples=500, 
                                                               seq_len=config.seq_len)
                        datasets[dataset_name] = dataset_info
                        continue
                    else:
                        raise e
            
            # Convert to numpy array with better error handling
            X_array = None
            
            if isinstance(X, list):
                # Handle list of arrays with variable lengths
                X_processed = []
                for x in X:
                    # Handle different shapes
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)
                    elif x.ndim == 3:
                        # Try to squeeze
                        if x.shape[-1] == 1:
                            x = x.squeeze(2)
                        elif x.shape[0] == 1:
                            x = x.squeeze(0)
                        else:
                            # Take first channel
                            x = x[:, :, 0]
                    X_processed.append(x)
                
                # Find common length (median for efficiency)
                lengths = [x.shape[0] for x in X_processed]
                target_length = min(config.seq_len, int(np.median(lengths)))
                
                # Process each sample
                X_fixed = []
                for x in X_processed:
                    if x.shape[0] > target_length:
                        # Take central segment
                        start_idx = (x.shape[0] - target_length) // 2
                        x = x[start_idx:start_idx + target_length]
                    elif x.shape[0] < target_length:
                        # Pad with edge values
                        pad_len = target_length - x.shape[0]
                        x = np.pad(x, (0, pad_len), mode='edge')
                    X_fixed.append(x.flatten())
                
                X_array = np.array(X_fixed)
                
            elif isinstance(X, np.ndarray):
                X_array = X
                
                # Handle different array shapes
                if X_array.ndim == 1:
                    X_array = X_array.reshape(1, -1)
                elif X_array.ndim == 2:
                    # Already good shape (n_samples, n_timesteps)
                    pass
                elif X_array.ndim == 3:
                    if X_array.shape[-1] == 1:
                        X_array = X_array.squeeze(2)
                    elif X_array.shape[1] == 1:
                        X_array = X_array.squeeze(1)
                    else:
                        # Take first feature
                        X_array = X_array[:, :, 0]
            
            if X_array is None or X_array.size == 0:
                if use_synthetic_fallback:
                    print(f"  {dataset_name}: Empty data, using synthetic")
                    dataset_info = create_synthetic_dataset(dataset_name, 
                                                           n_samples=500, 
                                                           seq_len=config.seq_len)
                    datasets[dataset_name] = dataset_info
                    continue
                else:
                    raise ValueError(f"Failed to process data for {dataset_name}")
            
            # Create dataset with fixed size
            dataset = FixedSizeUnivariateDataset(
                X_array,
                target_seq_len=config.seq_len,
                config=config
            )
            
            # Get reshaped data
            X_reshaped = dataset.data
            
            # Calculate statistics
            stats = dataset.get_padding_stats()
            
            # Store dataset info
            datasets[dataset_name] = {
                'original_data': X_array,
                'reshaped_data': X_reshaped,
                'dataset_object': dataset,
                'labels': y if y is not None else np.zeros(len(X_reshaped)),
                'n_samples': X_reshaped.shape[0],
                'n_timesteps': X_reshaped.shape[1],
                'n_features': 1,
                'padding_stats': stats,
                'is_synthetic': False
            }
            
            print(f"✓ {dataset_name:25s} | "
                  f"Samples: {X_reshaped.shape[0]:5d} | "
                  f"Length: {X_reshaped.shape[1]:4d}→{config.seq_len:4d}")
            
        except Exception as e:
            failed_datasets.append((dataset_name, str(e)))
            
            if use_synthetic_fallback:
                print(f"⚠ {dataset_name:25s} | Using synthetic fallback")
                dataset_info = create_synthetic_dataset(dataset_name, 
                                                       n_samples=500, 
                                                       seq_len=config.seq_len)
                datasets[dataset_name] = dataset_info
            else:
                print(f"✗ {dataset_name:25s} | Failed: {str(e)[:50]}...")
    
    print("=" * 60)
    
    # If no datasets loaded, create at least one synthetic
    if not datasets and use_synthetic_fallback:
        print("No datasets loaded. Creating synthetic test dataset...")
        dataset_info = create_synthetic_dataset("TestDataset", 
                                               n_samples=500, 
                                               seq_len=config.seq_len)
        datasets["TestDataset"] = dataset_info
    
    # Save to cache
    cache_path = os.path.join(config.cache_dir, "fixed_datasets_cache.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(datasets, f)
    
    return datasets

def preprocess_dataset_fixed(data: np.ndarray, scaler_type: str = 'robust', 
                           config: Optional[Any] = None) -> Tuple[np.ndarray, object]:
    """
    Preprocess fixed-size univariate time series data
    """
    n_samples, seq_len = data.shape
    
    # Reshape for scaling
    data_1d = data.reshape(-1, 1)
    
    # Initialize scaler
    if scaler_type == 'robust':
        scaler = RobustScaler(quantile_range=(5, 95))
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")
    
    # Fit and transform
    scaled_1d = scaler.fit_transform(data_1d)
    
    # Reshape back
    scaled_data = scaled_1d.reshape(n_samples, seq_len)
    
    # Add bifurcation-friendly noise if configured
    if config and config.use_phase_noise:
        noise = np.random.normal(0, config.phase_noise_std, scaled_data.shape)
        # Apply temporal smoothing
        from scipy.ndimage import gaussian_filter1d
        for i in range(n_samples):
            scaled_data[i] = gaussian_filter1d(scaled_data[i] + noise[i], sigma=1.0)
    
    return scaled_data, scaler

def prepare_dataset_with_fixed_size(dataset_info: Dict, config) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    """
    Prepare a single dataset for model training with fixed dimensions
    """
    # Extract reshaped data or original data
    if 'reshaped_data' in dataset_info:
        data = dataset_info['reshaped_data']
    elif 'data' in dataset_info:
        data = dataset_info['data']
    else:
        raise KeyError(f"Dataset info missing data")
    
    labels = dataset_info.get('labels', None)
    
    print(f"\nPreparing fixed-size dataset:")
    print(f"  Shape: {data.shape}")
    print(f"  Type: {'Synthetic' if dataset_info.get('is_synthetic', False) else 'Real'}")
    
    # Preprocess
    scaled_data, scaler = preprocess_dataset_fixed(data, scaler_type='robust', config=config)
    
    # Create dataset
    dataset = FixedSizeUnivariateDataset(
        scaled_data, 
        target_seq_len=config.seq_len,
        config=config
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_fixed_dataloaders(
        dataset, config
    )
    
    return train_loader, val_loader, test_loader, scaler

def create_fixed_dataloaders(dataset: Dataset, config: Any,
                           train_ratio: float = 0.7, 
                           val_ratio: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders with fixed batch size
    """
    n_samples = len(dataset)
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    test_size = n_samples - train_size - val_size
    
    # Split dataset
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Create dataloaders using safe method
    train_loader = create_safe_dataloader(train_dataset, config, is_train=True)
    val_loader = create_safe_dataloader(val_dataset, config, is_train=False)
    test_loader = create_safe_dataloader(test_dataset, config, is_train=False)
    
    print(f"  Dataloaders: Train={len(train_loader)} batches, "
          f"Val={len(val_loader)} batches, Test={len(test_loader)} batches")
    print(f"  Workers: Train={train_loader.num_workers}, "
          f"Val={val_loader.num_workers}, Test={test_loader.num_workers}")
    
    return train_loader, val_loader, test_loader

def safe_prepare_dataset(dataset_info: Dict, config) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    """
    Safely prepare dataset with comprehensive error handling
    """
    try:
        return prepare_dataset_with_fixed_size(dataset_info, config)
    except Exception as e:
        print(f"Error in dataset preparation: {e}")
        
        # Create synthetic dataset as fallback
        print("Creating synthetic dataset as fallback...")
        synthetic_info = create_synthetic_dataset("FallbackDataset", 
                                                 n_samples=500, 
                                                 seq_len=config.seq_len)
        
        return prepare_dataset_with_fixed_size(synthetic_info, config)

def analyze_fixed_dataset_shapes(datasets: Dict[str, Dict]):
    """Analyze and print fixed dataset shape statistics"""
    print("\n" + "=" * 80)
    print("DATASET SHAPE ANALYSIS")
    print("=" * 80)
    
    summary_data = []
    
    for name, info in datasets.items():
        if 'reshaped_data' in info:
            data = info['reshaped_data']
        elif 'data' in info:
            data = info['data']
        else:
            continue
            
        stats = info.get('padding_stats', {})
        is_synthetic = info.get('is_synthetic', False)
        
        summary_data.append({
            'Dataset': name,
            'Samples': data.shape[0],
            'Fixed Length': data.shape[1],
            'Type': 'Synthetic' if is_synthetic else 'Real',
            'Padded': stats.get('n_padded', 0),
            'Truncated': stats.get('n_truncated', 0)
        })
    
    # Print summary table
    if summary_data:
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        print("=" * 80)
    
    return summary_data

import contextlib

@contextlib.contextmanager
def clean_multiprocessing():
    """Context manager to handle multiprocessing cleanup"""
    try:
        yield
    finally:
        # Clean up multiprocessing resources
        try:
            import multiprocessing
            multiprocessing.active_children()  # Join any leftover processes
        except:
            pass

def create_safe_dataloader(dataset: Dataset, config: Any, is_train: bool = True) -> DataLoader:
    """
    Create a DataLoader with safe multiprocessing settings
    """
    # Determine number of workers
    if config.num_workers <= 0 or len(dataset) < config.batch_size * 2:
        num_workers = 0
    else:
        num_workers = min(2, config.num_workers)  # Limit workers to avoid issues
    
    # Set multiprocessing context
    multiprocessing_context = None
    if num_workers > 0:
        try:
            import multiprocessing
            # Use 'fork' on Unix, 'spawn' on Windows
            if hasattr(multiprocessing, 'get_context'):
                multiprocessing_context = multiprocessing.get_context('fork')
        except:
            pass
    
    return DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        drop_last=is_train,
        multiprocessing_context=multiprocessing_context
    )

# Update the main function to use robust loader
def load_datasets_for_pipeline(config):
    """Main function to load datasets for pipeline"""
    return load_and_reshape_aeon_datasets_robust(config, use_synthetic_fallback=True)