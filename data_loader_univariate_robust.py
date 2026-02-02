"""
Robust univariate time series data loader with fallback options
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
import urllib.request
import zipfile
import requests
import io

class RobustUnivariateDataset(Dataset):
    """Robust dataset with multiple fallback options"""
    
    def __init__(self, data: np.ndarray, target_seq_len: int = 100, config: Optional[Any] = None):
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
            if data.shape[-1] == 1:
                data = data.squeeze(2)
            else:
                raise ValueError(f"Expected univariate data, got shape {data.shape}")
        
        self.target_seq_len = target_seq_len
        self.config = config
        
        # Reshape data
        self.data = self._reshape_to_fixed_size(data)
        self.n_samples, self.n_timesteps = self.data.shape
        
        print(f"RobustDataset: {self.n_samples} samples, {self.n_timesteps} timesteps")
    
    def _reshape_to_fixed_size(self, data: np.ndarray) -> np.ndarray:
        """Reshape data to fixed dimensions"""
        n_samples, n_timesteps = data.shape
        
        # Initialize output array
        reshaped_data = np.zeros((n_samples, self.target_seq_len))
        
        for i in range(n_samples):
            sample = data[i]
            
            if n_timesteps > self.target_seq_len:
                # For univariate, take central segment
                start_idx = (n_timesteps - self.target_seq_len) // 2
                reshaped_data[i] = sample[start_idx:start_idx + self.target_seq_len]
            elif n_timesteps < self.target_seq_len:
                # Pad with reflection
                pad_len = self.target_seq_len - n_timesteps
                reshaped_data[i] = np.pad(sample, (0, pad_len), mode='reflect')
            else:
                reshaped_data[i] = sample
        
        return reshaped_data
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx].reshape(-1, 1)
        
        return {
            'data': torch.FloatTensor(sample),
            'original_length': torch.tensor(self.n_timesteps),
            'sample_idx': torch.tensor(idx)
        }

def load_aeon_dataset_safely(dataset_name: str, use_cache: bool = True):
    """
    Safely load dataset from aeon with multiple fallback strategies
    """
    cache_dir = "./cache_univariate"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_name}.pkl")
    
    # Try to load from cache
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    try:
        # Try different aeon import strategies
        try:
            from aeon.datasets import load_classification
            X, y = load_classification(dataset_name)
            data_type = "classification"
        except:
            from aeon.datasets import load_regression
            X, y = load_regression(dataset_name)
            data_type = "regression"
        
        # Convert to numpy with proper handling
        if isinstance(X, list):
            # Handle list of variable length arrays
            max_len = max(x.shape[0] for x in X)
            min_len = min(x.shape[0] for x in X)
            
            print(f"  Variable lengths: min={min_len}, max={max_len}")
            
            # Process each sample
            X_processed = []
            for x in X:
                if x.ndim == 1:
                    x = x.reshape(-1, 1)
                elif x.ndim == 3:
                    x = x.squeeze(2)
                
                X_processed.append(x)
            
            # Pad to maximum length
            target_length = max_len
            for i in range(len(X_processed)):
                if X_processed[i].shape[0] < target_length:
                    pad_len = target_length - X_processed[i].shape[0]
                    X_processed[i] = np.pad(X_processed[i], ((0, pad_len), (0, 0)), mode='edge')
            
            X_array = np.stack(X_processed)
            
        elif isinstance(X, np.ndarray):
            X_array = X
            
            # Handle different array shapes
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
            elif X_array.ndim == 2:
                # Assume (n_samples, n_timesteps)
                pass
            elif X_array.ndim == 3:
                if X_array.shape[-1] == 1:
                    X_array = X_array.squeeze(2)
                else:
                    # Take first channel for univariate
                    X_array = X_array[:, :, 0]
        
        # Ensure we have 2D array
        if X_array.ndim != 2:
            X_array = X_array.reshape(X_array.shape[0], -1)
        
        result = {
            'data': X_array,
            'labels': y if y is not None else np.zeros(len(X_array)),
            'data_type': data_type,
            'n_samples': X_array.shape[0],
            'n_timesteps': X_array.shape[1]
        }
        
        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        
        return result
        
    except Exception as e:
        print(f"  Failed to load {dataset_name} from aeon: {e}")
        return None

def load_synthetic_dataset(dataset_name: str, n_samples: int = 1000, seq_len: int = 100):
    """
    Generate synthetic datasets for testing when real datasets fail
    """
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
        'data_type': 'synthetic',
        'n_samples': n_samples,
        'n_timesteps': seq_len
    }

def load_robust_datasets(config, use_synthetic_fallback: bool = True):
    """
    Load datasets with robust error handling and fallback options
    """
    datasets = {}
    successful = 0
    failed = 0
    
    print(f"Loading {len(config.dataset_names)} datasets with robust loader...")
    print("=" * 60)
    
    for dataset_name in tqdm(config.dataset_names, desc="Loading datasets"):
        try:
            # Try to load from aeon
            dataset_info = load_aeon_dataset_safely(dataset_name)
            
            if dataset_info is None and use_synthetic_fallback:
                print(f"  Using synthetic fallback for {dataset_name}")
                dataset_info = load_synthetic_dataset(
                    dataset_name, 
                    n_samples=min(1000, config.max_series_length),
                    seq_len=config.seq_len
                )
            
            if dataset_info is not None:
                datasets[dataset_name] = dataset_info
                successful += 1
                
                print(f"✓ {dataset_name:25s} | "
                      f"Samples: {dataset_info['n_samples']:5d} | "
                      f"Length: {dataset_info['n_timesteps']:5d}")
            else:
                failed += 1
                print(f"✗ {dataset_name:25s} | Failed")
                
        except Exception as e:
            failed += 1
            print(f"✗ {dataset_name:25s} | Error: {str(e)[:50]}...")
    
    print("=" * 60)
    print(f"Successfully loaded: {successful}/{len(config.dataset_names)} datasets")
    
    if successful == 0 and use_synthetic_fallback:
        print("\nNo datasets loaded. Creating minimal test dataset...")
        test_dataset = load_synthetic_dataset("TestDataset", n_samples=500, seq_len=config.seq_len)
        datasets["TestDataset"] = test_dataset
    
    return datasets

def preprocess_dataset_robust(data: np.ndarray, labels: Optional[np.ndarray] = None,
                            scaler_type: str = 'robust', config: Optional[Any] = None):
    """
    Robust preprocessing for univariate time series
    """
    n_samples, n_timesteps = data.shape
    
    # Handle labels
    if labels is not None:
        labels = np.array(labels)
        if labels.dtype.kind in ('f', 'c'):
            # Discretize continuous labels
            n_bins = min(10, len(np.unique(labels)))
            labels = pd.qcut(labels.flatten(), q=n_bins, labels=False, duplicates='drop')
        labels = labels.astype(int)
    else:
        labels = np.zeros(n_samples, dtype=int)
    
    # Choose scaler
    if scaler_type == 'robust':
        scaler = RobustScaler(quantile_range=(10, 90))
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")
    
    # Reshape for scaling
    data_2d = data.reshape(-1, 1)
    
    # Fit and transform
    try:
        scaled_2d = scaler.fit_transform(data_2d)
    except:
        # Fallback to simple normalization
        mean_val = np.mean(data_2d)
        std_val = np.std(data_2d)
        if std_val > 1e-8:
            scaled_2d = (data_2d - mean_val) / std_val
        else:
            scaled_2d = data_2d - mean_val
    
    # Reshape back
    scaled_data = scaled_2d.reshape(n_samples, n_timesteps)
    
    # Add small noise if configured
    if config and config.use_phase_noise:
        noise_std = config.phase_noise_std * np.std(scaled_data)
        scaled_data += np.random.normal(0, noise_std, scaled_data.shape)
    
    return scaled_data, scaler, labels

def create_robust_dataloaders(dataset_info: Dict, config) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    """
    Create robust dataloaders with error handling
    """
    try:
        data = dataset_info['data']
        labels = dataset_info.get('labels')
        
        print(f"\nPreparing dataset:")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {dataset_info.get('data_type', 'unknown')}")
        
        # Preprocess
        scaled_data, scaler, processed_labels = preprocess_dataset_robust(
            data, labels, scaler_type='robust', config=config
        )
        
        # Create dataset
        dataset = RobustUnivariateDataset(
            scaled_data,
            target_seq_len=config.seq_len,
            config=config
        )
        
        # Split
        n_samples = len(dataset)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        test_size = n_samples - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(config.batch_size, len(train_dataset)),
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(config.batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=min(config.batch_size, len(test_dataset)),
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        print(f"  Created loaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
        
        return train_loader, val_loader, test_loader, scaler
        
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        raise

def test_data_loading():
    """Test function to verify data loading works"""
    from config_univariate import config
    
    # Test with a few datasets
    test_datasets = ['ECG5000', 'FordB', 'CBF']
    
    for dataset_name in test_datasets:
        print(f"\nTesting {dataset_name}...")
        
        # Try loading
        dataset_info = load_aeon_dataset_safely(dataset_name)
        
        if dataset_info:
            print(f"  ✓ Loaded: {dataset_info['data'].shape}")
            
            # Try preprocessing
            scaled_data, scaler, labels = preprocess_dataset_robust(
                dataset_info['data'], dataset_info.get('labels')
            )
            print(f"  ✓ Preprocessed: mean={np.mean(scaled_data):.3f}, std={np.std(scaled_data):.3f}")
        else:
            print(f"  ✗ Failed to load")
    
    # Test synthetic fallback
    print("\nTesting synthetic dataset generation...")
    synth_info = load_synthetic_dataset("Test", n_samples=100, seq_len=100)
    print(f"  ✓ Synthetic: {synth_info['data'].shape}")

if __name__ == "__main__":
    test_data_loading()