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

class UnivariateTimeSeriesDataset(Dataset):
    """Advanced Dataset for univariate time series with bifurcation-aware sampling"""
    
    def __init__(self, data: np.ndarray, seq_len: int = 100, 
                 labels: Optional[np.ndarray] = None,
                 config: Optional[Any] = None):
        """
        Args:
            data: numpy array of shape (n_samples, n_timesteps)
            seq_len: sequence length for training
            labels: optional labels for conditional generation
            config: configuration object
        """
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim == 3:
            data = data.squeeze(1)
        
        self.data = data
        self.seq_len = seq_len
        self.labels = labels
        self.config = config
        self.n_samples, self.n_timesteps = data.shape
        
        # Pre-calculate valid start indices
        self.valid_starts = self._calculate_valid_start_indices()
        
        # Calculate sample weights
        self.sample_weights = self._calculate_sample_weights()
        
        print(f"Dataset initialized: {self.n_samples} samples, "
              f"{self.n_timesteps} timesteps")
    
    def _calculate_valid_start_indices(self) -> List[np.ndarray]:
        """Calculate valid start indices for each sample"""
        valid_starts = []
        for i in range(self.n_samples):
            if self.n_timesteps > self.seq_len:
                starts = np.arange(0, self.n_timesteps - self.seq_len + 1)
            else:
                starts = np.array([0])
            valid_starts.append(starts)
        return valid_starts
    
    def _calculate_sample_weights(self) -> np.ndarray:
        """Calculate weights for each sample to balance sampling"""
        weights = np.ones(self.n_samples)
        
        if self.labels is not None:
            # Balance by class
            unique_classes, class_counts = np.unique(self.labels, return_counts=True)
            class_weights = 1.0 / class_counts
            for cls, weight in zip(unique_classes, class_weights):
                weights[self.labels == cls] = weight
        
        # Adjust weights by sequence length
        sequence_lengths = [len(starts) for starts in self.valid_starts]
        length_weights = 1.0 / np.array(sequence_lengths)
        weights = weights * length_weights
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset with bifurcation-aware sampling"""
        sample = self.data[idx].copy()
        
        # Get sequence information
        if self.n_timesteps > self.seq_len:
            # Sample starting position with bias towards dynamic regions
            starts = self.valid_starts[idx]
            start_idx = self._sample_start_position(starts, sample)
            sample = sample[start_idx:start_idx + self.seq_len]
        
        elif self.n_timesteps < self.seq_len:
            # Pad sequence
            pad_len = self.seq_len - self.n_timesteps
            sample = np.pad(sample, (0, pad_len), mode='reflect')
        
        # Add small noise for regularization
        if self.config and self.config.use_phase_noise:
            noise_std = self.config.phase_noise_std * np.std(sample)
            sample = sample + np.random.normal(0, noise_std, sample.shape)
        
        # Reshape to (seq_len, 1) for univariate
        sample = sample.reshape(-1, 1)
        
        # Prepare return dictionary
        item = {
            'data': torch.FloatTensor(sample),
            'original_length': torch.tensor(min(self.n_timesteps, self.seq_len)),
            'sample_idx': torch.tensor(idx)
        }
        
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx])
        
        return item
    
    def _sample_start_position(self, starts: np.ndarray, sample: np.ndarray) -> int:
        """Sample starting position with bias towards regions with high dynamics"""
        if len(starts) == 1:
            return starts[0]
        
        # Calculate dynamic score for each possible segment
        dynamic_scores = []
        for start in starts:
            segment = sample[start:start + self.seq_len]
            # Calculate variance and gradient magnitude
            variance = np.var(segment)
            gradient = np.mean(np.abs(np.diff(segment)))
            score = variance * gradient
            dynamic_scores.append(score)
        
        dynamic_scores = np.array(dynamic_scores)
        dynamic_scores = dynamic_scores + 1e-8
        
        # Normalize to probabilities
        probabilities = dynamic_scores / dynamic_scores.sum()
        
        # Sample with probability proportional to dynamic score
        return np.random.choice(starts, p=probabilities)
    
    def get_sequence_lengths(self) -> List[int]:
        """Get original sequence lengths"""
        return [len(sample) for sample in self.data]

def load_aeon_datasets(config) -> Dict[str, Dict]:
    """Load multiple univariate datasets from aeon"""
    try:
        from aeon.datasets import load_classification, load_regression
    except ImportError:
        raise ImportError("Please install aeon: pip install aeon")
    
    datasets = {}
    failed_datasets = []
    
    print(f"Loading {len(config.dataset_names)} univariate datasets from aeon...")
    print("=" * 60)
    
    for dataset_name in tqdm(config.dataset_names, desc="Loading datasets"):
        try:
            # Try classification first, then regression
            try:
                X, y = load_classification(dataset_name)
            except:
                X, y = load_regression(dataset_name)
            
            # Convert to numpy array
            if isinstance(X, list):
                # Handle list of arrays with variable lengths
                X_processed = []
                max_length = 0
                
                for x in X:
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)
                    elif x.ndim > 2:
                        # Flatten if needed
                        x = x.reshape(-1, 1)
                    
                    X_processed.append(x)
                    max_length = max(max_length, x.shape[0])
                
                # Pad sequences to same length if needed
                target_length = min(max_length, config.max_series_length)
                for i, x in enumerate(X_processed):
                    if x.shape[0] < target_length:
                        # Pad with reflection
                        pad_len = target_length - x.shape[0]
                        X_processed[i] = np.pad(x, ((0, pad_len), (0, 0)), mode='reflect')
                    elif x.shape[0] > target_length:
                        # Truncate
                        X_processed[i] = x[:target_length]
                
                X_array = np.stack(X_processed)
                
            elif isinstance(X, np.ndarray):
                X_array = X
            
            # Ensure shape is (n_samples, n_timesteps, 1)
            if X_array.ndim == 2:
                X_array = X_array.reshape(X_array.shape[0], X_array.shape[1], 1)
            
            # Limit sequence length
            if X_array.shape[1] > config.max_series_length:
                X_array = X_array[:, :config.max_series_length, :]
            
            # Ensure minimum length
            if X_array.shape[1] < config.min_series_length:
                repeat_factor = config.min_series_length // X_array.shape[1] + 1
                X_array = np.tile(X_array, (1, repeat_factor, 1))[:, :config.min_series_length, :]
            
            # Calculate dataset statistics
            mean_val = np.mean(X_array)
            std_val = np.std(X_array)
            min_val = np.min(X_array)
            max_val = np.max(X_array)
            
            # Store dataset info
            datasets[dataset_name] = {
                'data': X_array,
                'labels': y if y is not None else np.zeros(len(X_array)),
                'n_samples': X_array.shape[0],
                'n_timesteps': X_array.shape[1],
                'n_features': 1,
                'statistics': {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'dynamic_range': max_val - min_val
                }
            }
            
            print(f"✓ {dataset_name:25s} | "
                  f"Samples: {X_array.shape[0]:5d} | "
                  f"Length: {X_array.shape[1]:5d}")
            
        except Exception as e:
            failed_datasets.append((dataset_name, str(e)))
            print(f"✗ {dataset_name:25s} | Failed: {str(e)[:50]}...")
    
    print("=" * 60)
    if failed_datasets:
        print(f"\nFailed to load {len(failed_datasets)} datasets:")
        for dataset, error in failed_datasets:
            print(f"  - {dataset}: {error}")
    
    # Save dataset metadata
    metadata_path = os.path.join(config.cache_dir, "dataset_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(datasets, f)
    
    return datasets

def preprocess_dataset(data: np.ndarray, labels: Optional[np.ndarray] = None,
                      scaler_type: str = 'robust', 
                      config: Optional[Any] = None) -> Tuple[np.ndarray, object, Optional[np.ndarray]]:
    """
    Advanced preprocessing for univariate time series data
    """
    n_samples, n_timesteps, n_features = data.shape
    
    # Handle labels
    if labels is not None:
        labels = np.array(labels)
        if labels.dtype.kind in ('f', 'c'):
            unique_labels = np.unique(labels)
            if len(unique_labels) > 100:
                labels = pd.qcut(labels.flatten(), q=10, labels=False)
            labels = labels.astype(int)
    else:
        labels = np.zeros(n_samples, dtype=int)
    
    # Reshape for scaling
    data_2d = data.reshape(-1, n_features)
    
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
    scaled_2d = scaler.fit_transform(data_2d)
    
    # Reshape back
    scaled_data = scaled_2d.reshape(n_samples, n_timesteps, n_features)
    
    # Add small amount of noise
    if config and config.use_phase_noise:
        noise_level = config.phase_noise_std * np.std(scaled_data, axis=(0, 1))
        scaled_data += np.random.normal(0, noise_level, scaled_data.shape)
    
    # Normalize per sample if needed
    if config and config.use_multiscale:
        for i in range(n_samples):
            sample_std = np.std(scaled_data[i])
            if sample_std > 1e-8:
                scaled_data[i] = scaled_data[i] / (sample_std + 1e-8)
    
    return scaled_data, scaler, labels

def create_dataloaders(dataset: Dataset, config: Any,
                      train_ratio: float = 0.7, 
                      val_ratio: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    """
    n_samples = len(dataset)
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    test_size = n_samples - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    print(f"Dataloaders created: Train={len(train_loader)} batches, "
          f"Val={len(val_loader)} batches, Test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

def prepare_dataset_for_model(dataset_info: Dict, config) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    """
    Prepare a single dataset for model training
    """
    # Extract data and labels
    data = dataset_info['data']
    labels = dataset_info.get('labels')
    
    print(f"\nPreprocessing dataset: {data.shape}")
    print(f"  Original shape: {data.shape}")
    
    # Preprocess
    scaled_data, scaler, processed_labels = preprocess_dataset(
        data, labels, scaler_type='robust', config=config
    )
    
    print(f"  Scaled shape: {scaled_data.shape}")
    
    # Create dataset
    dataset = UnivariateTimeSeriesDataset(
        scaled_data.squeeze(-1),  # Remove channel dimension for univariate
        seq_len=config.seq_len,
        labels=processed_labels,
        config=config
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, config
    )
    
    return train_loader, val_loader, test_loader, scaler