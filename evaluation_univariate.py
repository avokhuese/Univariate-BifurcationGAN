"""
Advanced evaluation metrics for univariate time series GANs
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats, spatial
from sklearn.metrics import pairwise_distances
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import scipy modules with safety
try:
    from scipy import stats, spatial
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, some metrics will be disabled")

try:
    from sklearn.metrics import pairwise_distances
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, some metrics will be disabled")
    
class UnivariateTimeSeriesEvaluator:
    """Comprehensive evaluator for univariate time series GANs"""
    
    def __init__(self, config):
        self.config = config
        #self.device = config.device
        self.metrics_history = {}
        
    def compute_all_metrics(self, real_data: torch.Tensor, fake_data: torch.Tensor,
                           real_labels: Optional[torch.Tensor] = None,
                           fake_labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute all evaluation metrics"""
        
        metrics = {}
        
 # Convert to numpy for scipy-based metrics
        real_np = real_data.cpu().numpy()
        fake_np = fake_data.cpu().numpy()
        
        # Clean data - remove NaNs and Infs
        real_np = self._clean_data(real_np)
        fake_np = self._clean_data(fake_np)
        
        # 1. Basic Distribution Metrics
        if self.config.calculate_wasserstein:
            metrics['wasserstein_distance'] = self._compute_wasserstein_distance(real_np, fake_np)
        
        if self.config.calculate_jsd:
            metrics['jensen_shannon_divergence'] = self._compute_js_divergence(real_np, fake_np)
        
        if self.config.calculate_ks_test:
            ks_stat, ks_pvalue = self._compute_ks_test(real_np, fake_np)
            metrics['ks_statistic'] = ks_stat
            metrics['ks_pvalue'] = ks_pvalue
        
        # 2. Diversity Metrics
        if self.config.calculate_fid:
                    fid_score = self._compute_fid_score_robust(real_np, fake_np)
                    metrics['fid_score'] = fid_score
        
        if self.config.calculate_prd:
            precision, recall, f1 = self._compute_prd_metrics(real_np, fake_np)
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1
        
        if self.config.calculate_mmd:
            metrics['mmd_rbf'] = self._compute_mmd(real_np, fake_np)
        
        # 3. Temporal Metrics
        if self.config.calculate_acf_similarity:
            metrics['acf_similarity'] = self._compute_acf_similarity(real_np, fake_np)
        
        if self.config.calculate_psd_similarity:
            metrics['psd_similarity'] = self._compute_psd_similarity(real_np, fake_np)
        
        # 4. Advanced Metrics
        metrics['correlation_structure'] = self._compute_correlation_structure(real_np, fake_np)
        metrics['stationarity_preservation'] = self._compute_stationarity_preservation(real_np, fake_np)
        metrics['bifurcation_consistency'] = self._compute_bifurcation_consistency(real_np, fake_np)
        
        # 5. Quality Scores
        metrics['overall_quality'] = self._compute_overall_quality(metrics)
        
        # Store in history
        self.metrics_history[len(self.metrics_history)] = metrics
        
        return metrics
    def _clean_data(self, data: np.ndarray) -> np.ndarray:
        """Clean data by removing NaNs and Infs"""
        # Flatten for processing
        original_shape = data.shape
        data_flat = data.flatten()
        
        # Remove NaNs and Infs
        mask = np.isfinite(data_flat)
        if not mask.all():
            print(f"  Warning: Found {np.sum(~mask)} non-finite values, replacing with mean")
            data_flat[~mask] = np.mean(data_flat[mask])
        
        # Reshape back
        data_clean = data_flat.reshape(original_shape)
        
        # Add small epsilon to avoid zeros
        data_clean = data_clean + 1e-10
        
        return data_clean
    
    # Update other methods to use robust versions
    def _compute_wasserstein_distance(self, real: np.ndarray, fake: np.ndarray) -> float:
        """Compute Wasserstein distance with robust handling"""
        try:
            from scipy.stats import wasserstein_distance
            
            real_clean = self._clean_sample(real.flatten())
            fake_clean = self._clean_sample(fake.flatten())
            
            # Ensure equal lengths
            min_len = min(len(real_clean), len(fake_clean))
            real_clean = real_clean[:min_len]
            fake_clean = fake_clean[:min_len]
            
            return wasserstein_distance(real_clean, fake_clean)
        except:
            return float('inf')
        
    def _compute_js_divergence(self, real: np.ndarray, fake: np.ndarray) -> float:
        """Compute Jensen-Shannon Divergence"""
        # Create histograms
        all_data = np.concatenate([real.flatten(), fake.flatten()])
        hist_range = (all_data.min(), all_data.max())
        
        hist_real, _ = np.histogram(real.flatten(), bins=50, range=hist_range, density=True)
        hist_fake, _ = np.histogram(fake.flatten(), bins=50, range=hist_range, density=True)
        
        # Add small epsilon to avoid zeros
        eps = 1e-10
        hist_real = hist_real + eps
        hist_fake = hist_fake + eps
        
        # Normalize
        hist_real = hist_real / hist_real.sum()
        hist_fake = hist_fake / hist_fake.sum()
        
        # Compute KL divergences
        kl_real_fake = stats.entropy(hist_real, hist_fake)
        kl_fake_real = stats.entropy(hist_fake, hist_real)
        
        # JS divergence
        js_div = 0.5 * (kl_real_fake + kl_fake_real)
        
        return js_div
  
    
    def _compute_ks_test(self, real: np.ndarray, fake: np.ndarray) -> Tuple[float, float]:
        """Compute Kolmogorov-Smirnov test"""
        real_flat = real.flatten()
        fake_flat = fake.flatten()
        
        # Sample if too large
        max_samples = 5000
        if len(real_flat) > max_samples:
            idx = np.random.choice(len(real_flat), max_samples, replace=False)
            real_flat = real_flat[idx]
        
        if len(fake_flat) > max_samples:
            idx = np.random.choice(len(fake_flat), max_samples, replace=False)
            fake_flat = fake_flat[idx]
        
        ks_stat, p_value = stats.ks_2samp(real_flat, fake_flat)
        
        return ks_stat, p_value
    
    # Update the _compute_fid_score_robust method to handle shape issues:
    def _compute_fid_score_robust(self, real: np.ndarray, fake: np.ndarray) -> float:
        """Compute Frechet Inception Distance with robust error handling - FIXED VERSION"""
        try:
            # Extract features using simple statistics
            real_features = self._extract_time_series_features_robust(real)
            fake_features = self._extract_time_series_features_robust(fake)
            
            # Check for valid features
            if real_features.size == 0 or fake_features.size == 0:
                print("  Warning: Empty features array in FID computation")
                return float('inf')
            
            # Ensure same number of features
            if real_features.shape[1] != fake_features.shape[1]:
                print(f"  Warning: Feature dimension mismatch: real={real_features.shape[1]}, fake={fake_features.shape[1]}")
                min_features = min(real_features.shape[1], fake_features.shape[1])
                real_features = real_features[:, :min_features]
                fake_features = fake_features[:, :min_features]
            
            # Limit number of samples for stability
            max_samples = 50
            if real_features.shape[0] > max_samples:
                idx = np.random.choice(real_features.shape[0], max_samples, replace=False)
                real_features = real_features[idx]
            
            if fake_features.shape[0] > max_samples:
                idx = np.random.choice(fake_features.shape[0], max_samples, replace=False)
                fake_features = fake_features[idx]
            
            # Compute statistics
            mu_real = np.mean(real_features, axis=0)
            mu_fake = np.mean(fake_features, axis=0)
            
            # Handle small sample sizes
            if len(real_features) < 2 or len(fake_features) < 2:
                print("  Warning: Insufficient samples for FID computation")
                return float('inf')
            
            sigma_real = np.cov(real_features, rowvar=False)
            sigma_fake = np.cov(fake_features, rowvar=False)
            
            # Handle singular matrices and ensure positive definite
            eps = 1e-6
            sigma_real = sigma_real + np.eye(sigma_real.shape[0]) * eps
            sigma_fake = sigma_fake + np.eye(sigma_fake.shape[0]) * eps
            
            # Compute matrix square root safely
            try:
                covmean = self._matrix_sqrt_safe(sigma_real.dot(sigma_fake))
                
                if np.iscomplexobj(covmean):
                    covmean = covmean.real
            except:
                # If matrix sqrt fails, use simpler distance
                print("  Warning: Matrix sqrt failed, using simple distance")
                diff = mu_real - mu_fake
                return float(diff.dot(diff))
            
            # FID calculation
            diff = mu_real - mu_fake
            fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
            
            # Ensure non-negative
            fid = max(fid, 0)
            
            return float(fid)
            
        except Exception as e:
            print(f"  FID computation failed: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')
        
    def _extract_time_series_features_robust(self, data: np.ndarray) -> np.ndarray:
        """Extract robust features from time series for FID computation - FIXED VERSION"""
        n_samples = min(100, data.shape[0])  # Limit samples for speed
        features_list = []
        
        # Define the number of features we expect
        n_features = 12  # We'll extract exactly 12 features
        
        for i in range(n_samples):
            sample = data[i].flatten()
            
            # Clean sample
            sample = self._clean_sample(sample)
            
            # Initialize feature array with zeros
            sample_features = np.zeros(n_features)
            
            # Extract features with error handling
            try:
                # Feature 0: Mean
                sample_features[0] = np.mean(sample)
                
                # Feature 1: Standard deviation
                sample_features[1] = np.std(sample)
                
                # Feature 2: Skewness (with safety)
                if len(sample) > 2 and np.std(sample) > 1e-8:
                    sample_features[2] = stats.skew(sample)
                
                # Feature 3: Kurtosis (with safety)
                if len(sample) > 3 and np.std(sample) > 1e-8:
                    sample_features[3] = stats.kurtosis(sample)
                
                # Features 4-8: Percentiles (10, 25, 50, 75, 90)
                percentiles = [10, 25, 50, 75, 90]
                for j, p in enumerate(percentiles, start=4):
                    try:
                        sample_features[j] = np.percentile(sample, p)
                    except:
                        sample_features[j] = np.median(sample)
                
                # Feature 9: Range
                sample_features[9] = np.max(sample) - np.min(sample)
                
                # Feature 10: Mean absolute difference
                if len(sample) > 1:
                    sample_features[10] = np.mean(np.abs(np.diff(sample)))
                
                # Feature 11: Simple entropy proxy (log of variance)
                if np.var(sample) > 0:
                    sample_features[11] = np.log(np.var(sample) + 1e-10)
                
            except Exception as e:
                # If any feature extraction fails, use default values
                print(f"  Warning: Feature extraction failed for sample {i}: {e}")
                # Keep the zero-initialized features
            
            features_list.append(sample_features)
        
        # Convert to numpy array and ensure consistent shape
        features_array = np.array(features_list)
        
        # Check shape consistency
        if features_array.ndim != 2:
            print(f"  Warning: Features array has {features_array.ndim} dimensions, expected 2")
            # Reshape if possible
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)
            elif features_array.ndim == 3:
                features_array = features_array.reshape(features_array.shape[0], -1)
        
        # Ensure all samples have the same number of features
        if features_array.shape[1] != n_features:
            print(f"  Warning: Features array has {features_array.shape[1]} features, expected {n_features}")
            # Pad or truncate to correct size
            if features_array.shape[1] > n_features:
                features_array = features_array[:, :n_features]
            else:
                padding = np.zeros((features_array.shape[0], n_features - features_array.shape[1]))
                features_array = np.hstack([features_array, padding])
        
        return features_array

    
    def _clean_sample(self, sample: np.ndarray) -> np.ndarray:
        """Clean a single sample - FIXED VERSION"""
        # Remove NaNs and Infs
        sample = sample[np.isfinite(sample)]
        
        # If empty or too short, return default values
        if len(sample) < 10:
            # Return a simple sinusoidal pattern
            t = np.linspace(0, 2*np.pi, 50)
            sample = np.sin(t) + 0.1 * np.random.randn(50)
        
        # Ensure minimum length
        if len(sample) < 10:
            sample = np.pad(sample, (0, 10 - len(sample)), mode='edge')
        
        # Add small epsilon to avoid zeros
        sample = sample + 1e-10
        
        return sample
    
    
    def _approximate_entropy_robust(self, data: np.ndarray, m: int, r: float) -> float:
        """Compute approximate entropy with robust error handling"""
        try:
            N = len(data)
            if N <= m + 1:
                return 0.0
            
            def _phi(m_val):
                vectors = np.array([data[i:i+m_val] for i in range(N - m_val + 1)])
                C = 0
                for i in range(len(vectors)):
                    distances = np.abs(vectors - vectors[i])
                    max_distances = np.max(distances, axis=1)
                    similar = np.sum(max_distances <= r)
                    C += similar / (N - m_val + 1)
                return C / (N - m_val + 1)
            
            return np.log(_phi(m) / _phi(m + 1))
        except:
            return 0.0
        
    def _matrix_sqrt_safe(self, mat: np.ndarray) -> np.ndarray:
            """Compute matrix square root with safety"""
            try:
                # Try eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(mat)
                eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive
                sqrt_mat = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                return sqrt_mat
            except:
                try:
                    # Fallback to scipy
                    from scipy.linalg import sqrtm
                    sqrt_mat = sqrtm(mat)
                    # Handle complex results
                    if np.iscomplexobj(sqrt_mat):
                        sqrt_mat = sqrt_mat.real
                    return sqrt_mat
                except:
                    # Final fallback: identity matrix
                    return np.eye(mat.shape[0])
   
    
    def _compute_prd_metrics(self, real: np.ndarray, fake: np.ndarray) -> Tuple[float, float, float]:
        """Compute Precision, Recall, and F1 for generated data"""
        try:
            # Use k-NN based precision and recall
            from sklearn.neighbors import NearestNeighbors
            
            # Reshape data
            real_2d = real.reshape(real.shape[0], -1)
            fake_2d = fake.reshape(fake.shape[0], -1)
            
            # Subsample if too large
            max_samples = min(1000, len(real_2d), len(fake_2d))
            
            if len(real_2d) > max_samples:
                idx = np.random.choice(len(real_2d), max_samples, replace=False)
                real_2d = real_2d[idx]
            
            if len(fake_2d) > max_samples:
                idx = np.random.choice(len(fake_2d), max_samples, replace=False)
                fake_2d = fake_2d[idx]
            
            # Fit nearest neighbors
            n_neighbors = 3
            nn_real = NearestNeighbors(n_neighbors=n_neighbors).fit(real_2d)
            nn_fake = NearestNeighbors(n_neighbors=n_neighbors).fit(fake_2d)
            
            # Precision: how many fake samples are close to real manifold
            distances_fake_to_real, _ = nn_real.kneighbors(fake_2d)
            precision = np.mean(distances_fake_to_real[:, 0] < np.percentile(distances_fake_to_real.flatten(), 95))
            
            # Recall: how many real samples are close to fake manifold
            distances_real_to_fake, _ = nn_fake.kneighbors(real_2d)
            recall = np.mean(distances_real_to_fake[:, 0] < np.percentile(distances_real_to_fake.flatten(), 95))
            
            # F1 score
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            return float(precision), float(recall), float(f1)
            
        except Exception as e:
            print(f"PRD computation failed: {e}")
            return 0.0, 0.0, 0.0
    
    def _compute_mmd(self, real: np.ndarray, fake: np.ndarray, sigma: float = 1.0) -> float:
        """Compute Maximum Mean Discrepancy with RBF kernel"""
        # Flatten to 1D for this implementation
        real_flat = real.flatten()
        fake_flat = fake.flatten()
        
        # Subsample
        max_samples = 1000
        if len(real_flat) > max_samples:
            idx = np.random.choice(len(real_flat), max_samples, replace=False)
            real_flat = real_flat[idx]
        
        if len(fake_flat) > max_samples:
            idx = np.random.choice(len(fake_flat), max_samples, replace=False)
            fake_flat = fake_flat[idx]
        
        # RBF kernel
        def rbf_kernel(x, y):
            return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))
        
        # Compute MMD
        n_real = len(real_flat)
        n_fake = len(fake_flat)
        
        # Real-real kernel
        K_real_real = 0
        for i in range(n_real):
            for j in range(i+1, n_real):
                K_real_real += rbf_kernel(real_flat[i], real_flat[j])
        K_real_real = 2 * K_real_real / (n_real * (n_real - 1)) if n_real > 1 else 0
        
        # Fake-fake kernel
        K_fake_fake = 0
        for i in range(n_fake):
            for j in range(i+1, n_fake):
                K_fake_fake += rbf_kernel(fake_flat[i], fake_flat[j])
        K_fake_fake = 2 * K_fake_fake / (n_fake * (n_fake - 1)) if n_fake > 1 else 0
        
        # Real-fake kernel
        K_real_fake = 0
        for i in range(n_real):
            for j in range(n_fake):
                K_real_fake += rbf_kernel(real_flat[i], fake_flat[j])
        K_real_fake = K_real_fake / (n_real * n_fake)
        
        # MMD^2
        mmd2 = K_real_real + K_fake_fake - 2 * K_real_fake
        
        return np.sqrt(max(mmd2, 0))
    
    def _compute_acf_similarity(self, real: np.ndarray, fake: np.ndarray, max_lag: int = 20) -> float:
        """Compute similarity of autocorrelation functions"""
        try:
            real_acf = self._compute_mean_acf(real, max_lag)
            fake_acf = self._compute_mean_acf(fake, max_lag)
            
            # Compute correlation between ACFs
            correlation = np.corrcoef(real_acf, fake_acf)[0, 1]
            
            return float(correlation)
        except:
            return 0.0
    
    def _compute_mean_acf(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute mean autocorrelation function across samples"""
        n_samples = min(100, data.shape[0])  # Limit for speed
        acfs = []
        
        for i in range(n_samples):
            sample = data[i].flatten()
            
            # Compute autocorrelation
            from statsmodels.tsa.stattools import acf
            try:
                sample_acf = acf(sample, nlags=max_lag, fft=True)
                acfs.append(sample_acf)
            except:
                pass
        
        if acfs:
            return np.mean(acfs, axis=0)
        else:
            return np.zeros(max_lag + 1)
    
    def _compute_psd_similarity(self, real: np.ndarray, fake: np.ndarray) -> float:
        """Compute similarity of power spectral densities"""
        try:
            from scipy.signal import welch
            
            # Compute mean PSD
            real_psd_mean = self._compute_mean_psd(real)
            fake_psd_mean = self._compute_mean_psd(fake)
            
            # Ensure same length
            min_len = min(len(real_psd_mean), len(fake_psd_mean))
            real_psd_mean = real_psd_mean[:min_len]
            fake_psd_mean = fake_psd_mean[:min_len]
            
            # Compute correlation in log space
            eps = 1e-10
            log_real = np.log(real_psd_mean + eps)
            log_fake = np.log(fake_psd_mean + eps)
            
            correlation = np.corrcoef(log_real, log_fake)[0, 1]
            
            return float(correlation)
        except:
            return 0.0
    
    def _compute_mean_psd(self, data: np.ndarray) -> np.ndarray:
        """Compute mean power spectral density"""
        n_samples = min(50, data.shape[0])
        psds = []
        
        for i in range(n_samples):
            sample = data[i].flatten()
            
            # Compute PSD using Welch's method
            from scipy.signal import welch
            try:
                freqs, psd = welch(sample, nperseg=min(256, len(sample)))
                psds.append(psd)
            except:
                pass
        
        if psds:
            return np.mean(psds, axis=0)
        else:
            return np.array([0.0])
    
    def _compute_correlation_structure(self, real: np.ndarray, fake: np.ndarray) -> float:
        """Compute similarity of correlation structure"""
        try:
            # Compute correlation matrices
            real_corr = self._compute_correlation_matrix(real)
            fake_corr = self._compute_correlation_matrix(fake)
            
            # Flatten correlation matrices (excluding diagonal)
            real_flat = real_corr[np.triu_indices_from(real_corr, k=1)]
            fake_flat = fake_corr[np.triu_indices_from(fake_corr, k=1)]
            
            # Compute correlation
            correlation = np.corrcoef(real_flat, fake_flat)[0, 1]
            
            return float(correlation)
        except:
            return 0.0
    
    def _compute_correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """Compute correlation matrix across time points"""
        # Reshape to (n_samples, seq_len)
        if data.ndim == 3:
            data = data.squeeze(2)
        
        # Limit samples for speed
        n_samples = min(200, data.shape[0])
        data_limited = data[:n_samples]
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(data_limited, rowvar=False)
        
        # Handle NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        return correlation_matrix
    
    def _compute_stationarity_preservation(self, real: np.ndarray, fake: np.ndarray) -> float:
        """Compute how well stationarity is preserved"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Compute stationarity scores
            real_stationarity = self._compute_stationarity_score(real)
            fake_stationarity = self._compute_stationarity_score(fake)
            
            # Similarity metric
            similarity = 1.0 - abs(real_stationarity - fake_stationarity)
            
            return float(similarity)
        except:
            return 0.5
    
    def _compute_stationarity_score(self, data: np.ndarray) -> float:
        """Compute fraction of samples that are stationary"""
        n_samples = min(50, data.shape[0])
        stationary_count = 0
        
        for i in range(n_samples):
            sample = data[i].flatten()
            
            try:
                # Augmented Dickey-Fuller test
                p_value = adfuller(sample)[1]
                if p_value < 0.05:  # Stationary at 95% confidence
                    stationary_count += 1
            except:
                pass
        
        return stationary_count / n_samples if n_samples > 0 else 0.0
    
    def _compute_bifurcation_consistency(self, real: np.ndarray, fake: np.ndarray) -> float:
        """Compute consistency of bifurcation-like dynamics"""
        try:
            # Compute Lyapunov exponents (approximate)
            real_lyap = self._estimate_lyapunov_exponent(real)
            fake_lyap = self._estimate_lyapunov_exponent(fake)
            
            # Similarity metric
            similarity = 1.0 / (1.0 + abs(real_lyap - fake_lyap))
            
            return float(similarity)
        except:
            return 0.5
    
    def _estimate_lyapunov_exponent(self, data: np.ndarray, embedding_dim: int = 3) -> float:
        """Estimate largest Lyapunov exponent (approximate)"""
        n_samples = min(20, data.shape[0])
        exponents = []
        
        for i in range(n_samples):
            sample = data[i].flatten()
            
            if len(sample) > embedding_dim * 10:
                try:
                    # Time delay embedding
                    embedded = np.array([sample[j:j+embedding_dim] 
                                        for j in range(len(sample) - embedding_dim)])
                    
                    # Nearest neighbor distances over time
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=2).fit(embedded)
                    distances, indices = nn.kneighbors(embedded)
                    
                    # Compute divergence rate
                    neighbor_distances = distances[:, 1]
                    if len(neighbor_distances) > 10:
                        # Simple regression on log distances
                        x = np.arange(len(neighbor_distances))
                        y = np.log(neighbor_distances + 1e-8)
                        
                        coeffs = np.polyfit(x[:min(100, len(x))], 
                                           y[:min(100, len(y))], 1)
                        exponents.append(coeffs[0])
                except:
                    pass
        
        return np.mean(exponents) if exponents else 0.0
    
    def _compute_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Compute overall quality score with robust handling"""
        try:
            # Weights for different metrics
            weights = {
                'wasserstein_distance': -0.2,
                'jensen_shannon_divergence': -0.15,
                'fid_score': -0.2,
                'mmd_rbf': -0.15,
                'precision': 0.1,
                'recall': 0.1,
                'f1_score': 0.2,
                'acf_similarity': 0.15,
                'psd_similarity': 0.15,
                'correlation_structure': 0.1,
                'stationarity_preservation': 0.1,
                'bifurcation_consistency': 0.15
            }
            
            total_weight = 0
            quality_score = 0.0
            
            for metric_name, weight in weights.items():
                if metric_name in metrics:
                    value = metrics[metric_name]
                    
                    # Skip infinite values
                    if np.isinf(value) or np.isnan(value):
                        continue
                    
                    # Normalize value
                    if metric_name in ['wasserstein_distance', 'jensen_shannon_divergence', 
                                     'fid_score', 'mmd_rbf']:
                        normalized = 1.0 / (1.0 + abs(value))
                    elif metric_name == 'ks_statistic':
                        normalized = 1.0 - min(value, 1.0)
                    elif metric_name == 'ks_pvalue':
                        normalized = value
                    else:
                        normalized = min(max(value, 0.0), 1.0)
                    
                    quality_score += normalized * weight
                    total_weight += abs(weight)
            
            # Normalize to [0, 1]
            if total_weight > 0:
                quality_score = (quality_score / total_weight + 1.0) / 2.0
            else:
                quality_score = 0.5
            
            return float(quality_score)
            
        except:
            return 0.5
        
    def validate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Validate and clean metrics"""
        validated = {}
        for key, value in metrics.items():
            if np.isfinite(value):
                validated[key] = value
            else:
                # Replace non-finite values with defaults
                if 'fid' in key.lower() or 'distance' in key.lower():
                    validated[key] = float('inf')
                elif 'similarity' in key.lower() or 'precision' in key.lower() or 'recall' in key.lower():
                    validated[key] = 0.0
                else:
                    validated[key] = 0.5
        return validated
    
    def save_metrics(self, model_name: str, dataset_name: str, metrics: Dict[str, float]):
        """Save metrics to file"""
        import pandas as pd
        import os
        
          # Validate metrics
        validated_metrics = self.validate_metrics(metrics)
        # Create DataFrame
        metrics_df = pd.DataFrame([metrics])
        metrics_df['model'] = model_name
        metrics_df['dataset'] = dataset_name
        metrics_df['timestamp'] = pd.Timestamp.now()
        
        # Save to CSV
        metrics_file = os.path.join(self.config.results_dir, f"metrics_{model_name}_{dataset_name}.csv")
        metrics_df.to_csv(metrics_file, index=False)
        
        print(f"Metrics saved to: {metrics_file}")
            # Also print key metrics
        print(f"  Key metrics - FID: {validated_metrics.get('fid_score', 'N/A'):.2f}, "
              f"Quality: {validated_metrics.get('overall_quality', 'N/A'):.3f}")
    def plot_metrics_comparison(self, model_metrics: Dict[str, Dict[str, float]], 
                               dataset_name: str):
        """Plot comparison of metrics across models"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Prepare data for plotting
        plot_data = []
        for model_name, metrics in model_metrics.items():
            for metric_name, value in metrics.items():
                if metric_name not in ['model', 'dataset', 'timestamp', 'ks_pvalue']:
                    plot_data.append({
                        'Model': model_name,
                        'Metric': metric_name,
                        'Value': value
                    })
        
        if not plot_data:
            return
        
        import pandas as pd
        df_plot = pd.DataFrame(plot_data)
        
        # Create figure
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        # Get unique metrics
        unique_metrics = df_plot['Metric'].unique()
        
        for idx, metric in enumerate(unique_metrics[:len(axes)]):
            ax = axes[idx]
            metric_data = df_plot[df_plot['Metric'] == metric]
            
            # Create bar plot
            sns.barplot(x='Model', y='Value', data=metric_data, ax=ax)
            ax.set_title(metric, fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for idx in range(len(unique_metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Model Comparison - {dataset_name}', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.config.results_dir, 
                                f"metrics_comparison_{dataset_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison plot saved to: {save_path}")

def evaluate_model_performance(model, dataloader, config, n_samples: int = 1000) -> Dict[str, float]:
    """Evaluate model performance by generating samples and comparing with real data"""
    
    evaluator = UnivariateTimeSeriesEvaluator(config)
    
    # Collect real data
    real_data = []
    for batch in dataloader:
        real_data.append(batch['data'])
        if len(real_data) * config.batch_size >= n_samples:
            break
    
    real_data = torch.cat(real_data, dim=0)[:n_samples]
    
    # Generate fake data
    model.eval()
    fake_data = []
    with torch.no_grad():
        n_batches = (n_samples + config.batch_size - 1) // config.batch_size
        for _ in range(n_batches):
            z = torch.randn(config.batch_size, config.latent_dim).to(config.device)
            batch_fake = model(z)
            fake_data.append(batch_fake.cpu())
    
    fake_data = torch.cat(fake_data, dim=0)[:n_samples]
    
    # Compute metrics
    metrics = evaluator.compute_all_metrics(real_data, fake_data)
    
    return metrics