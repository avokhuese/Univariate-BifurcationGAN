"""
Standalone test for FID computation fix
"""

import numpy as np
import torch

# Mock config class
class MockConfig:
    def __init__(self):
        self.calculate_fid = True
        self.calculate_wasserstein = True
        self.calculate_jsd = True
        self.calculate_ks_test = True
        self.calculate_prd = True
        self.calculate_mmd = True
        self.calculate_acf_similarity = True
        self.calculate_psd_similarity = True

# Import evaluator
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation_univariate import UnivariateTimeSeriesEvaluator

def test_fid_with_simple_data():
    """Test FID with simple, clean data"""
    print("Testing FID computation with simple data...")
    
    config = MockConfig()
    evaluator = UnivariateTimeSeriesEvaluator(config)
    
    # Create simple, clean data
    np.random.seed(42)
    
    # Real data: sine waves
    n_samples = 30
    seq_len = 50
    
    real_data = []
    for i in range(n_samples):
        t = np.linspace(0, 2*np.pi, seq_len)
        signal = np.sin(t + np.random.uniform(0, 0.5))
        signal += np.random.normal(0, 0.05, seq_len)
        real_data.append(signal)
    
    # Fake data: similar but noisier
    fake_data = []
    for i in range(n_samples):
        t = np.linspace(0, 2*np.pi, seq_len)
        signal = 0.9 * np.sin(t + np.random.uniform(0, 0.5))
        signal += np.random.normal(0, 0.1, seq_len)
        fake_data.append(signal)
    
    # Reshape properly
    real_array = np.array(real_data).reshape(n_samples, seq_len, 1)
    fake_array = np.array(fake_data).reshape(n_samples, seq_len, 1)
    
    # Test feature extraction
    print("\n1. Testing feature extraction...")
    real_features = evaluator._extract_time_series_features_robust(real_array)
    fake_features = evaluator._extract_time_series_features_robust(fake_array)
    
    print(f"   Real features shape: {real_features.shape}")
    print(f"   Fake features shape: {fake_features.shape}")
    print(f"   Features finite: {np.isfinite(real_features).all()} and {np.isfinite(fake_features).all()}")
    
    # Test FID directly
    print("\n2. Testing FID computation directly...")
    fid_score = evaluator._compute_fid_score_robust(real_array, fake_array)
    print(f"   FID score: {fid_score:.4f}")
    print(f"   FID finite: {np.isfinite(fid_score)}")
    
    # Test full metrics
    print("\n3. Testing full metrics computation...")
    real_tensor = torch.FloatTensor(real_array)
    fake_tensor = torch.FloatTensor(fake_array)
    
    metrics = evaluator.compute_all_metrics(real_tensor, fake_tensor)
    
    print(f"   Number of metrics computed: {len(metrics)}")
    
    # Check all metrics
    all_ok = True
    for key, value in metrics.items():
        if not np.isfinite(value):
            print(f"   ✗ {key}: {value} (non-finite)")
            all_ok = False
        else:
            print(f"   ✓ {key}: {value:.4f}")
    
    return all_ok

if __name__ == "__main__":
    print("=" * 60)
    print("FID COMPUTATION FIX TEST")
    print("=" * 60)
    
    success = test_fid_with_simple_data()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ FID COMPUTATION TEST PASSED!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ FID COMPUTATION TEST FAILED")
        print("=" * 60)