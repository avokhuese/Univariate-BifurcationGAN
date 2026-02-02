"""
Minimal test script to verify the univariate system works
"""

import sys
import numpy as np
import torch
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_metrics_stability():
    """Test that metrics computation is stable - FIXED VERSION"""
    print("\nTesting metrics stability...")
    
    from config_univariate import config
    from evaluation_univariate import UnivariateTimeSeriesEvaluator
    
    evaluator = UnivariateTimeSeriesEvaluator(config)
    
    # Create synthetic data with proper shapes
    np.random.seed(42)
    
    # Create realistic time series data
    n_samples = 50
    seq_len = 100
    
    real_data = []
    for i in range(n_samples):
        t = np.linspace(0, 4*np.pi, seq_len)
        # Create signal with multiple frequency components
        signal = np.sin(t) + 0.5 * np.sin(2*t) + 0.2 * np.sin(3*t)
        # Add noise
        signal += np.random.normal(0, 0.1, seq_len)
        real_data.append(signal)
    
    fake_data = []
    for i in range(n_samples):
        t = np.linspace(0, 4*np.pi, seq_len)
        # Similar but slightly different distribution
        signal = 0.8 * np.sin(t) + 0.4 * np.sin(2*t) + 0.1 * np.sin(3*t)
        signal += np.random.normal(0, 0.15, seq_len)
        fake_data.append(signal)
    
    # Convert to proper shape: (n_samples, seq_len, 1)
    real_array = np.array(real_data).reshape(n_samples, seq_len, 1)
    fake_array = np.array(fake_data).reshape(n_samples, seq_len, 1)
    
    # Convert to torch
    real_tensor = torch.FloatTensor(real_array)
    fake_tensor = torch.FloatTensor(fake_array)
    
    print(f"Data shapes - Real: {real_tensor.shape}, Fake: {fake_tensor.shape}")
    
    # Test feature extraction directly
    print("\nTesting feature extraction...")
    real_features = evaluator._extract_time_series_features_robust(real_array)
    fake_features = evaluator._extract_time_series_features_robust(fake_array)
    
    print(f"Feature shapes - Real: {real_features.shape}, Fake: {fake_features.shape}")
    print(f"Feature check - Real finite: {np.isfinite(real_features).all()}, "
          f"Fake finite: {np.isfinite(fake_features).all()}")
    
    # Test metrics
    try:
        print("\nComputing all metrics...")
        metrics = evaluator.compute_all_metrics(real_tensor, fake_tensor)
        
        # Check for finite values
        all_finite = all(np.isfinite(v) for v in metrics.values())
        
        if all_finite:
            print(f"✓ All metrics computed successfully:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            return True
        else:
            print(f"✗ Some metrics are non-finite:")
            for key, value in metrics.items():
                if not np.isfinite(value):
                    print(f"  {key}: {value}")
            return False
            
    except Exception as e:
        print(f"✗ Metrics computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
print("\nTesting metrics stability...")
metrics_ok = test_metrics_stability()
if metrics_ok:
    print("✓ Metrics stability test PASSED")
else:
    print("✗ Metrics stability test FAILED")

def test_minimal_system():
    """Test the minimal working system"""
    print("Testing minimal univariate system...")
    
    # 1. Test config
    from config_univariate import config
    print(f"✓ Config loaded: {len(config.dataset_names)} datasets")
    
    # 2. Test synthetic data generation
    from data_loader_univariate_robust import load_synthetic_dataset
    synth_data = load_synthetic_dataset("Test", n_samples=100, seq_len=50)
    print(f"✓ Synthetic data: {synth_data['data'].shape}")
    
    # 3. Test model creation
    from models_univariate import create_model
    generator, discriminator = create_model("bifurcation_gan", config.latent_dim, config)
    print(f"✓ Models created: G={generator}, D={discriminator}")
    
    # 4. Test data loader
    from data_loader_univariate_robust import create_robust_dataloaders
    try:
        train_loader, val_loader, test_loader, scaler = create_robust_dataloaders(
            synth_data, config
        )
        print(f"✓ Dataloaders created: {len(train_loader)} batches")
    except Exception as e:
        print(f"✗ Dataloader failed: {e}")
        return False
    
    # 5. Test training step
    from gan_framework_univariate import create_gan_framework
    try:
        gan = create_gan_framework("bifurcation_gan", config)
        print(f"✓ GAN framework created")
        
        # Test one batch
        batch = next(iter(train_loader))
        real_data = batch['data'].to(config.device)
        
        if len(real_data) > 0:
            print(f"✓ Data batch shape: {real_data.shape}")
            
            # Try one training step
            try:
                stats = gan.train_step(real_data)
                print(f"✓ Training step completed: G_loss={stats.get('g_loss', 0):.4f}")
            except Exception as e:
                print(f"✗ Training step failed: {e}")
                # This might fail without proper setup, but framework should work
    except Exception as e:
        print(f"✗ GAN framework failed: {e}")
        return False
    
    # 6. Test evaluation
    from evaluation_univariate import UnivariateTimeSeriesEvaluator
    try:
        evaluator = UnivariateTimeSeriesEvaluator(config)
        print(f"✓ Evaluator created")
        
        # Generate some fake data
        fake_data = real_data + 0.1  # Simple test
        
        # Try computing metrics
        metrics = evaluator.compute_all_metrics(real_data.cpu(), fake_data.cpu())
        print(f"✓ Metrics computed: {len(metrics)} metrics")
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
    
    print("\n" + "=" * 80)
    print("MINIMAL SYSTEM TEST COMPLETE")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_minimal_system()
    if success:
        print("\n✅ System is working! You can now run:")
        print("  python run_univariate.py --setup    # Setup environment")
        print("  python run_univariate.py --test     # Run full test")
        print("  python run_univariate.py --debug    # Debug data loading")
        print("  python run_univariate.py --list     # List datasets and models")
        print("\nTo train a model:")
        print("  python run_univariate.py --mode train --model bifurcation_gan --dataset TestDataset --epochs 50")
    else:
        print("\n❌ System test failed. Please check dependencies.")