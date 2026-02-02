"""
Simple debug without cache dependencies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_simple():
    """Simple debug without cache"""
    print("=" * 60)
    print("SIMPLE DEBUG (NO CACHE)")
    print("=" * 60)
    
    # Test basic imports
    print("\n1. Testing imports...")
    try:
        import torch
        import numpy as np
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"  ✓ NumPy: {np.__version__}")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    # Test data generation
    print("\n2. Testing synthetic data generation...")
    try:
        from data_loader_univariate_robust import load_synthetic_dataset
        data = load_synthetic_dataset("Test", n_samples=10, seq_len=50)
        print(f"  ✓ Generated data: {data['data'].shape}")
    except Exception as e:
        print(f"  ✗ Data generation failed: {e}")
        return False
    
    # Test model creation
    print("\n3. Testing model creation...")
    try:
        from config_univariate import config
        from models_univariate import create_model
        
        # Create simple config
        config.seq_len = 50
        config.latent_dim = 32
        config.batch_size = 8
        
        generator, discriminator = create_model("vanilla_gan", config.latent_dim, config)
        print(f"  ✓ Created: {generator.__class__.__name__}")
        print(f"  ✓ Parameters: {sum(p.numel() for p in generator.parameters())}")
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False
    
    # Test training step
    print("\n4. Testing training step...")
    try:
        from gan_framework_univariate import create_gan_framework
        
        gan = create_gan_framework("vanilla_gan", config)
        #vanilla_gan, wgan, wgan_gp, tts_gan, tts_wgan_gp, sig_wgan, sig_cwgan
        #bifurcation_gan, oscillatory_bifurcation_gan
        # Create dummy data
        dummy_data = torch.randn(config.batch_size, config.seq_len, 1)
        
        # One training step
        stats = gan.train_step(dummy_data)
        print(f"  ✓ Training step: G_loss={stats('g_loss', 0):.4f}")
    except Exception as e:
        print(f"  ✗ Training step failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ SIMPLE DEBUG COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = debug_simple()
    
    if success:
        print("\nNext steps:")
        print("1. Clean cache: python run_univariate.py --clean-cache")
        print("2. Run debug: python run_univariate.py --debug")
        print("3. Test training: python run_univariate.py --mode train --model vanilla_gan --dataset TestDataset --epochs 3")
    else:
        print("\nDebug failed. Check your installation.")