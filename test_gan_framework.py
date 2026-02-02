"""
Test the GAN framework initialization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gan_initialization():
    """Test that all GAN models initialize properly"""
    print("=" * 60)
    print("TESTING GAN FRAMEWORK INITIALIZATION")
    print("=" * 60)
    
    from config_univariate import config
    
    # Test with simplified config
    config.seq_len = 50
    config.latent_dim = 32
    config.batch_size = 8
    config.num_workers = 0
    
    # Models to test
    models_to_test = [
        'vanilla_gan',
        'wgan',
        'bifurcation_gan',
        'oscillatory_bifurcation_gan'
    ]
    
    success_count = 0
    total_count = len(models_to_test)
    
    for model_type in models_to_test:
        print(f"\nTesting {model_type}...")
        
        try:
            from gan_framework_univariate import create_gan_framework
            
            # Create GAN
            gan = create_gan_framework(model_type, config)
            
            # Check attributes
            checks = []
            
            # Check generator
            if hasattr(gan, 'generator') and gan.generator is not None:
                checks.append("✓ Generator")
            else:
                checks.append("✗ Generator")
            
            # Check discriminator
            if hasattr(gan, 'discriminator') and gan.discriminator is not None:
                checks.append("✓ Discriminator")
            else:
                checks.append("✗ Discriminator")
            
            # Check optimizers (not all models have them)
            if hasattr(gan, 'optimizer_G') and gan.optimizer_G is not None:
                checks.append("✓ Optimizer_G")
            elif hasattr(gan, '_baseline_train_step'):
                checks.append("✓ Baseline train_step")
            else:
                checks.append("⚠ No optimizer_G")
            
            if hasattr(gan, 'optimizer_D') and gan.optimizer_D is not None:
                checks.append("✓ Optimizer_D")
            elif hasattr(gan, '_baseline_train_step'):
                checks.append("✓ Baseline train_step")
            else:
                checks.append("⚠ No optimizer_D")
            
            # Test training step
            try:
                dummy_data = torch.randn(config.batch_size, config.seq_len, 1)
                stats = gan.train_step(dummy_data)
                
                if 'g_loss' in stats and 'd_loss' in stats:
                    checks.append(f"✓ Train step (G: {stats['g_loss']:.4f}, D: {stats['d_loss']:.4f})")
                else:
                    checks.append(f"✗ Train step incomplete")
            except Exception as e:
                checks.append(f"✗ Train step failed: {e}")
            
            print(f"  {' | '.join(checks)}")
            
            # Count as success if we have both generator and discriminator
            if (hasattr(gan, 'generator') and gan.generator is not None and
                hasattr(gan, 'discriminator') and gan.discriminator is not None):
                success_count += 1
                print(f"  → Model initialized successfully")
            else:
                print(f"  → Model initialization incomplete")
            
        except Exception as e:
            print(f"  ✗ Failed to create {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print(f"RESULTS: {success_count}/{total_count} models initialized successfully")
    print("=" * 60)
    
    return success_count == total_count

if __name__ == "__main__":
    success = test_gan_initialization()
    
    if success:
        print("\n✅ ALL GAN MODELS INITIALIZED SUCCESSFULLY!")
        print("\nYou can now run:")
        print("  python run_univariate.py --mode train --model vanilla_gan --dataset TestDataset --epochs 5")
    else:
        print("\n❌ SOME MODELS FAILED TO INITIALIZE")
        print("\nTry running with minimal setup:")
        print("  python debug_simple.py")