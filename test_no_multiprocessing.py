"""
Test without multiprocessing to avoid semaphore warnings
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test_without_multiprocessing():
    """Test the system without multiprocessing"""
    print("Testing without multiprocessing...")
    
    # Import and modify config
    from config_univariate import config
    
    # Disable multiprocessing for testing
    config.num_workers = 0
    config.batch_size = 16  # Smaller batch for testing
    config.seq_len = 50  # Shorter sequences
    config.dataset_names = ['TestDataset']  # Single dataset
    
    print(f"Config: workers={config.num_workers}, batch_size={config.batch_size}")
    
    # Test data loading
    from data_loader_univariate_fixed import load_datasets_for_pipeline
    datasets = load_datasets_for_pipeline(config)
    
    if datasets:
        print(f"✓ Loaded {len(datasets)} datasets")
        
        # Test training
        try:
            from main_univariate_pipeline import UnivariateAugmentationPipeline
            pipeline = UnivariateAugmentationPipeline(config)
            pipeline.datasets = datasets
            
            # Run a quick training test
            from gan_framework_univariate import create_gan_framework
            gan = create_gan_framework("vanilla_gan", config)
            
            print("✓ Created GAN framework")
            
            # Test with one batch
            dataset_name = list(datasets.keys())[0]
            from data_loader_univariate_fixed import safe_prepare_dataset
            train_loader, _, _, _ = safe_prepare_dataset(datasets[dataset_name], config)
            
            batch = next(iter(train_loader))
            real_data = batch['data'].to(config.device)
            
            # One training step
            stats = gan.train_step(real_data)
            #print(f"✓ Training step completed: G_loss={stats('g_loss', 0):.4f}")
            print("✓ Training step completed")
            return True
            
        except Exception as e:
            print(f"✗ Training test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING WITHOUT MULTIPROCESSING")
    print("=" * 60)
    
    success = test_without_multiprocessing()
    
    if success:
        print("\n✅ Test PASSED without multiprocessing issues!")
        print("\nYou can now run the full pipeline with:")
        print("  python main_unitivariate_pipeline.py")
        print(f"Device: {device}")

        print("Torch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA version:", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())

        if torch.cuda.is_available():
            print("GPU name:", torch.cuda.get_device_name(0))
    else:

        print("\n❌ Test FAILED")   