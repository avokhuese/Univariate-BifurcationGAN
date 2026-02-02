"""
Test the robust data loader
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_univariate import config
from data_loader_univariate_fixed import load_datasets_for_pipeline, safe_prepare_dataset

def test_robust_loader():
    print("Testing robust data loader...")
    print("=" * 60)
    
    # Use a smaller subset for testing
    config.dataset_names = ['ECG5000', 'FordB', 'CBF']
    config.seq_len = 100
    config.batch_size = 32
    
    # Load datasets
    datasets = load_datasets_for_pipeline(config)
    
    print(f"\nLoaded {len(datasets)} datasets")
    
    # Test preparing one dataset
    if datasets:
        dataset_name = list(datasets.keys())[0]
        print(f"\nTesting dataset preparation for: {dataset_name}")
        
        dataset_info = datasets[dataset_name]
        
        try:
            train_loader, val_loader, test_loader, scaler = safe_prepare_dataset(
                dataset_info, config
            )
            
            print(f"✓ Successfully prepared dataset")
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Val batches: {len(val_loader)}")
            print(f"  Test batches: {len(test_loader)}")
            
            # Test one batch
            batch = next(iter(train_loader))
            print(f"  Batch shape: {batch['data'].shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to prepare dataset: {e}")
            return False
    
    return False

if __name__ == "__main__":
    success = test_robust_loader()
    if success:
        print("\n✅ Robust loader test PASSED!")
    else:
        print("\n❌ Robust loader test FAILED!")