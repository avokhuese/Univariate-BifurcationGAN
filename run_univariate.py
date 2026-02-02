"""
Updated main execution script with better error handling
"""

import argparse
import sys
import os
import subprocess
from typing import List, Optional

# Add current directory to path
os.environ['PYTHONWARNINGS'] = 'ignore'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'matplotlib',
        'seaborn',
        'tqdm',
        'aeon'
    ]
    
    print("Checking dependencies...")
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("Dependencies check complete!")

def setup_environment():
    """Setup the environment"""
    print("Setting up environment...")
    
    # Create directories
    directories = [
        './data/univariate',
        './saved_models_univariate',
        './results_univariate',
        './logs_univariate',
        './cache_univariate'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}")
    
    # Check dependencies
    check_dependencies()
    
    print("\nEnvironment setup complete!")

def run_debug_mode():
    """Run debug mode to test data loading"""
    print("\n" + "=" * 80)
    print("DEBUG MODE: TESTING DATA LOADING")
    print("=" * 80)
    
    try:
        # Import after dependencies are checked
        from data_loader_univariate_robust import test_data_loading
        test_data_loading()
    except Exception as e:
        print(f"Debug mode failed: {e}")
        import traceback
        traceback.print_exc()

def run_quick_test():
    """Run quick test of the system"""
    print("\n" + "=" * 80)
    print("QUICK TEST MODE")
    print("=" * 80)
    
    try:
        # Test imports
        print("Testing imports...")
        import torch
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        print("  ✓ Basic imports OK")
        
        # Test PyTorch
        print("Testing PyTorch...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  ✓ PyTorch device: {device}")
        
        # Test data loader
        print("Testing data loader...")
        from data_loader_univariate_robust import load_synthetic_dataset
        test_data = load_synthetic_dataset("Test", n_samples=10, seq_len=50)
        print(f"  ✓ Data loader: {test_data['data'].shape}")
        
        # Test model creation
        print("Testing model creation...")
        from config_univariate import config
        from models_univariate import create_model
        
        # Create a simple model
        generator, discriminator = create_model("bifurcation_gan", config.latent_dim, config)
        print(f"  ✓ Model creation: G={generator.__class__.__name__}, D={discriminator.__class__.__name__}")
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

def list_datasets_and_models():
    """List available datasets and models"""
    from config_univariate import config
    
    print("\n" + "=" * 80)
    print("AVAILABLE DATASETS")
    print("=" * 80)
    for i, dataset in enumerate(config.dataset_names, 1):
        params = config.get_dataset_params(dataset)
        print(f"{i:2d}. {dataset:25s} | Length: {params.get('avg_length', 'N/A'):4d} | "
              f"Classes: {params.get('n_classes', 'N/A'):2d}")
    
    print("\n" + "=" * 80)
    print("AVAILABLE MODELS")
    print("=" * 80)
    for i, model in enumerate(config.benchmark_models, 1):
        print(f"{i:2d}. {model}")
    
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Sequence length: {config.seq_len}")
    print(f"Latent dimension: {config.latent_dim}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Device: {config.device}")

def run_single_experiment(model_type: str, dataset_name: str, epochs: int):
    """Run a single experiment"""
    print("\n" + "=" * 80)
    print(f"SINGLE EXPERIMENT: {model_type} on {dataset_name}")
    print("=" * 80)
    
    # Update config
    from config_univariate import config
    config.epochs = epochs
    config.dataset_names = [dataset_name]
    
    # Run pipeline
    from main_univariate_pipeline import UnivariateAugmentationPipeline
    pipeline = UnivariateAugmentationPipeline(config)
    
    # Load datasets with robust loader
    from data_loader_univariate_robust import load_robust_datasets
    pipeline.datasets = load_robust_datasets(config, use_synthetic_fallback=True)
    
    # Train model
    result = pipeline.train_model_on_dataset(model_type, dataset_name, run_idx=0)
    
    if result:
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {result['model_path']}")
    else:
        print("\nExperiment failed!")

def run_ablation_study(dataset_name: str):
    """Run ablation study"""
    print("\n" + "=" * 80)
    print(f"ABLATION STUDY: {dataset_name}")
    print("=" * 80)
    
    from config_univariate import config
    
    # Test different configurations
    configurations = [
        {'name': 'baseline', 'use_bifurcation': False, 'use_oscillatory_dynamics': False},
        {'name': 'bifurcation_only', 'use_bifurcation': True, 'use_oscillatory_dynamics': False},
        {'name': 'oscillatory_only', 'use_bifurcation': False, 'use_oscillatory_dynamics': True},
        {'name': 'full_model', 'use_bifurcation': True, 'use_oscillatory_dynamics': True},
    ]
    
    results = {}
    
    for config_setup in configurations:
        print(f"\nTesting: {config_setup['name']}")
        
        # Update config
        config.use_bifurcation = config_setup['use_bifurcation']
        config.use_oscillatory_dynamics = config_setup['use_oscillatory_dynamics']
        
        # Run experiment
        try:
            result = run_single_experiment("bifurcation_gan", dataset_name, epochs=50)
            results[config_setup['name']] = result
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)
    for name, result in results.items():
        if result:
            metrics = result.get('final_metrics', {})
            print(f"{name:20#s} | FID: {metrics.get('fid_score', 'N/A'):.2f} | "
                  f"Quality: #{metrics.get('overall_quality', 'N/A'):.3f}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Univariate Time Series Augmentation with BifurcationGAN'
    )
    
    parser.add_argument('--setup', action='store_true',
                       help='Setup environment and install dependencies')
    
    parser.add_argument('--debug', action='store_true',
                       help='Run debug mode to test data loading')
    
    parser.add_argument('--test', action='store_true',
                       help='Run quick system test')
    
    parser.add_argument('--list', action='store_true',
                       help='List available datasets and models')
    
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'ablation', 'full', 'benchmark'],
                       help='Execution mode')
    
    parser.add_argument('--model', type=str, default='bifurcation_gan',
                       help='Model to use')
    
    parser.add_argument('--dataset', type=str, default='TestDataset',
                       help='Dataset to use')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    
    parser.add_argument('--install', action='store_true',
                       help='Install missing packages')
    # Add to argument parser:
    parser.add_argument('--test-minimal', action='store_true',
                    help='Run minimal working example')
    
    parser.add_argument('--no-mp', action='store_true',
                   help='Disable multiprocessing for DataLoader')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("UNIVARIATE TIME SERIES AUGMENTATION SYSTEM")
    print("=" * 80)
    
    # Setup environment
    if args.setup:
        setup_environment()
        return
    
    # Install packages
    if args.install:
        check_dependencies()
        return
    
    # Debug mode
    if args.debug:
        run_debug_mode()
        return
    
    # Quick test
    if args.test:
        run_quick_test()
        return
    
    # List datasets and models
    if args.list:
        list_datasets_and_models()
        return
    # Minimal working example
    if args.test_minimal:
        print("\nRunning minimal working example...")
        subprocess.run([sys.executable, "test_bifurcation_gan_minimal.py"])
        return
    
    if args.no_mp:
        print("Disabling multiprocessing...")
        config.num_workers = 0

    # Execute based on mode
    if args.mode == 'train':
        run_single_experiment(args.model, args.dataset, args.epochs)
    
    elif args.mode == 'ablation':
        run_ablation_study(args.dataset)
    
    elif args.mode == 'full':
        # Import and run full pipeline
        from main_univariate_pipeline import UnivariateAugmentationPipeline
        from config_univariate import config
        
        pipeline = UnivariateAugmentationPipeline(config)
        pipeline.run_full_pipeline()
    
    elif args.mode == 'benchmark':
        # Run benchmark with synthetic datasets
        from config_univariate import config
        config.dataset_names = ['TestDataset1', 'TestDataset2', 'TestDataset3']
        
        from main_univariate_pipeline import UnivariateAugmentationPipeline
        pipeline = UnivariateAugmentationPipeline(config)
        pipeline.benchmark_all_models(n_runs=2)
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()