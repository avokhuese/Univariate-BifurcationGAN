"""
Load trained GAN models and generate comprehensive comparison plots across 15 datasets
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os
import sys
import glob
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
from model_loader_utils import load_model_checkpoint, find_best_checkpoint
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_univariate import config
from data_loader_univariate_fixed import load_datasets_for_pipeline, safe_prepare_dataset
from gan_framework_univariate import create_gan_framework

class TrainedModelsPlotter:
    """Plotter for trained GAN models across 15 datasets"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Define models and their order
        self.models = ['vanilla_gan', 'wgan', 'wgan_gp', 'tts_gan', 'bifurcation_gan']
        
        # All 15 datasets from the benchmark
        self.all_datasets = [
            'ECG5000', 'FordB', 'CBF', 'ScreenType', 'StrawBerry',
            'Yoga', 'EOGHorizonSignal', 'Fungi', 'GestureMidAirD1',
            'InsectEPGRegularTrain', 'MelbournePedestrian', 'PigCVP',
            'PowerCons', 'SemgHandMovement', 'GunPointAgeSpan'
        ]
        
        # Colors for each model
        self.model_colors = {
            'vanilla_gan': '#1f77b4',      # blue
            'wgan': '#ff7f0e',             # orange
            'wgan_gp': '#2ca02c',          # green
            'tts_gan': '#d62728',          # red
            'bifurcation_gan': '#9467bd'   # purple
        }
        
        # Model display names
        self.model_display_names = {
            'vanilla_gan': 'Vanilla GAN',
            'wgan': 'WGAN',
            'wgan_gp': 'WGAN-GP',
            'tts_gan': 'TTS-GAN',
            'bifurcation_gan': 'BifurcationGAN'
        }
        
        # Dataset display names (shortened for plots)
        self.dataset_display_names = {
            'ECG5000': 'ECG',
            'FordB': 'FordB',
            'CBF': 'CBF',
            'ScreenType': 'Screen',
            'StrawBerry': 'StrawBerry',
            'Yoga': 'Yoga',
            'EOGHorizonSignal': 'EOG',
            'Fungi': 'Fungi',
            'GestureMidAirD1': 'Gesture',
            'InsectEPGRegularTrain': 'Insect',
            'MelbournePedestrian': 'Pedestrian',
            'PigCVP': 'PigCVP',
            'PowerCons': 'Power',
            'SemgHandMovement': 'SEMG',
            'GunPointAgeSpan': 'GunPoint'
        }
        
        # Line styles for different model groups
        self.line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
        
        # Create results directory
        self.plots_dir = os.path.join(config.results_dir, 'trained_models_comparison')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Track which models are available for each dataset
        self.available_models = {}
        
    def find_trained_models(self):
        """Find all trained models in the saved_models directory"""
        print("\n" + "=" * 80)
        print("SEARCHING FOR TRAINED MODELS")
        print("=" * 80)
        
        model_files = glob.glob(os.path.join(self.config.save_dir, "*.pth"))
        
        available_models = {}
        
        for model_file in model_files:
            filename = os.path.basename(model_file)
            
            # Parse filename pattern: {model}_{dataset}_run{run_idx}_{type}.pth
            try:
                parts = filename.split('_')
                
                # Find model name
                model_name = None
                for model in self.models:
                    if filename.startswith(model):
                        model_name = model
                        break
                
                if not model_name:
                    continue
                
                # Extract dataset name (could have underscores)
                remaining = filename[len(model_name)+1:]
                dataset_parts = []
                
                # Handle different filename patterns
                if '_run' in remaining:
                    run_split = remaining.split('_run')
                    dataset_name = run_split[0]
                elif '_best' in remaining:
                    dataset_name = remaining.replace('_best.pth', '')
                elif '_final' in remaining:
                    dataset_name = remaining.replace('_final.pth', '')
                else:
                    # Try to extract dataset name
                    dataset_name = remaining.replace('.pth', '')
                
                # Clean up dataset name
                dataset_name = dataset_name.rstrip('_')
                
                # Check if this is one of our datasets
                if dataset_name in self.all_datasets:
                    if dataset_name not in available_models:
                        available_models[dataset_name] = []
                    
                    if model_name not in available_models[dataset_name]:
                        available_models[dataset_name].append(model_name)
                        print(f"✓ Found {model_name} for {dataset_name}")
                        
            except Exception as e:
                continue
        
        self.available_models = available_models
        
        # Print summary
        print("\n" + "=" * 80)
        print("MODEL AVAILABILITY SUMMARY")
        print("=" * 80)
        
        for dataset in self.all_datasets:
            if dataset in available_models:
                models = available_models[dataset]
                print(f"{dataset:25s}: {len(models)} models - {', '.join(models)}")
            else:
                print(f"{dataset:25s}: No models found")
        
        total_models = sum(len(models) for models in available_models.values())
        print(f"\nTotal model-dataset pairs found: {total_models}")
        
        return available_models
    
    def load_trained_model(self, model_name: str, dataset_name: str):
        """Load a trained model with robust error handling"""
        
        # Find the best checkpoint
        checkpoint_path = find_best_checkpoint(model_name, dataset_name, self.config.save_dir)
        
        if not checkpoint_path:
            print(f"  ✗ No checkpoint found for {model_name} on {dataset_name}")
            return None
        
        print(f"  Found checkpoint: {os.path.basename(checkpoint_path)}")
        
        try:
            # Create GAN framework
            gan = create_gan_framework(model_name, self.config)
            
            # Load with robust method
            if load_model_checkpoint(gan.generator, checkpoint_path, self.device):
                print(f"    ✓ Successfully loaded generator")
            else:
                print(f"    ✗ Failed to load generator")
                return None
            
            # Try to load discriminator if present in checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'discriminator_state_dict' in checkpoint:
                    load_model_checkpoint(gan.discriminator, checkpoint_path, self.device)
            except:
                pass  # Discriminator loading is optional for generation
            
            return gan
            
        except Exception as e:
            print(f"    ✗ Failed to create GAN framework: {e}")
            return None
    
    def generate_comparison_data(self, n_samples_per_model: int = 3):
        """Generate samples from all available trained models - UPDATED"""
        print("\n" + "=" * 80)
        print("GENERATING COMPARISON SAMPLES (ROBUST LOADING)")
        print("=" * 80)
        
        # Load datasets
        print("Loading datasets...")
        datasets_info = load_datasets_for_pipeline(self.config)
        
        all_samples = {}
        
        for dataset_name in self.all_datasets:
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}")
            print('='*60)
            
            if dataset_name not in self.available_models:
                print(f"  ✗ No trained models available")
                continue
            
            dataset_samples = {}
            available_models = self.available_models[dataset_name]
            
            # Get real data
            try:
                if dataset_name in datasets_info:
                    _, _, test_loader, _ = safe_prepare_dataset(
                        datasets_info[dataset_name], self.config
                    )
                    
                    # Get real samples
                    real_batch = next(iter(test_loader))
                    real_data = real_batch['data'][:n_samples_per_model]
                    dataset_samples['real'] = {
                        'data': real_data.cpu().numpy(),
                        'color': 'black',
                        'label': 'Real Data',
                        'linewidth': 2.5,
                        'alpha': 0.9
                    }
                    print(f"  ✓ Real samples: {real_data.shape}")
                else:
                    print(f"  ✗ Dataset not in loaded data")
                    continue
                    
            except Exception as e:
                print(f"  ✗ Failed to get real samples: {e}")
                continue
            
            # Generate samples from each available model
            successful_models = 0
            for model_name in available_models:
                print(f"\n  Attempting {model_name}...")
                
                gan = self.load_trained_model(model_name, dataset_name)
                if gan is None:
                    continue
                
                try:
                    # Generate samples
                    print(f"    Generating {n_samples_per_model} samples...")
                    fake_samples = gan.generate_samples(n_samples_per_model)
                    
                    dataset_samples[model_name] = {
                        'data': fake_samples.numpy(),
                        'color': self.model_colors[model_name],
                        'label': self.model_display_names[model_name],
                        'linewidth': 1.5,
                        'alpha': 0.7,
                        'linestyle': self.line_styles[available_models.index(model_name) % len(self.line_styles)]
                    }
                    print(f"    ✓ Generated: {fake_samples.shape}")
                    successful_models += 1
                    
                except Exception as e:
                    print(f"    ✗ Generation failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            if len(dataset_samples) > 1:  # At least real + one model
                all_samples[dataset_name] = dataset_samples
                print(f"\n  → Successfully collected samples from {successful_models} models for {dataset_name}")
            else:
                print(f"\n  → No usable models for {dataset_name}")
        
        print(f"\n{'='*80}")
        print(f"SUMMARY: Collected samples from {len(all_samples)} datasets")
        print('='*80)
        
        for dataset_name, samples in all_samples.items():
            n_models = len([k for k in samples.keys() if k != 'real'])
            print(f"  {dataset_name}: {n_models} models")
        
        return all_samples

    def create_main_comparison_figure(self, all_samples: Dict):
        """Create the main 3x5 subplot figure comparing all models"""
        print("\n" + "=" * 80)
        print("CREATING MAIN COMPARISON FIGURE")
        print("=" * 80)
        
        # Group datasets into 3 groups of 5
        dataset_groups = [
            self.all_datasets[:5],    # Group 1: First 5 datasets
            self.all_datasets[5:10],  # Group 2: Next 5 datasets
            self.all_datasets[10:15]  # Group 3: Last 5 datasets
        ]
        
        # Create figure with 3 rows, 5 columns
        fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        axes = axes.flatten()
        
        # Plot each dataset in its own subplot
        plot_idx = 0
        for dataset_group in dataset_groups:
            for dataset_name in dataset_group:
                if plot_idx >= len(axes):
                    break
                    
                ax = axes[plot_idx]
                
                if dataset_name in all_samples:
                    self._plot_single_dataset(ax, dataset_name, all_samples[dataset_name])
                else:
                    ax.text(0.5, 0.5, f'{dataset_name}\n(No data)',
                           ha='center', va='center', fontsize=12)
                    ax.set_title(self.dataset_display_names.get(dataset_name, dataset_name), 
                                fontsize=12, fontweight='bold')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(False)
                
                plot_idx += 1
        
        # Add overall figure title
        plt.suptitle('Comparison of GAN Models Across 15 Time Series Datasets\n'
                    'Real Data (Black) vs Generated Samples (Colored by Model)', 
                    fontsize=20, fontweight='bold', y=1.02)
        
        # Add legend on the side
        self._add_comprehensive_legend(fig)
        
        plt.tight_layout()
        
        # Save the figure
        main_plot_path = os.path.join(self.plots_dir, 'main_comparison_all_datasets.png')
        plt.savefig(main_plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Main comparison figure saved to: {main_plot_path}")
        
        plt.show()
        
        return main_plot_path
    
    def _plot_single_dataset(self, ax, dataset_name: str, dataset_samples: Dict):
        """Plot a single dataset with all available models"""
        # Plot real data
        if 'real' in dataset_samples:
            real_info = dataset_samples['real']
            real_data = real_info['data']
            
            # Plot multiple real samples if available
            for i in range(min(2, len(real_data))):
                sample = real_data[i].flatten()
                ax.plot(sample,
                       color=real_info['color'],
                       linewidth=real_info['linewidth'],
                       alpha=real_info['alpha'] - i*0.2,
                       label=real_info['label'] if i == 0 else '')
        
        # Plot generated data from each model
        for model_name, model_info in dataset_samples.items():
            if model_name == 'real':
                continue
            
            fake_data = model_info['data']
            if len(fake_data) > 0:
                # Plot the first generated sample
                sample = fake_data[0].flatten()
                
                ax.plot(sample,
                       color=model_info['color'],
                       linestyle=model_info.get('linestyle', '-'),
                       linewidth=model_info['linewidth'],
                       alpha=model_info['alpha'],
                       label=model_info['label'])
        
        # Customize subplot
        ax.set_title(self.dataset_display_names.get(dataset_name, dataset_name), 
                    fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Time Steps', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Limit x-axis to show full sequence
        ax.set_xlim(0, self.config.seq_len)
        
        # Add dataset name as text in corner
        ax.text(0.02, 0.98, self.dataset_display_names.get(dataset_name, dataset_name),
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def _add_comprehensive_legend(self, fig):
        """Add a comprehensive legend to the figure"""
        # Create custom legend entries
        legend_elements = []
        
        # Real data entry
        legend_elements.append(
            plt.Line2D([0], [0], color='black', lw=2.5, label='Real Data')
        )
        
        # Model entries
        for model_name in self.models:
            if model_name in self.model_colors:
                legend_elements.append(
                    plt.Line2D([0], [0], color=self.model_colors[model_name], 
                              lw=1.5, label=self.model_display_names[model_name])
                )
        
        # Add legend
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=6, fontsize=11, framealpha=0.9,
                  bbox_to_anchor=(0.5, 0.01))
    
    def create_model_performance_summary(self):
        """Create a summary plot showing model performance across datasets"""
        print("\n" + "=" * 80)
        print("CREATING MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        
        # Try to load metrics from CSV files
        metrics_dir = os.path.join(self.config.results_dir, 'metrics')
        metrics_files = glob.glob(os.path.join(metrics_dir, "*.csv"))
        
        if not metrics_files:
            print("No metrics files found. Creating availability summary instead.")
            self._create_model_availability_summary()
            return
        
        # Load all metrics
        all_metrics = []
        for metrics_file in metrics_files:
            try:
                df = pd.read_csv(metrics_file)
                all_metrics.append(df)
                print(f"✓ Loaded: {os.path.basename(metrics_file)}")
            except:
                continue
        
        if not all_metrics:
            print("No valid metrics data found.")
            self._create_model_availability_summary()
            return
        
        # Combine all metrics
        metrics_df = pd.concat(all_metrics, ignore_index=True)
        
        # Create performance summary figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: FID scores across datasets
        ax1 = axes[0, 0]
        self._plot_fid_comparison(ax1, metrics_df)
        
        # Plot 2: Overall quality scores
        ax2 = axes[0, 1]
        self._plot_quality_comparison(ax2, metrics_df)
        
        # Plot 3: Model availability across datasets
        ax3 = axes[1, 0]
        self._plot_model_availability(ax3)
        
        # Plot 4: Best performing model per dataset
        ax4 = axes[1, 1]
        self._plot_best_models(ax4, metrics_df)
        
        plt.suptitle('GAN Model Performance Analysis Across 15 Datasets', 
                    fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save performance summary
        perf_plot_path = os.path.join(self.plots_dir, 'model_performance_summary.png')
        plt.savefig(perf_plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Performance summary saved to: {perf_plot_path}")
        
        plt.show()
        
        # Save detailed metrics table
        self._save_detailed_metrics_table(metrics_df)
    
    def _plot_fid_comparison(self, ax, metrics_df: pd.DataFrame):
        """Plot FID scores comparison"""
        # Prepare data for plotting
        plot_data = []
        
        for model_name in self.models:
            model_metrics = metrics_df[metrics_df['model'] == model_name]
            if len(model_metrics) > 0:
                # Get FID scores, handling infinite values
                fid_scores = []
                for _, row in model_metrics.iterrows():
                    fid = row.get('fid_score', float('inf'))
                    if np.isfinite(fid):
                        fid_scores.append(min(fid, 100))  # Cap at 100 for visualization
                    else:
                        fid_scores.append(100)
                
                if fid_scores:
                    plot_data.append({
                        'model': self.model_display_names[model_name],
                        'color': self.model_colors.get(model_name, '#888888'),
                        'fid_mean': np.mean(fid_scores),
                        'fid_std': np.std(fid_scores),
                        'fid_scores': fid_scores
                    })
        
        if not plot_data:
            ax.text(0.5, 0.5, 'No FID data available', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('FID Scores (Data Not Available)', fontsize=14)
            return
        
        # Create bar plot
        models = [d['model'] for d in plot_data]
        means = [d['fid_mean'] for d in plot_data]
        stds = [d['fid_std'] for d in plot_data]
        colors = [d['color'] for d in plot_data]
        
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('FID Score (Lower is Better)', fontsize=12)
        ax.set_title('Fréchet Inception Distance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{mean_val:.1f}', ha='center', va='bottom', fontsize=10)
    
    def _plot_quality_comparison(self, ax, metrics_df: pd.DataFrame):
        """Plot overall quality scores"""
        # Prepare data
        plot_data = []
        
        for model_name in self.models:
            model_metrics = metrics_df[metrics_df['model'] == model_name]
            if len(model_metrics) > 0:
                quality_scores = model_metrics['overall_quality'].dropna().values
                if len(quality_scores) > 0:
                    plot_data.append({
                        'model': self.model_display_names[model_name],
                        'color': self.model_colors.get(model_name, '#888888'),
                        'quality_mean': np.mean(quality_scores),
                        'quality_std': np.std(quality_scores),
                        'quality_scores': quality_scores
                    })
        
        if not plot_data:
            ax.text(0.5, 0.5, 'No quality data available', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Quality Scores (Data Not Available)', fontsize=14)
            return
        
        # Create bar plot
        models = [d['model'] for d in plot_data]
        means = [d['quality_mean'] for d in plot_data]
        stds = [d['quality_std'] for d in plot_data]
        colors = [d['color'] for d in plot_data]
        
        x_pos = np.arange(len(models))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Quality Score (Higher is Better)', fontsize=12)
        ax.set_title('Overall Quality Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    def _plot_model_availability(self, ax):
        """Plot model availability across datasets"""
        # Create availability matrix
        availability_matrix = []
        dataset_labels = []
        
        for dataset_name in self.all_datasets:
            dataset_labels.append(self.dataset_display_names.get(dataset_name, dataset_name))
            row = []
            for model_name in self.models:
                if dataset_name in self.available_models and model_name in self.available_models[dataset_name]:
                    row.append(1)  # Available
                else:
                    row.append(0)  # Not available
            availability_matrix.append(row)
        
        # Create heatmap
        im = ax.imshow(availability_matrix, cmap='RdYlGn', aspect='auto')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Dataset', fontsize=12)
        ax.set_title('Model Availability Across Datasets', fontsize=14, fontweight='bold')
        
        # Set ticks
        ax.set_xticks(np.arange(len(self.models)))
        ax.set_xticklabels([self.model_display_names[m] for m in self.models], 
                          rotation=45, ha='right')
        
        ax.set_yticks(np.arange(len(dataset_labels)))
        ax.set_yticklabels(dataset_labels, fontsize=9)
        
        # Add text annotations
        for i in range(len(dataset_labels)):
            for j in range(len(self.models)):
                text = ax.text(j, i, '✓' if availability_matrix[i][j] else '✗',
                             ha="center", va="center", 
                             color="black" if availability_matrix[i][j] else "gray",
                             fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, ticks=[0, 1])
    
    def _plot_best_models(self, ax, metrics_df: pd.DataFrame):
        """Plot best performing model for each dataset"""
        # Find best model for each dataset
        best_models = {}
        
        for dataset_name in self.all_datasets:
            dataset_metrics = metrics_df[metrics_df['dataset'] == dataset_name]
            if len(dataset_metrics) > 0:
                # Find model with highest quality score
                best_idx = dataset_metrics['overall_quality'].idxmax()
                best_model = dataset_metrics.loc[best_idx, 'model']
                best_quality = dataset_metrics.loc[best_idx, 'overall_quality']
                best_models[dataset_name] = (best_model, best_quality)
        
        if not best_models:
            ax.text(0.5, 0.5, 'No best model data available', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Best Models (Data Not Available)', fontsize=14)
            return
        
        # Prepare data for plotting
        dataset_labels = [self.dataset_display_names.get(d, d) for d in best_models.keys()]
        model_names = [self.model_display_names[best_models[d][0]] for d in best_models.keys()]
        quality_scores = [best_models[d][1] for d in best_models.keys()]
        colors = [self.model_colors[best_models[d][0]] for d in best_models.keys()]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(dataset_labels))
        
        bars = ax.barh(y_pos, quality_scores, color=colors, alpha=0.7)
        
        ax.set_xlabel('Quality Score', fontsize=12)
        ax.set_ylabel('Dataset', fontsize=12)
        ax.set_title('Best Performing Model per Dataset', fontsize=14, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(dataset_labels, fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels and model names
        for i, (bar, model_name, score) in enumerate(zip(bars, model_names, quality_scores)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{model_name} ({score:.3f})', 
                   ha='left', va='center', fontsize=9)
    
    def _create_model_availability_summary(self):
        """Create summary of which models are available for which datasets"""
        print("\nCreating model availability summary...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Count available models per dataset
        dataset_labels = []
        model_counts = []
        
        for dataset_name in self.all_datasets:
            dataset_labels.append(self.dataset_display_names.get(dataset_name, dataset_name))
            if dataset_name in self.available_models:
                model_counts.append(len(self.available_models[dataset_name]))
            else:
                model_counts.append(0)
        
        # Create bar plot
        y_pos = np.arange(len(dataset_labels))
        colors = ['green' if count == len(self.models) else 
                 'yellow' if count > 0 else 'red' 
                 for count in model_counts]
        
        bars = ax.barh(y_pos, model_counts, color=colors, alpha=0.7)
        
        ax.set_xlabel('Number of Available Models', fontsize=12)
        ax.set_ylabel('Dataset', fontsize=12)
        ax.set_title('Model Availability Across Datasets', fontsize=14, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(dataset_labels, fontsize=9)
        ax.set_xlim(0, len(self.models))
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, count in zip(bars, model_counts):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{count}/{len(self.models)}', 
                   ha='left', va='center', fontsize=9)
        
        # Add legend for colors
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.7, label='All models available'),
            plt.Rectangle((0,0),1,1, facecolor='yellow', alpha=0.7, label='Some models available'),
            plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='No models available')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        # Save availability plot
        avail_plot_path = os.path.join(self.plots_dir, 'model_availability.png')
        plt.savefig(avail_plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Model availability plot saved to: {avail_plot_path}")
        
        plt.show()
    
    def _save_detailed_metrics_table(self, metrics_df: pd.DataFrame):
        """Save detailed metrics table"""
        # Select relevant columns
        relevant_cols = ['model', 'dataset', 'fid_score', 'overall_quality', 
                        'mmd_rbf', 'acf_similarity', 'psd_similarity']
        
        # Filter available columns
        available_cols = [col for col in relevant_cols if col in metrics_df.columns]
        
        if not available_cols:
            print("No relevant metrics columns found")
            return
        
        # Create summary table
        summary_df = metrics_df[available_cols].copy()
        
        # Save to CSV
        summary_path = os.path.join(self.plots_dir, 'detailed_metrics_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Detailed metrics summary saved to: {summary_path}")
        
        # Also create a readable version
        readable_path = os.path.join(self.plots_dir, 'metrics_readable.txt')
        with open(readable_path, 'w') as f:
            f.write("GAN MODEL PERFORMANCE METRICS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            for model_name in self.models:
                model_data = summary_df[summary_df['model'] == model_name]
                if len(model_data) > 0:
                    f.write(f"\n{self.model_display_names[model_name].upper()}:\n")
                    f.write("-" * 40 + "\n")
                    
                    for _, row in model_data.iterrows():
                        dataset = row['dataset']
                        f.write(f"  {dataset}: ")
                        
                        metrics_str = []
                        if 'fid_score' in row:
                            fid = row['fid_score']
                            metrics_str.append(f"FID={fid:.2f}")
                        if 'overall_quality' in row:
                            quality = row['overall_quality']
                            metrics_str.append(f"Quality={quality:.3f}")
                        if 'mmd_rbf' in row:
                            mmd = row['mmd_rbf']
                            metrics_str.append(f"MMD={mmd:.4f}")
                        
                        f.write(", ".join(metrics_str) + "\n")
        
        print(f"✓ Readable metrics summary saved to: {readable_path}")
    
    def create_individual_dataset_plots(self, all_samples: Dict):
        """Create individual plots for each dataset showing all models"""
        print("\n" + "=" * 80)
        print("CREATING INDIVIDUAL DATASET PLOTS")
        print("=" * 80)
        
        for dataset_name in self.all_datasets:
            if dataset_name not in all_samples:
                continue
            
            print(f"\nCreating individual plot for {dataset_name}...")
            
            dataset_samples = all_samples[dataset_name]
            available_models = [m for m in self.models if m in dataset_samples]
            
            if len(available_models) == 0:
                print(f"  No models available for plotting")
                continue
            
            # Create figure with subplots
            n_models = len(available_models)
            fig, axes = plt.subplots(n_models + 1, 1, 
                                    figsize=(14, 3 * (n_models + 1)),
                                    sharex=True)
            
            if n_models == 0:
                axes = [axes] if not isinstance(axes, list) else axes
            
            # Plot real data
            if 'real' in dataset_samples:
                real_info = dataset_samples['real']
                real_data = real_info['data']
                
                # Plot multiple real samples
                for i in range(min(3, len(real_data))):
                    sample = real_data[i].flatten()
                    axes[0].plot(sample,
                               color=real_info['color'],
                               linewidth=real_info['linewidth'] - i*0.5,
                               alpha=real_info['alpha'] - i*0.2,
                               label=f'Real Sample {i+1}')
                
                axes[0].set_title(f'{dataset_name} - Real Data', 
                                 fontsize=14, fontweight='bold')
                axes[0].legend(loc='upper right', fontsize=10)
                axes[0].grid(True, alpha=0.3)
                axes[0].set_ylabel('Value', fontsize=11)
            
            # Plot generated data from each model
            for model_idx, model_name in enumerate(available_models):
                ax_idx = model_idx + 1
                model_info = dataset_samples[model_name]
                fake_data = model_info['data']
                
                # Plot multiple generated samples
                for i in range(min(3, len(fake_data))):
                    sample = fake_data[i].flatten()
                    axes[ax_idx].plot(sample,
                                    color=model_info['color'],
                                    linestyle=model_info.get('linestyle', '-'),
                                    linewidth=model_info['linewidth'] - i*0.3,
                                    alpha=model_info['alpha'] - i*0.2,
                                    label=f'Generated Sample {i+1}')
                
                axes[ax_idx].set_title(f'{dataset_name} - {self.model_display_names[model_name]}', 
                                      fontsize=12, fontweight='bold')
                axes[ax_idx].legend(loc='upper right', fontsize=9)
                axes[ax_idx].grid(True, alpha=0.3)
                axes[ax_idx].set_ylabel('Value', fontsize=11)
            
            # Set common x-label for bottom subplot
            axes[-1].set_xlabel('Time Steps', fontsize=11)
            
            # Add overall title
            plt.suptitle(f'Dataset: {dataset_name} - Detailed Model Comparison', 
                        fontsize=16, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            
            # Save individual plot
            safe_dataset_name = dataset_name.replace('/', '_').replace('\\', '_')
            plot_path = os.path.join(self.plots_dir, f'{safe_dataset_name}_detailed.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: {plot_path}")
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 80)
        print("COMPLETE GAN MODEL ANALYSIS PIPELINE")
        print("=" * 80)
        
        # Step 1: Find trained models
        self.find_trained_models()
        
        # Step 2: Generate comparison samples
        all_samples = self.generate_comparison_data(n_samples_per_model=3)
        
        if not all_samples:
            print("\n❌ No samples generated. Analysis cannot proceed.")
            return
        
        # Step 3: Create main comparison figure
        self.create_main_comparison_figure(all_samples)
        
        # Step 4: Create individual dataset plots
        self.create_individual_dataset_plots(all_samples)
        
        # Step 5: Create performance summary
        self.create_model_performance_summary()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nAll plots and analysis saved to: {self.plots_dir}")
        print("\nGenerated files:")
        print(f"  - main_comparison_all_datasets.png (3x5 subplot comparison)")
        print(f"  - model_performance_summary.png (performance metrics)")
        print(f"  - individual dataset plots (*_detailed.png)")
        print(f"  - detailed_metrics_summary.csv (metrics data)")
        print(f"  - metrics_readable.txt (readable summary)")

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("TRAINED GAN MODELS COMPARISON VISUALIZATION")
    print("=" * 80)
    print(f"Looking for models in: {config.save_dir}")
    
    # Initialize plotter
    plotter = TrainedModelsPlotter(config)
    
    # Run complete analysis
    plotter.run_complete_analysis()

if __name__ == "__main__":
    main()