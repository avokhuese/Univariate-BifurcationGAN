"""
Plot training losses for all GAN models across all datasets with early stopping points
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os
import sys
import glob
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_univariate import config

class TrainingLossPlotter:
    """Plotter for training losses across all models and datasets"""
    
    def __init__(self, config):
        self.config = config
        
        # Define models and their order
        self.models = ['vanilla_gan', 'wgan', 'wgan_gp', 'tts_gan', 'bifurcation_gan']
        
        # All 15 datasets
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
        
        # Dataset display names (shortened)
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
        
        # Create results directory
        self.plots_dir = os.path.join(config.results_dir, 'training_loss_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def find_training_checkpoints(self):
        """Find all training checkpoints with loss history"""
        print("\n" + "=" * 80)
        print("SEARCHING FOR TRAINING CHECKPOINTS")
        print("=" * 80)
        
        checkpoints = {}
        
        for model_name in self.models:
            for dataset_name in self.all_datasets:
                # Look for best checkpoints (they contain training history)
                patterns = [
                    f"{model_name}_{dataset_name}_run0_best.pth",
                    f"{model_name}_{dataset_name}_best.pth",
                    f"{model_name}_{dataset_name}_run0_final.pth",
                    f"{model_name}_{dataset_name}_final.pth"
                ]
                
                for pattern in patterns:
                    checkpoint_path = os.path.join(self.config.save_dir, pattern)
                    if os.path.exists(checkpoint_path):
                        if model_name not in checkpoints:
                            checkpoints[model_name] = {}
                        checkpoints[model_name][dataset_name] = checkpoint_path
                        print(f"✓ Found {model_name} - {dataset_name}: {pattern}")
                        break
        
        return checkpoints
    
    def load_training_history(self, checkpoint_path: str):
        """Load training history from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if isinstance(checkpoint, dict) and 'training_stats' in checkpoint:
                return checkpoint['training_stats']
            else:
                # Try to find training history in other formats
                for key in ['history', 'loss_history', 'stats', 'training_history']:
                    if key in checkpoint:
                        return checkpoint[key]
            
            return None
        except Exception as e:
            print(f"  Error loading {checkpoint_path}: {e}")
            return None
    
    def extract_losses_from_history(self, history: Dict):
        """Extract generator and discriminator losses from training history"""
        if not history:
            return None, None, None
        
        try:
            # Different possible key formats
            g_losses = []
            d_losses = []
            early_stop_epoch = None
            
            # Try to extract from different formats
            if 'g_losses' in history and 'd_losses' in history:
                g_losses = history['g_losses']
                d_losses = history['d_losses']
            elif 'epoch_losses' in history:
                # Parse epoch-wise losses
                for epoch_stats in history['epoch_losses']:
                    if isinstance(epoch_stats, dict):
                        g_losses.append(epoch_stats.get('g_loss', epoch_stats.get('generator_loss', 0)))
                        d_losses.append(epoch_stats.get('d_loss', epoch_stats.get('discriminator_loss', 0)))
            
            # Try to find early stopping point
            if 'early_stopping' in history:
                early_stop_epoch = history['early_stopping'].get('epoch', None)
            elif len(g_losses) < self.config.epochs:
                # If training stopped early, assume it's early stopping
                early_stop_epoch = len(g_losses) - 1
            
            # Convert to numpy arrays and smooth if needed
            if g_losses and d_losses:
                g_losses = np.array(g_losses)
                d_losses = np.array(d_losses)
                
                # Apply smoothing for better visualization
                if len(g_losses) > 10:
                    g_losses_smoothed = self._smooth_losses(g_losses, window_size=5)
                    d_losses_smoothed = self._smooth_losses(d_losses, window_size=5)
                else:
                    g_losses_smoothed = g_losses
                    d_losses_smoothed = d_losses
                
                return g_losses_smoothed, d_losses_smoothed, early_stop_epoch
            
        except Exception as e:
            print(f"  Error extracting losses: {e}")
        
        return None, None, None
    
    def _smooth_losses(self, losses: np.ndarray, window_size: int = 5):
        """Apply simple moving average smoothing to losses"""
        if len(losses) < window_size:
            return losses
        
        smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        # Pad to maintain same length
        pad_left = window_size // 2
        pad_right = window_size - pad_left - 1
        smoothed = np.pad(smoothed, (pad_left, pad_right), mode='edge')
        
        return smoothed
    
    def collect_all_losses(self):
        """Collect loss histories for all models and datasets"""
        print("\n" + "=" * 80)
        print("COLLECTING TRAINING LOSS HISTORIES")
        print("=" * 80)
        
        checkpoints = self.find_training_checkpoints()
        
        all_losses = {}
        
        for model_name in self.models:
            if model_name not in checkpoints:
                continue
            
            print(f"\nProcessing {model_name}...")
            model_losses = {}
            
            for dataset_name, checkpoint_path in checkpoints[model_name].items():
                print(f"  {dataset_name}: ", end='')
                
                history = self.load_training_history(checkpoint_path)
                if history:
                    g_losses, d_losses, early_stop = self.extract_losses_from_history(history)
                    
                    if g_losses is not None and d_losses is not None:
                        model_losses[dataset_name] = {
                            'g_losses': g_losses,
                            'd_losses': d_losses,
                            'early_stop': early_stop,
                            'checkpoint': os.path.basename(checkpoint_path)
                        }
                        print(f"✓ {len(g_losses)} epochs")
                    else:
                        print("✗ No loss data")
                else:
                    print("✗ No history")
            
            if model_losses:
                all_losses[model_name] = model_losses
                print(f"  → Collected {len(model_losses)} datasets")
        
        print(f"\n✓ Collected losses for {len(all_losses)} models")
        return all_losses
    
    def create_overview_plot(self, all_losses: Dict):
        """Create overview plot showing all models on one figure"""
        print("\n" + "=" * 80)
        print("CREATING OVERVIEW PLOT")
        print("=" * 80)
        
        # Create figure with subplots for each model
        fig, axes = plt.subplots(len(self.models), 2, figsize=(20, 4 * len(self.models)))
        
        if len(self.models) == 1:
            axes = axes.reshape(1, -1)
        
        for model_idx, model_name in enumerate(self.models):
            if model_name not in all_losses:
                continue
            
            model_losses = all_losses[model_name]
            
            # Generator loss subplot
            ax_g = axes[model_idx, 0]
            # Discriminator loss subplot
            ax_d = axes[model_idx, 1]
            
            # Plot each dataset
            for dataset_name, loss_info in model_losses.items():
                color_idx = list(model_losses.keys()).index(dataset_name)
                color = plt.cm.Set2(color_idx / max(1, len(model_losses) - 1))
                
                # Plot generator losses
                epochs = np.arange(len(loss_info['g_losses']))
                ax_g.plot(epochs, loss_info['g_losses'],
                         color=color,
                         alpha=0.7,
                         linewidth=1.5,
                         label=self.dataset_display_names.get(dataset_name, dataset_name))
                
                # Plot discriminator losses
                ax_d.plot(epochs, loss_info['d_losses'],
                         color=color,
                         alpha=0.7,
                         linewidth=1.5,
                         label=self.dataset_display_names.get(dataset_name, dataset_name))
                
                # Mark early stopping point if exists
                if loss_info['early_stop'] is not None and loss_info['early_stop'] < len(epochs):
                    ax_g.axvline(x=loss_info['early_stop'], color=color, linestyle='--', alpha=0.5)
                    ax_d.axvline(x=loss_info['early_stop'], color=color, linestyle='--', alpha=0.5)
            
            # Customize generator subplot
            ax_g.set_title(f'{self.model_display_names[model_name]} - Generator Losses', 
                          fontsize=12, fontweight='bold')
            ax_g.set_xlabel('Epoch', fontsize=10)
            ax_g.set_ylabel('Loss', fontsize=10)
            ax_g.grid(True, alpha=0.3)
            ax_g.legend(fontsize=8, loc='upper right', ncol=2)
            
            # Customize discriminator subplot
            ax_d.set_title(f'{self.model_display_names[model_name]} - Discriminator Losses', 
                          fontsize=12, fontweight='bold')
            ax_d.set_xlabel('Epoch', fontsize=10)
            ax_d.set_ylabel('Loss', fontsize=10)
            ax_d.grid(True, alpha=0.3)
            ax_d.legend(fontsize=8, loc='upper right', ncol=2)
            
            # Add model name annotation
            ax_g.text(0.02, 0.98, f'Model: {model_name}',
                     transform=ax_g.transAxes, fontsize=10, fontweight='bold',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.suptitle('GAN Training Losses Across All Models and Datasets\n'
                    '(Dashed vertical lines indicate early stopping points)', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save overview plot
        overview_path = os.path.join(self.plots_dir, 'training_losses_overview.png')
        plt.savefig(overview_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Overview plot saved to: {overview_path}")
        
        plt.show()
        
        return overview_path
    
    def create_individual_model_plots(self, all_losses: Dict):
        """Create individual plots for each model"""
        print("\n" + "=" * 80)
        print("CREATING INDIVIDUAL MODEL PLOTS")
        print("=" * 80)
        
        for model_name in self.models:
            if model_name not in all_losses:
                continue
            
            print(f"\nCreating plot for {model_name}...")
            model_losses = all_losses[model_name]
            
            # Create figure with subplots
            n_datasets = len(model_losses)
            n_cols = min(5, n_datasets)
            n_rows = (n_datasets + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
            
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            axes = axes.flatten()
            
            for idx, (dataset_name, loss_info) in enumerate(model_losses.items()):
                if idx >= len(axes):
                    break
                
                ax = axes[idx]
                
                # Plot both generator and discriminator losses
                epochs = np.arange(len(loss_info['g_losses']))
                
                ax.plot(epochs, loss_info['g_losses'],
                       color=self.model_colors[model_name],
                       linewidth=2,
                       alpha=0.8,
                       label='Generator Loss')
                
                ax.plot(epochs, loss_info['d_losses'],
                       color='darkorange',
                       linewidth=2,
                       alpha=0.8,
                       label='Discriminator Loss')
                
                # Mark early stopping point
                if loss_info['early_stop'] is not None and loss_info['early_stop'] < len(epochs):
                    ax.axvline(x=loss_info['early_stop'], 
                              color='red', 
                              linestyle='--', 
                              linewidth=1.5,
                              alpha=0.7,
                              label=f'Early Stop (Epoch {loss_info["early_stop"]})')
                
                # Customize subplot
                ax.set_title(f'{self.dataset_display_names.get(dataset_name, dataset_name)}', 
                            fontsize=11, fontweight='bold')
                ax.set_xlabel('Epoch', fontsize=9)
                ax.set_ylabel('Loss', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                
                # Add annotation for early stopping
                if loss_info['early_stop'] is not None:
                    ax.text(0.02, 0.98, f'Early Stop: {loss_info["early_stop"]}',
                           transform=ax.transAxes, fontsize=8,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for idx in range(len(model_losses), len(axes)):
                axes[idx].set_visible(False)
            
            # Add overall title
            plt.suptitle(f'{self.model_display_names[model_name]} - Training Losses Across Datasets\n'
                        'Generator (Blue/Model Color) vs Discriminator (Orange) Losses', 
                        fontsize=14, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            
            # Save individual model plot
            model_plot_path = os.path.join(self.plots_dir, f'{model_name}_training_losses.png')
            plt.savefig(model_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: {model_plot_path}")
    
    def create_comparison_matrix(self, all_losses: Dict):
        """Create comparison matrix showing convergence patterns"""
        print("\n" + "=" * 80)
        print("CREATING CONVERGENCE COMPARISON MATRIX")
        print("=" * 80)
        
        # Prepare data for matrix
        datasets = self.all_datasets
        models = self.models
        
        # Create convergence metrics matrix
        convergence_data = np.zeros((len(datasets), len(models)))
        early_stop_data = np.zeros((len(datasets), len(models)))
        
        for model_idx, model_name in enumerate(models):
            if model_name not in all_losses:
                continue
            
            for dataset_idx, dataset_name in enumerate(datasets):
                if dataset_name in all_losses[model_name]:
                    loss_info = all_losses[model_name][dataset_name]
                    
                    # Calculate convergence metric (final loss stability)
                    if len(loss_info['g_losses']) > 10:
                        last_10 = loss_info['g_losses'][-10:]
                        convergence = np.std(last_10) / (np.mean(last_10) + 1e-8)
                        convergence_data[dataset_idx, model_idx] = convergence
                    
                    # Record early stopping epoch
                    if loss_info['early_stop'] is not None:
                        early_stop_data[dataset_idx, model_idx] = loss_info['early_stop']
                    else:
                        early_stop_data[dataset_idx, model_idx] = len(loss_info['g_losses']) - 1
        
        # Create figure with two heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        
        # Plot 1: Convergence stability
        ax1 = axes[0]
        im1 = ax1.imshow(convergence_data, cmap='RdYlGn_r', aspect='auto')
        
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Dataset', fontsize=12)
        ax1.set_title('Loss Convergence Stability (Lower is Better)', 
                     fontsize=14, fontweight='bold')
        
        ax1.set_xticks(np.arange(len(models)))
        ax1.set_xticklabels([self.model_display_names[m] for m in models], 
                           rotation=45, ha='right')
        
        ax1.set_yticks(np.arange(len(datasets)))
        ax1.set_yticklabels([self.dataset_display_names[d] for d in datasets], 
                           fontsize=9)
        
        # Add text annotations
        for i in range(len(datasets)):
            for j in range(len(models)):
                if convergence_data[i, j] > 0:
                    text = ax1.text(j, i, f'{convergence_data[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        # Plot 2: Early stopping epochs
        ax2 = axes[1]
        im2 = ax2.imshow(early_stop_data, cmap='viridis', aspect='auto')
        
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Dataset', fontsize=12)
        ax2.set_title('Early Stopping Epoch (Higher = Longer Training)', 
                     fontsize=14, fontweight='bold')
        
        ax2.set_xticks(np.arange(len(models)))
        ax2.set_xticklabels([self.model_display_names[m] for m in models], 
                           rotation=45, ha='right')
        
        ax2.set_yticks(np.arange(len(datasets)))
        ax2.set_yticklabels([self.dataset_display_names[d] for d in datasets], 
                           fontsize=9)
        
        # Add text annotations
        for i in range(len(datasets)):
            for j in range(len(models)):
                if early_stop_data[i, j] > 0:
                    text = ax2.text(j, i, f'{int(early_stop_data[i, j])}',
                                   ha="center", va="center", color="white", fontsize=8)
        
        # Add colorbars
        plt.colorbar(im1, ax=ax1, label='Convergence Metric (σ/μ)')
        plt.colorbar(im2, ax=ax2, label='Training Epoch')
        
        plt.suptitle('GAN Training Performance Comparison\n'
                    'Left: Loss Stability | Right: Training Duration', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save comparison matrix
        matrix_path = os.path.join(self.plots_dir, 'convergence_comparison_matrix.png')
        plt.savefig(matrix_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Convergence matrix saved to: {matrix_path}")
        
        plt.show()
        
        return matrix_path
    
    def create_loss_statistics_summary(self, all_losses: Dict):
        """Create statistical summary of losses"""
        print("\n" + "=" * 80)
        print("CREATING LOSS STATISTICS SUMMARY")
        print("=" * 80)
        
        # Prepare data for statistics
        stats_data = []
        
        for model_name in self.models:
            if model_name not in all_losses:
                continue
            
            for dataset_name, loss_info in all_losses[model_name].items():
                g_losses = loss_info['g_losses']
                d_losses = loss_info['d_losses']
                
                if len(g_losses) > 0 and len(d_losses) > 0:
                    # Calculate statistics
                    stats = {
                        'Model': self.model_display_names[model_name],
                        'Dataset': self.dataset_display_names.get(dataset_name, dataset_name),
                        'Final_G_Loss': g_losses[-1],
                        'Final_D_Loss': d_losses[-1],
                        'Min_G_Loss': np.min(g_losses),
                        'Min_D_Loss': np.min(d_losses),
                        'Avg_G_Loss': np.mean(g_losses),
                        'Avg_D_Loss': np.mean(d_losses),
                        'Std_G_Loss': np.std(g_losses),
                        'Std_D_Loss': np.std(d_losses),
                        'Training_Epochs': len(g_losses),
                        'Early_Stop_Epoch': loss_info['early_stop'] if loss_info['early_stop'] else len(g_losses) - 1,
                        'Convergence_Ratio': np.std(g_losses[-10:]) / (np.mean(g_losses[-10:]) + 1e-8) if len(g_losses) >= 10 else np.nan
                    }
                    stats_data.append(stats)
        
        if not stats_data:
            print("No statistics data available")
            return
        
        # Create DataFrame
        stats_df = pd.DataFrame(stats_data)
        
        # Save to CSV
        csv_path = os.path.join(self.plots_dir, 'training_loss_statistics.csv')
        stats_df.to_csv(csv_path, index=False)
        print(f"✓ Statistics saved to: {csv_path}")
        
        # Create summary visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Final losses by model
        ax1 = axes[0, 0]
        final_losses = stats_df.groupby('Model')[['Final_G_Loss', 'Final_D_Loss']].mean()
        final_losses.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e'])
        ax1.set_title('Average Final Losses by Model', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(['Generator', 'Discriminator'])
        
        # Plot 2: Training epochs by model
        ax2 = axes[0, 1]
        epoch_stats = stats_df.groupby('Model')['Training_Epochs'].agg(['mean', 'std'])
        epoch_stats['mean'].plot(kind='bar', yerr=epoch_stats['std'], ax=ax2, 
                                color=[self.model_colors.get(m, '#888888') for m in epoch_stats.index])
        ax2.set_title('Average Training Epochs by Model', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Epochs', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Loss convergence by model
        ax3 = axes[1, 0]
        convergence_stats = stats_df.groupby('Model')['Convergence_Ratio'].mean()
        convergence_stats.plot(kind='bar', ax=ax3, 
                              color=[self.model_colors.get(m, '#888888') for m in convergence_stats.index])
        ax3.set_title('Loss Convergence Stability by Model', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Convergence Ratio (Lower is Better)', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Early stopping frequency
        ax4 = axes[1, 1]
        early_stop_counts = stats_df.groupby('Model').apply(
            lambda x: (x['Early_Stop_Epoch'] < x['Training_Epochs']).sum() / len(x) * 100
        )
        early_stop_counts.plot(kind='bar', ax=ax4, 
                              color=[self.model_colors.get(m, '#888888') for m in early_stop_counts.index])
        ax4.set_title('Percentage of Early Stopped Trainings', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Percentage (%)', fontsize=10)
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('GAN Training Statistics Summary', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save statistics plot
        stats_plot_path = os.path.join(self.plots_dir, 'training_statistics_summary.png')
        plt.savefig(stats_plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Statistics plot saved to: {stats_plot_path}")
        
        plt.show()
        
        # Print key insights
        print("\n" + "=" * 80)
        print("KEY INSIGHTS FROM TRAINING STATISTICS")
        print("=" * 80)
        
        # Find best performing model
        best_convergence = stats_df.groupby('Model')['Convergence_Ratio'].mean().idxmin()
        print(f"\nBest Converging Model: {best_convergence}")
        
        # Find most stable model (lowest loss variance)
        most_stable = stats_df.groupby('Model')['Std_G_Loss'].mean().idxmin()
        print(f"Most Stable Model: {most_stable}")
        
        # Find fastest converging model
        fastest = stats_df.groupby('Model')['Training_Epochs'].mean().idxmin()
        print(f"Fastest Training Model: {fastest}")
        
        # Calculate average early stopping rate
        early_stop_rate = (stats_df['Early_Stop_Epoch'] < stats_df['Training_Epochs']).mean() * 100
        print(f"Overall Early Stopping Rate: {early_stop_rate:.1f}%")
        
        return stats_df
    
    def run_complete_analysis(self):
        """Run complete training loss analysis"""
        print("\n" + "=" * 80)
        print("COMPLETE TRAINING LOSS ANALYSIS")
        print("=" * 80)
        
        # Step 1: Collect all loss histories
        all_losses = self.collect_all_losses()
        
        if not all_losses:
            print("\n❌ No training loss data found. Analysis cannot proceed.")
            print("\nMake sure models were trained with the framework that saves training history.")
            return
        
        # Step 2: Create overview plot
        self.create_overview_plot(all_losses)
        
        # Step 3: Create individual model plots
        self.create_individual_model_plots(all_losses)
        
        # Step 4: Create comparison matrix
        self.create_comparison_matrix(all_losses)
        
        # Step 5: Create statistics summary
        stats_df = self.create_loss_statistics_summary(all_losses)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nAll plots and analysis saved to: {self.plots_dir}")
        print("\nGenerated files:")
        print(f"  - training_losses_overview.png (all models overview)")
        print(f"  - {model_name}_training_losses.png (individual model plots)")
        print(f"  - convergence_comparison_matrix.png (performance matrix)")
        print(f"  - training_statistics_summary.png (statistics plots)")
        print(f"  - training_loss_statistics.csv (detailed statistics)")

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("GAN TRAINING LOSS ANALYSIS AND VISUALIZATION")
    print("=" * 80)
    print(f"Looking for training checkpoints in: {config.save_dir}")
    
    # Initialize plotter
    plotter = TrainingLossPlotter(config)
    
    # Run complete analysis
    plotter.run_complete_analysis()

if __name__ == "__main__":
    main()