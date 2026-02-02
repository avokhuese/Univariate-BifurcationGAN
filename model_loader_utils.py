"""
Robust model loading utilities to handle version mismatches
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import os

def load_model_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device = None):
    """
    Load model checkpoint with robust error handling for version mismatches
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"  Loading checkpoint: {os.path.basename(checkpoint_path)}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Check if it's a full checkpoint or just state_dict
            if 'generator_state_dict' in checkpoint:
                state_dict = checkpoint['generator_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Clean state_dict before loading
        cleaned_state_dict = clean_state_dict(state_dict, model)
        
        # Load the cleaned state_dict
        model.load_state_dict(cleaned_state_dict, strict=False)
        
        print(f"    ✓ Successfully loaded (strict=False)")
        return True
        
    except Exception as e:
        print(f"    ✗ Failed to load: {e}")
        return False

def clean_state_dict(state_dict: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
    """
    Clean state_dict to remove mismatched keys and handle version changes
    """
    model_state_dict = model.state_dict()
    cleaned_dict = {}
    
    # Keys to ignore/remove (common mismatches)
    ignore_keys = [
        'positional_encoding',  # Common buffer that causes issues
        '_positional_encoding',
        'pos_encoding',
        'position_encoding',
        'pe',
        '.num_batches_tracked',  # BatchNorm buffers
    ]
    
    # Keys to rename (handle renames between versions)
    rename_map = {
        # Add any known renames here
        # 'old_key_name': 'new_key_name',
    }
    
    print(f"    Cleaning state_dict: {len(state_dict)} keys -> ", end='')
    
    for key, value in state_dict.items():
        # Apply renames first
        new_key = rename_map.get(key, key)
        
        # Check if we should ignore this key
        if any(ignore in key for ignore in ignore_keys):
            continue
        
        # Check if key exists in model
        if new_key in model_state_dict:
            # Check if shapes match
            if value.shape == model_state_dict[new_key].shape:
                cleaned_dict[new_key] = value
            else:
                print(f"\n      ⚠ Shape mismatch for {key}: {value.shape} vs {model_state_dict[new_key].shape}")
                # Try to reshape if possible
                if value.numel() == model_state_dict[new_key].numel():
                    cleaned_dict[new_key] = value.reshape(model_state_dict[new_key].shape)
                else:
                    print(f"      ✗ Cannot reshape, skipping")
        else:
            # Key not in model, but might be useful for partial loading
            # Try to find similar key (for handling module restructuring)
            found_match = False
            for model_key in model_state_dict.keys():
                # Check if this is a restructured version of the key
                simplified_key = key.replace('generator.', '').replace('discriminator.', '')
                simplified_model_key = model_key.replace('generator.', '').replace('discriminator.', '')
                
                if simplified_key == simplified_model_key and value.shape == model_state_dict[model_key].shape:
                    cleaned_dict[model_key] = value
                    found_match = True
                    print(f"\n      ↳ Mapped {key} -> {model_key}")
                    break
            
            if not found_match:
                # Try partial match (for hierarchical models)
                key_parts = key.split('.')
                for model_key in model_state_dict.keys():
                    model_key_parts = model_key.split('.')
                    
                    # Check if last parts match (common when module structure changes)
                    if key_parts[-1] == model_key_parts[-1] and value.shape == model_state_dict[model_key].shape:
                        cleaned_dict[model_key] = value
                        found_match = True
                        print(f"\n      ↳ Partial match: {key} -> {model_key}")
                        break
    
    print(f"{len(cleaned_dict)} usable keys")
    
    # Calculate load percentage
    if len(model_state_dict) > 0:
        load_percentage = len(cleaned_dict) / len(model_state_dict) * 100
        print(f"    Load percentage: {load_percentage:.1f}% ({len(cleaned_dict)}/{len(model_state_dict)})")
    
    return cleaned_dict

def find_best_checkpoint(model_name: str, dataset_name: str, save_dir: str):
    """
    Find the best checkpoint for a model-dataset pair
    """
    # Try different checkpoint patterns in order of preference
    patterns = [
        f"{model_name}_{dataset_name}_run0_best.pth",
        f"{model_name}_{dataset_name}_best.pth",
        f"{model_name}_{dataset_name}_run0_final.pth",
        f"{model_name}_{dataset_name}_final.pth",
        f"{model_name}_{dataset_name}.pth",
    ]
    
    # Also try with variations (underscores, case, etc.)
    variations = [
        f"{model_name.replace('_', '')}_{dataset_name}_best.pth",
        f"{model_name}_{dataset_name.replace('_', '')}_best.pth",
    ]
    
    all_patterns = patterns + variations
    
    for pattern in all_patterns:
        checkpoint_path = os.path.join(save_dir, pattern)
        if os.path.exists(checkpoint_path):
            return checkpoint_path
    
    # Try to find any file containing both model and dataset names
    for filename in os.listdir(save_dir):
        if model_name in filename and dataset_name in filename and filename.endswith('.pth'):
            return os.path.join(save_dir, filename)
    
    return None