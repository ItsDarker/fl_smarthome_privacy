"""
Gradient Compression Defense
"""
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compress_gradients(update, compression_rate):
    """
    Compress gradients by keeping only top-k largest magnitude values
    
    Args:
        update: Dictionary of parameter updates
        compression_rate: Fraction of gradients to keep (0.1 = keep top 10%)
    
    Returns:
        Compressed update dictionary
    """
    compressed_update = {}
    
    for param_name, param_values in update.items():
        # Flatten
        flat_values = param_values.flatten()
        
        # Calculate k (number of values to keep)
        k = max(1, int(len(flat_values) * compression_rate))
        
        # Get indices of top-k absolute values
        _, top_indices = torch.topk(torch.abs(flat_values), k)
        
        # Create sparse tensor (set non-top-k to zero)
        compressed_flat = torch.zeros_like(flat_values)
        compressed_flat[top_indices] = flat_values[top_indices]
        
        # Reshape back
        compressed_update[param_name] = compressed_flat.reshape(param_values.shape)
    
    return compressed_update


def apply_compression_defense(client_updates, compression_rate):
    """
    Apply gradient compression to all client updates
    
    Args:
        client_updates: List of client update dictionaries
        compression_rate: Fraction of gradients to keep
    
    Returns:
        List of compressed client updates
    """
    compressed_updates = []
    
    for update_info in client_updates:
        # Compress update
        compressed = compress_gradients(update_info['update'], compression_rate)
        
        # Create new update info
        compressed_info = {
            'client_id': update_info['client_id'],
            'update': compressed,
            'has_elderly': update_info['has_elderly'],
            'num_samples': update_info['num_samples']
        }
        compressed_updates.append(compressed_info)
    
    return compressed_updates


class GradientCompressionDefense:
    """Gradient Compression defense wrapper"""
    
    def __init__(self, compression_rate):
        self.compression_rate = compression_rate
        self.name = f"Compression ({int(compression_rate*100)}%)"
    
    def apply(self, client_updates):
        """Apply defense to client updates"""
        return apply_compression_defense(client_updates, self.compression_rate)


def test_compression():
    """Test gradient compression"""
    print("Testing Gradient Compression...")
    
    # Create dummy update
    update = {
        'update': {
            'param1': torch.randn(10, 5),
            'param2': torch.randn(10)
        },
        'client_id': 0,
        'has_elderly': 1,
        'num_samples': 100
    }
    
    print("\nOriginal update:")
    for name, values in update['update'].items():
        non_zero = torch.count_nonzero(values).item()
        print(f"  {name}: shape={values.shape}, non-zero={non_zero}/{values.numel()}")
    
    # Test different compression rates
    for rate in [0.1, 0.25, 0.5]:
        print(f"\nCompressing to {int(rate*100)}%...")
        compressed = apply_compression_defense([update], rate)[0]
        
        print(f"Compressed update:")
        for name, values in compressed['update'].items():
            non_zero = torch.count_nonzero(values).item()
            print(f"  {name}: non-zero={non_zero}/{values.numel()} ({non_zero/values.numel()*100:.1f}%)")


if __name__ == '__main__':
    test_compression()
