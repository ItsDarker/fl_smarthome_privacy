"""
Local Differential Privacy Defense
"""
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def add_laplace_noise(update, epsilon, sensitivity=config.DP_SENSITIVITY):
    """
    Add Laplace noise to model update for differential privacy
    
    Args:
        update: Dictionary of parameter updates
        epsilon: Privacy budget (smaller = more privacy)
        sensitivity: L2 sensitivity (gradient clipping norm)
    
    Returns:
        Noisy update dictionary
    """
    noisy_update = {}
    scale = sensitivity / epsilon
    
    for param_name, param_values in update.items():
        # Generate Laplace noise
        noise = torch.from_numpy(
            np.random.laplace(0, scale, param_values.shape)
        ).float()
        
        noisy_update[param_name] = param_values + noise
    
    return noisy_update


def clip_gradients(update, max_norm=config.DP_SENSITIVITY):
    """
    Clip gradients to bounded L2 norm (gradient clipping)
    
    Args:
        update: Dictionary of parameter updates
        max_norm: Maximum L2 norm
    
    Returns:
        Clipped update dictionary
    """
    clipped_update = {}
    
    # Calculate total norm
    total_norm = 0.0
    for param_values in update.values():
        total_norm += torch.sum(param_values ** 2).item()
    total_norm = np.sqrt(total_norm)
    
    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = min(clip_coef, 1.0)
    
    for param_name, param_values in update.items():
        clipped_update[param_name] = param_values * clip_coef
    
    return clipped_update


def apply_ldp_defense(client_updates, epsilon):
    """
    Apply Local Differential Privacy to all client updates
    
    Args:
        client_updates: List of client update dictionaries
        epsilon: Privacy budget
    
    Returns:
        List of noisy client updates
    """
    protected_updates = []
    
    for update_info in client_updates:
        # Clip gradients first
        clipped_update = clip_gradients(update_info['update'])
        
        # Add noise
        noisy_update = add_laplace_noise(clipped_update, epsilon)
        
        # Create new update info with noisy update
        protected_info = {
            'client_id': update_info['client_id'],
            'update': noisy_update,
            'has_elderly': update_info['has_elderly'],
            'num_samples': update_info['num_samples']
        }
        protected_updates.append(protected_info)
    
    return protected_updates


class LDPDefense:
    """Local Differential Privacy defense wrapper"""
    
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.name = f"LDP (ε={epsilon})"
    
    def apply(self, client_updates):
        """Apply defense to client updates"""
        return apply_ldp_defense(client_updates, self.epsilon)


def test_ldp():
    """Test LDP defense"""
    print("Testing Local Differential Privacy...")
    
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
    
    print("\nOriginal update statistics:")
    for name, values in update['update'].items():
        print(f"  {name}: mean={values.mean():.4f}, std={values.std():.4f}")
    
    # Test different epsilon values
    for epsilon in [0.1, 1.0, 10.0]:
        print(f"\nApplying LDP with ε={epsilon}...")
        protected = apply_ldp_defense([update], epsilon)[0]
        
        print(f"Noisy update statistics:")
        for name, values in protected['update'].items():
            print(f"  {name}: mean={values.mean():.4f}, std={values.std():.4f}")


if __name__ == '__main__':
    test_ldp()
