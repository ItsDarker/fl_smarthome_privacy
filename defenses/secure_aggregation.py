"""
Secure Aggregation Defense (Simplified Simulation)
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def simulate_secure_aggregation(client_updates):
    """
    Simulate secure aggregation by hiding individual updates
    In real secure aggregation, cryptographic protocols ensure server
    only sees aggregated sum without seeing individual contributions.
    
    For simulation: We aggregate first, then provide only aggregate to attack
    
    Args:
        client_updates: List of client update dictionaries
    
    Returns:
        Single aggregated update (simulating what attacker sees)
    """
    if len(client_updates) == 0:
        return []
    
    # Calculate total samples
    total_samples = sum(u['num_samples'] for u in client_updates)
    
    # Initialize aggregated update
    aggregated_update = {}
    first_update = client_updates[0]['update']
    
    for param_name in first_update:
        aggregated_update[param_name] = torch.zeros_like(first_update[param_name])
    
    # Weighted averaging
    for update_info in client_updates:
        weight = update_info['num_samples'] / total_samples
        for param_name in update_info['update']:
            aggregated_update[param_name] += weight * update_info['update'][param_name]
    
    # Return as single "update" that adversary sees
    # Label is unknown to adversary in secure aggregation
    secure_update = {
        'client_id': -1,  # Unknown
        'update': aggregated_update,
        'has_elderly': -1,  # Unknown - adversary must guess
        'num_samples': total_samples
    }
    
    return [secure_update]  # Return as list for consistency


class SecureAggregationDefense:
    """
    Secure Aggregation defense wrapper
    
    Note: In practice, this requires cryptographic protocols (e.g., Bonawitz et al. 2017)
    Here we simulate the privacy property: adversary only sees aggregated result
    """
    
    def __init__(self):
        self.name = "Secure Aggregation"
    
    def apply(self, client_updates):
        """
        Apply defense - in reality returns encrypted updates that 
        only reveal aggregate. Here we simulate by returning aggregate only.
        """
        return simulate_secure_aggregation(client_updates)


def test_secure_aggregation():
    """Test secure aggregation simulation"""
    print("Testing Secure Aggregation Simulation...")
    
    # Create multiple dummy updates
    updates = []
    for i in range(5):
        update = {
            'update': {
                'param1': torch.randn(10, 5) * (i + 1),  # Different distributions
                'param2': torch.randn(10) * (i + 1)
            },
            'client_id': i,
            'has_elderly': i % 2,
            'num_samples': 100
        }
        updates.append(update)
    
    print(f"\nOriginal: {len(updates)} individual updates")
    for i, u in enumerate(updates):
        print(f"  Client {i}: has_elderly={u['has_elderly']}")
    
    # Apply secure aggregation
    secure_updates = simulate_secure_aggregation(updates)
    
    print(f"\nAfter Secure Aggregation: {len(secure_updates)} aggregate update")
    print(f"  Aggregated from {len(updates)} clients")
    print(f"  Individual properties hidden from adversary")
    
    # Show that individual info is lost
    print("\n  Original update 0 mean: ", updates[0]['update']['param1'].mean().item())
    print("  Aggregated update mean:", secure_updates[0]['update']['param1'].mean().item())
    print("  (Different - individual contributions masked)")


if __name__ == '__main__':
    test_secure_aggregation()
