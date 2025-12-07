"""
Federated Learning Server
"""
import torch
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from federated_learning.model import create_model


class FLServer:
    """Central server for federated learning"""
    
    def __init__(self):
        self.global_model = create_model()
        self.round_number = 0
        self.history = {
            'rounds': [],
            'global_loss': [],
            'participating_clients': []
        }
    
    def get_global_weights(self):
        """Get current global model weights"""
        return self.global_model.get_weights()
    
    def select_clients(self, all_clients):
        """Randomly select subset of clients for this round"""
        num_selected = max(1, int(len(all_clients) * config.CLIENT_PARTICIPATION_RATE))
        selected_indices = np.random.choice(
            len(all_clients), 
            size=num_selected, 
            replace=False
        )
        return [all_clients[i] for i in selected_indices]
    
    def aggregate_updates(self, client_updates):
        """
        Aggregate client updates using FedAvg (weighted averaging)
        
        Args:
            client_updates: List of dictionaries with 'update' and 'num_samples'
        
        Returns:
            Aggregated update dictionary
        """
        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in client_updates)
        
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
        
        return aggregated_update, client_updates  # Return both for attack
    
    def update_global_model(self, aggregated_update):
        """Apply aggregated update to global model"""
        current_weights = self.global_model.get_weights()
        
        for param_name in current_weights:
            current_weights[param_name] += aggregated_update[param_name]
        
        self.global_model.set_weights(current_weights)
    
    def federated_round(self, clients, return_updates=False):
        """
        Execute one round of federated learning
        
        Args:
            clients: List of all FLClient objects
            return_updates: If True, return individual client updates (for attack)
        
        Returns:
            Average loss, and optionally client updates
        """
        # Select clients
        selected_clients = self.select_clients(clients)
        
        if config.VERBOSE:
            print(f"\nRound {self.round_number + 1}: Selected {len(selected_clients)} clients")
        
        # Broadcast global weights
        global_weights = self.get_global_weights()
        for client in selected_clients:
            client.set_global_weights(global_weights)
        
        # Local training
        client_updates = []
        losses = []
        
        for client in selected_clients:
            loss = client.train()
            update_info = client.get_update()
            
            client_updates.append(update_info)
            losses.append(loss)
        
        # Aggregate updates
        aggregated_update, raw_updates = self.aggregate_updates(client_updates)
        
        # Update global model
        self.update_global_model(aggregated_update)
        
        # Record history
        avg_loss = np.mean(losses)
        self.history['rounds'].append(self.round_number)
        self.history['global_loss'].append(avg_loss)
        self.history['participating_clients'].append([c.client_id for c in selected_clients])
        
        self.round_number += 1
        
        if config.VERBOSE:
            print(f"Average loss: {avg_loss:.4f}")
        
        if return_updates:
            return avg_loss, raw_updates
        return avg_loss
    
    def evaluate_global_model(self, test_clients):
        """Evaluate global model on test clients"""
        global_weights = self.get_global_weights()
        losses = []
        
        for client in test_clients:
            client.set_global_weights(global_weights)
            loss = client.evaluate()
            losses.append(loss)
        
        avg_loss = np.mean(losses)
        return avg_loss
    
    def train(self, clients, num_rounds=config.FL_ROUNDS, return_all_updates=False):
        """
        Run complete federated learning training
        
        Args:
            clients: List of all FLClient objects
            num_rounds: Number of FL rounds
            return_all_updates: If True, collect all updates for attack training
        
        Returns:
            Training history, and optionally all collected updates
        """
        all_updates = [] if return_all_updates else None
        
        print(f"\n{'='*60}")
        print(f"Starting Federated Learning Training")
        print(f"Total clients: {len(clients)}")
        print(f"Rounds: {num_rounds}")
        print(f"{'='*60}")
        
        for round_num in tqdm(range(num_rounds), desc="FL Rounds"):
            if return_all_updates:
                _, updates = self.federated_round(clients, return_updates=True)
                all_updates.extend(updates)
            else:
                self.federated_round(clients, return_updates=False)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Final average loss: {self.history['global_loss'][-1]:.4f}")
        print(f"{'='*60}\n")
        
        if return_all_updates:
            return self.history, all_updates
        return self.history


def test_server():
    """Test server functionality"""
    from federated_learning.client import FLClient
    import numpy as np
    
    print("Testing FLServer...")
    
    # Create dummy clients
    clients = []
    for i in range(5):
        X = np.random.randn(100, config.INPUT_DIM)
        y = np.random.randn(100, 1)
        client = FLClient(client_id=i, X=X, y=y, has_elderly=i % 2)
        clients.append(client)
    
    print(f"Created {len(clients)} clients")
    
    # Create server
    server = FLServer()
    print("Created FL server")
    
    # Run one round
    print("\nRunning one FL round...")
    loss = server.federated_round(clients)
    print(f"Round loss: {loss:.4f}")
    
    return server


if __name__ == '__main__':
    server = test_server()
