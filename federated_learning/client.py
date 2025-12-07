"""
Federated Learning Client
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from federated_learning.model import create_model


class FLClient:
    """Represents a single client (smart home) in federated learning"""
    
    def __init__(self, client_id, X, y, has_elderly):
        self.client_id = client_id
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.has_elderly = has_elderly
        
        # Create data loader
        dataset = TensorDataset(self.X, self.y)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True
        )
        
        # Initialize local model
        self.model = create_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE
        )
        
        # Store initial weights for update calculation
        self.initial_weights = None
    
    def set_global_weights(self, global_weights):
        """Receive global model weights from server"""
        self.model.set_weights(global_weights)
        self.initial_weights = {k: v.clone() for k, v in global_weights.items()}
    
    def train(self, epochs=config.LOCAL_EPOCHS):
        """Train model locally for specified epochs"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in self.dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss / len(self.dataloader)
        
        avg_loss = total_loss / epochs
        return avg_loss
    
    def get_update(self):
        """
        Calculate model update (difference between trained and initial weights)
        Returns: update dictionary and property label
        """
        current_weights = self.model.get_weights()
        
        update = {}
        for name in current_weights:
            update[name] = current_weights[name] - self.initial_weights[name]
        
        return {
            'client_id': self.client_id,
            'update': update,
            'has_elderly': self.has_elderly,
            'num_samples': len(self.X)
        }
    
    def evaluate(self):
        """Evaluate model on local data"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.dataloader:
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.dataloader)
        return avg_loss


def test_client():
    """Test client functionality"""
    import numpy as np
    
    print("Testing FLClient...")
    
    # Create dummy data
    X = np.random.randn(100, config.INPUT_DIM)
    y = np.random.randn(100, 1)
    has_elderly = 1
    
    # Create client
    client = FLClient(client_id=0, X=X, y=y, has_elderly=has_elderly)
    print(f"Created client {client.client_id} with {len(X)} samples")
    print(f"Has elderly: {client.has_elderly}")
    
    # Set initial weights
    initial_weights = client.model.get_weights()
    client.set_global_weights(initial_weights)
    
    # Train
    print("\nTraining for 5 epochs...")
    loss = client.train(epochs=5)
    print(f"Average training loss: {loss:.4f}")
    
    # Get update
    update_info = client.get_update()
    print(f"\nUpdate contains {len(update_info['update'])} parameter sets")
    
    # Evaluate
    eval_loss = client.evaluate()
    print(f"Evaluation loss: {eval_loss:.4f}")
    
    return client


if __name__ == '__main__':
    client = test_client()
