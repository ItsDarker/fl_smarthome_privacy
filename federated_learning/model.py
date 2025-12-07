"""
Neural network model for energy prediction
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EnergyPredictionModel(nn.Module):
    """Simple feedforward network for predicting energy consumption"""
    
    def __init__(self):
        super(EnergyPredictionModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, config.OUTPUT_DIM)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def get_weights(self):
        """Get model weights as a dictionary"""
        return {name: param.data.clone() for name, param in self.named_parameters()}
    
    def set_weights(self, weights):
        """Set model weights from a dictionary"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.copy_(weights[name])
    
    def get_gradients(self):
        """Get gradients as a dictionary"""
        return {name: param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                for name, param in self.named_parameters()}


def create_model():
    """Factory function to create a new model instance"""
    return EnergyPredictionModel()


def test_model():
    """Test model creation and forward pass"""
    model = create_model()
    print(f"Model architecture:\n{model}")
    
    # Test forward pass
    dummy_input = torch.randn(config.BATCH_SIZE, config.INPUT_DIM)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    
    return model


if __name__ == '__main__':
    print("Testing model creation...")
    model = test_model()
