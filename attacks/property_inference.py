"""
Property Inference Attack on Federated Learning
"""
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PropertyInferenceAttack:
    """
    Adversary that infers household properties from FL updates
    """
    
    def __init__(self):
        self.attack_model = RandomForestClassifier(
            n_estimators=config.RANDOM_FOREST_ESTIMATORS,
            max_depth=config.RANDOM_FOREST_MAX_DEPTH,
            min_samples_leaf=config.RANDOM_FOREST_MIN_SAMPLES_LEAF,
            random_state=config.RANDOM_SEED
        )
        self.is_trained = False
    
    def extract_features_from_update(self, update):
        """
        Extract statistical features from model update
        
        Args:
            update: Dictionary of parameter updates (tensors)
        
        Returns:
            Feature vector (numpy array)
        """
        features = []
        
        for param_name, param_values in update.items():
            # Convert to numpy
            values = param_values.cpu().numpy().flatten()
            
            # Extract statistics
            features.extend([
                np.mean(values),
                np.std(values),
                np.median(values),
                np.min(values),
                np.max(values),
                np.percentile(values, 25),
                np.percentile(values, 75),
                np.linalg.norm(values, ord=1),  # L1 norm
                np.linalg.norm(values, ord=2),  # L2 norm
            ])
        
        return np.array(features)
    
    def prepare_training_data(self, client_updates):
        """
        Prepare training data from collected client updates
        
        Args:
            client_updates: List of update dictionaries with 'update' and 'has_elderly'
        
        Returns:
            X (features), y (labels)
        """
        X = []
        y = []
        
        print(f"Extracting features from {len(client_updates)} updates...")
        
        for update_info in client_updates:
            features = self.extract_features_from_update(update_info['update'])
            X.append(features)
            y.append(update_info['has_elderly'])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label distribution: {np.bincount(y)}")
        
        return X, y
    
    def train(self, client_updates):
        """
        Train attack model on collected updates
        
        Args:
            client_updates: List of update dictionaries from FL training
        
        Returns:
            Training accuracy
        """
        print("\n" + "="*60)
        print("Training Property Inference Attack Model")
        print("="*60)
        
        # Prepare data
        X, y = self.prepare_training_data(client_updates)
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=1-config.ATTACK_TRAIN_SPLIT,
            random_state=config.RANDOM_SEED,
            stratify=y
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Train attack model
        print("\nTraining Random Forest classifier...")
        self.attack_model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        y_pred = self.attack_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"\nValidation Accuracy: {accuracy:.4f}")
        print("="*60 + "\n")
        
        return accuracy
    
    def attack(self, target_updates):
        """
        Perform attack on target updates
        
        Args:
            target_updates: List of update dictionaries to attack
        
        Returns:
            Dictionary with attack results
        """
        if not self.is_trained:
            raise ValueError("Attack model must be trained first!")
        
        print("\n" + "="*60)
        print("Executing Property Inference Attack")
        print("="*60)
        
        # Prepare features
        X_test, y_true = self.prepare_training_data(target_updates)
        
        print(f"\nAttacking {len(X_test)} targets...")
        
        # Predict
        y_pred = self.attack_model.predict(X_test)
        y_proba = self.attack_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_proba,
            'true_labels': y_true
        }
        
        print(f"\n{'='*60}")
        print("Attack Results:")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        print(f"{'='*60}\n")
        
        return results
    
    def save_model(self, filepath):
        """Save trained attack model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.attack_model, f)
        print(f"Attack model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load trained attack model"""
        with open(filepath, 'rb') as f:
            self.attack_model = pickle.load(f)
        self.is_trained = True
        print(f"Attack model loaded from: {filepath}")


def test_attack():
    """Test attack functionality"""
    print("Testing PropertyInferenceAttack...")
    
    # Create dummy updates
    dummy_updates = []
    for i in range(50):
        update = {
            'update': {
                'param1': torch.randn(16, 2),
                'param2': torch.randn(16),
                'param3': torch.randn(1, 16)
            },
            'has_elderly': i % 2
        }
        dummy_updates.append(update)
    
    # Create and train attack
    attack = PropertyInferenceAttack()
    accuracy = attack.train(dummy_updates)
    print(f"Attack training accuracy: {accuracy:.4f}")
    
    # Test attack
    test_updates = dummy_updates[:10]
    results = attack.attack(test_updates)
    
    return attack, results


if __name__ == '__main__':
    attack, results = test_attack()
