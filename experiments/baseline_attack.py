"""
Baseline Attack Experiment (No Defense)
"""
import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.generate_data import load_data, get_client_data, split_train_test_clients
from federated_learning.client import FLClient
from federated_learning.server import FLServer
from attacks.property_inference import PropertyInferenceAttack


def run_baseline_experiment():
    """
    Run baseline experiment: FL training + property inference attack with no defense
    """
    print("\n" + "="*80)
    print("BASELINE EXPERIMENT: Property Inference Attack (No Defense)")
    print("="*80 + "\n")
    
    # Load data
    print("Loading smart home data...")
    df, home_labels = load_data()
    
    # Split into train/test clients
    train_homes, test_homes = split_train_test_clients(config.NUM_HOMES)
    print(f"Training clients: {len(train_homes)}")
    print(f"Test clients: {len(test_homes)}")
    
    # Create FL clients
    print("\nCreating FL clients...")
    train_clients = []
    for home_id in train_homes:
        X, y, has_elderly = get_client_data(df, home_id)
        client = FLClient(home_id, X, y, has_elderly)
        train_clients.append(client)
    
    test_clients = []
    for home_id in test_homes:
        X, y, has_elderly = get_client_data(df, home_id)
        client = FLClient(home_id, X, y, has_elderly)
        test_clients.append(client)
    
    print(f"Created {len(train_clients)} training clients")
    print(f"Created {len(test_clients)} test clients")
    
    # Run FL training and collect updates
    print("\nRunning Federated Learning...")
    server = FLServer()
    history, all_updates = server.train(
        train_clients, 
        num_rounds=config.FL_ROUNDS,
        return_all_updates=True
    )
    
    print(f"Collected {len(all_updates)} client updates during training")
    
    # Evaluate global model
    test_loss = server.evaluate_global_model(test_clients)
    print(f"Global model test loss: {test_loss:.4f}")
    
    # Train attack model
    print("\nTraining attack model on collected updates...")
    attack = PropertyInferenceAttack()
    attack_train_acc = attack.train(all_updates)
    
    # Collect updates from test clients for attack evaluation
    print("\nCollecting updates from test clients...")
    global_weights = server.get_global_weights()
    test_updates = []
    
    for client in test_clients:
        client.set_global_weights(global_weights)
        client.train(epochs=config.LOCAL_EPOCHS)
        update_info = client.get_update()
        test_updates.append(update_info)
    
    # Execute attack on test clients
    print("\nExecuting attack on test clients...")
    attack_results = attack.attack(test_updates)
    
    # Save results
    results = {
        'fl_history': history,
        'test_loss': test_loss,
        'attack_train_accuracy': attack_train_acc,
        'attack_results': attack_results,
        'num_train_clients': len(train_clients),
        'num_test_clients': len(test_clients),
        'num_updates_collected': len(all_updates)
    }
    
    results_file = os.path.join(config.RESULTS_DIR, 'baseline_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE EXPERIMENT SUMMARY")
    print("="*80)
    print(f"FL Training Rounds: {config.FL_ROUNDS}")
    print(f"Final Model Test Loss: {test_loss:.4f}")
    print(f"\nAttack Performance:")
    print(f"  Training Accuracy: {attack_train_acc:.4f} ({attack_train_acc*100:.2f}%)")
    print(f"  Test Accuracy: {attack_results['accuracy']:.4f} ({attack_results['accuracy']*100:.2f}%)")
    print(f"  Precision: {attack_results['precision']:.4f}")
    print(f"  Recall: {attack_results['recall']:.4f}")
    print(f"  F1-Score: {attack_results['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(attack_results['confusion_matrix'])
    print("="*80 + "\n")
    
    return results


if __name__ == '__main__':
    results = run_baseline_experiment()
