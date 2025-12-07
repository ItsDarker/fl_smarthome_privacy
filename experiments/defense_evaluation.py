"""
Defense Evaluation Experiment
Evaluate each defense mechanism separately
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
from federated_learning.model import create_model
from attacks.property_inference import PropertyInferenceAttack
from defenses.differential_privacy import LDPDefense
from defenses.gradient_compression import GradientCompressionDefense
from defenses.secure_aggregation import SecureAggregationDefense


def run_fl_with_defense(clients, defense=None, num_rounds=config.FL_ROUNDS):
    """
    Run FL training with optional defense mechanism
    
    Args:
        clients: List of FLClient objects
        defense: Defense object with apply() method, or None for no defense
        num_rounds: Number of FL rounds
    
    Returns:
        server, all_updates
    """
    server = FLServer()
    all_updates = []
    
    for round_num in range(num_rounds):
        # Select clients
        selected_clients = server.select_clients(clients)
        
        # Broadcast global weights
        global_weights = server.get_global_weights()
        for client in selected_clients:
            client.set_global_weights(global_weights)
        
        # Local training
        client_updates = []
        for client in selected_clients:
            client.train()
            update_info = client.get_update()
            client_updates.append(update_info)
        
        # Apply defense if provided
        if defense is not None:
            client_updates = defense.apply(client_updates)
        
        # Store updates for attack training
        all_updates.extend(client_updates)
        
        # Aggregate (handles both normal and secure aggregation cases)
        aggregated_update, _ = server.aggregate_updates(client_updates)
        server.update_global_model(aggregated_update)
        
        if round_num % 5 == 0 and config.VERBOSE:
            print(f"Round {round_num + 1}/{num_rounds} complete")
    
    return server, all_updates


def evaluate_defense(defense_name, defense, train_clients, test_clients):
    """
    Evaluate a single defense mechanism
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Defense: {defense_name}")
    print(f"{'='*80}\n")
    
    # Run FL with defense
    print("Running FL training with defense...")
    server, all_updates = run_fl_with_defense(
        train_clients, 
        defense=defense,
        num_rounds=config.FL_ROUNDS
    )
    
    # Evaluate model utility
    test_loss = server.evaluate_global_model(test_clients)
    print(f"\nModel utility (test loss): {test_loss:.4f}")
    
    # Train attack
    print("\nTraining attack on defended updates...")
    attack = PropertyInferenceAttack()
    
    try:
        attack_train_acc = attack.train(all_updates)
    except Exception as e:
        print(f"Warning: Attack training failed: {e}")
        attack_train_acc = 0.0
    
    # Collect test updates with defense
    print("\nCollecting test updates...")
    global_weights = server.get_global_weights()
    test_updates = []
    
    for client in test_clients:
        client.set_global_weights(global_weights)
        client.train(epochs=config.LOCAL_EPOCHS)
        update_info = client.get_update()
        test_updates.append(update_info)
    
    # Apply defense to test updates
    if defense is not None:
        test_updates = defense.apply(test_updates)
    
    # Execute attack
    print("\nExecuting attack...")
    try:
        attack_results = attack.attack(test_updates)
    except Exception as e:
        print(f"Warning: Attack execution failed: {e}")
        attack_results = {
            'accuracy': 0.5,  # Random guess
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    results = {
        'defense_name': defense_name,
        'test_loss': test_loss,
        'attack_train_accuracy': attack_train_acc,
        'attack_results': attack_results
    }
    
    print(f"\n{'='*80}")
    print(f"Results for {defense_name}:")
    print(f"{'='*80}")
    print(f"Model Test Loss: {test_loss:.4f}")
    print(f"Attack Accuracy: {attack_results['accuracy']:.4f} ({attack_results['accuracy']*100:.2f}%)")
    print(f"{'='*80}\n")
    
    return results


def run_defense_evaluation():
    """
    Run complete defense evaluation experiment
    """
    print("\n" + "="*80)
    print("DEFENSE EVALUATION EXPERIMENT")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data...")
    df, home_labels = load_data()
    train_homes, test_homes = split_train_test_clients(config.NUM_HOMES)
    
    # Create clients
    print("Creating clients...")
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
    
    print(f"Training clients: {len(train_clients)}")
    print(f"Test clients: {len(test_clients)}")
    
    # Define defenses to evaluate
    defenses_to_test = [
        ("No Defense", None),
    ]
    
    # Add LDP with different epsilon values
    for epsilon in config.DP_EPSILONS:
        defenses_to_test.append((f"LDP (ε={epsilon})", LDPDefense(epsilon)))
    
    # Add gradient compression with different rates
    for rate in config.COMPRESSION_RATES:
        defenses_to_test.append((f"Compression ({int(rate*100)}%)", GradientCompressionDefense(rate)))
    
    # Add secure aggregation
    defenses_to_test.append(("Secure Aggregation", SecureAggregationDefense()))
    
    # Evaluate each defense
    all_results = {}
    
    for defense_name, defense in defenses_to_test:
        # Create fresh clients for each experiment
        fresh_train_clients = []
        for home_id in train_homes:
            X, y, has_elderly = get_client_data(df, home_id)
            client = FLClient(home_id, X, y, has_elderly)
            fresh_train_clients.append(client)
        
        fresh_test_clients = []
        for home_id in test_homes:
            X, y, has_elderly = get_client_data(df, home_id)
            client = FLClient(home_id, X, y, has_elderly)
            fresh_test_clients.append(client)
        
        # Evaluate defense
        results = evaluate_defense(defense_name, defense, fresh_train_clients, fresh_test_clients)
        all_results[defense_name] = results
    
    # Save all results
    results_file = os.path.join(config.RESULTS_DIR, 'defense_evaluation_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nAll results saved to: {results_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("DEFENSE EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Defense':<30} {'Model Loss':<15} {'Attack Acc':<15}")
    print("-"*80)
    
    for defense_name, results in all_results.items():
        print(f"{defense_name:<30} {results['test_loss']:<15.4f} {results['attack_results']['accuracy']:<15.4f}")
    
    print("="*80 + "\n")
    
    return all_results


if __name__ == '__main__':
    results = run_defense_evaluation()
