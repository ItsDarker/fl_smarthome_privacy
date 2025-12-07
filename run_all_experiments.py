#!/usr/bin/env python3
"""
Main script to run all experiments
Property Inference Attacks on Federated Learning in Smart Home IoT
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data.generate_data import generate_smart_home_data
from experiments.baseline_attack import run_baseline_experiment
from experiments.defense_evaluation import run_defense_evaluation
from visualization.plot_results import generate_all_plots
import time


def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║           Property Inference Attacks on Federated Learning                  ║
    ║                    in Smart Home IoT Networks                                ║
    ║                                                                              ║
    ║  Detection, Exploitation, and Privacy-Enhancing Countermeasures             ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Run complete experimental pipeline"""
    start_time = time.time()
    
    print_banner()
    
    print("\n" + "="*80)
    print("EXPERIMENTAL PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Homes: {config.NUM_HOMES}")
    print(f"  FL Rounds: {config.FL_ROUNDS}")
    print(f"  Local Epochs: {config.LOCAL_EPOCHS}")
    print(f"  Results Directory: {config.RESULTS_DIR}")
    print("="*80 + "\n")
    
    # Step 1: Generate data
    print("\n[STEP 1/4] GENERATING DATA")
    print("-" * 80)
    generate_smart_home_data()
    
    # Step 2: Run baseline attack experiment
    print("\n\n[STEP 2/4] BASELINE ATTACK EXPERIMENT")
    print("-" * 80)
    baseline_results = run_baseline_experiment()
    
    # Step 3: Run defense evaluation
    print("\n\n[STEP 3/4] DEFENSE EVALUATION EXPERIMENTS")
    print("-" * 80)
    defense_results = run_defense_evaluation()
    
    # Step 4: Generate visualizations
    print("\n\n[STEP 4/4] GENERATING VISUALIZATIONS")
    print("-" * 80)
    generate_all_plots()
    
    # Final summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("EXPERIMENTAL PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nTotal Execution Time: {elapsed_time/60:.2f} minutes")
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print("\nGenerated files:")
    print(f"  - baseline_results.pkl")
    print(f"  - defense_evaluation_results.pkl")
    print(f"  - baseline_results.png")
    print(f"  - defense_comparison.png")
    print(f"  - ldp_tradeoff.png")
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    # Print key results
    baseline_acc = baseline_results['attack_results']['accuracy']
    baseline_loss = baseline_results['test_loss']
    
    print(f"\nBaseline (No Defense):")
    print(f"  - Model Loss: {baseline_loss:.4f}")
    print(f"  - Attack Accuracy: {baseline_acc:.2%}")
    
    # Find best defense
    best_defense = None
    best_privacy_gain = 0
    acceptable_utility_loss = 0.15  # 15% utility degradation threshold
    
    for name, results in defense_results.items():
        if name == 'No Defense':
            continue
        
        attack_acc = results['attack_results']['accuracy']
        model_loss = results['test_loss']
        
        privacy_gain = (baseline_acc - attack_acc) / baseline_acc
        utility_loss = (model_loss - baseline_loss) / baseline_loss
        
        # Find defense with best privacy gain and acceptable utility loss
        if privacy_gain > best_privacy_gain and utility_loss < acceptable_utility_loss:
            best_privacy_gain = privacy_gain
            best_defense = {
                'name': name,
                'attack_acc': attack_acc,
                'model_loss': model_loss,
                'privacy_gain': privacy_gain,
                'utility_loss': utility_loss
            }
    
    if best_defense:
        print(f"\nBest Defense: {best_defense['name']}")
        print(f"  - Model Loss: {best_defense['model_loss']:.4f} (+{best_defense['utility_loss']:.1%})")
        print(f"  - Attack Accuracy: {best_defense['attack_acc']:.2%} ({best_defense['privacy_gain']:.1%} reduction)")
        print(f"  - Privacy Gain: {best_defense['privacy_gain']:.1%}")
        print(f"  - Utility Cost: {best_defense['utility_loss']:.1%}")
    
    print("\n" + "="*80)
    print("For detailed analysis, check the visualization plots in the results/ directory")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
