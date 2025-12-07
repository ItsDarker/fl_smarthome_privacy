"""
Visualization module for plotting experimental results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = config.FIGURE_SIZE
plt.rcParams['font.size'] = config.FONT_SIZE


def plot_baseline_results(results):
    """Plot baseline attack results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: FL training loss
    ax1 = axes[0]
    rounds = results['fl_history']['rounds']
    losses = results['fl_history']['global_loss']
    ax1.plot(rounds, losses, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('FL Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Global Model Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Federated Learning Training', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Attack confusion matrix
    ax2 = axes[1]
    cm = results['attack_results']['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['No Elderly', 'Has Elderly'],
                yticklabels=['No Elderly', 'Has Elderly'])
    ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax2.set_title(f'Attack Confusion Matrix\nAccuracy: {results["attack_results"]["accuracy"]:.2%}',
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(config.RESULTS_DIR, 'baseline_results.png')
    plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
    print(f"Saved baseline results plot: {filepath}")
    plt.close()


def plot_defense_comparison(results_dict):
    """Plot comparison of all defenses"""
    # Extract data
    defense_names = []
    attack_accs = []
    model_losses = []
    
    for name, results in results_dict.items():
        defense_names.append(name)
        attack_accs.append(results['attack_results']['accuracy'])
        model_losses.append(results['test_loss'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Attack accuracy comparison
    ax1 = axes[0]
    bars = ax1.bar(range(len(defense_names)), attack_accs, color='steelblue', alpha=0.7)
    
    # Highlight baseline and best defense
    baseline_idx = defense_names.index('No Defense')
    bars[baseline_idx].set_color('red')
    bars[baseline_idx].set_alpha(0.7)
    
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random Guess')
    ax1.set_ylabel('Attack Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Attack Accuracy by Defense', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(defense_names)))
    ax1.set_xticklabels(defense_names, rotation=45, ha='right')
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for i, (bar, acc) in enumerate(zip(bars, attack_accs)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Model utility comparison
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(defense_names)), model_losses, color='coral', alpha=0.7)
    bars2[baseline_idx].set_color('red')
    bars2[baseline_idx].set_alpha(0.7)
    
    ax2.set_ylabel('Model Test Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Model Utility by Defense', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(defense_names)))
    ax2.set_xticklabels(defense_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, loss in zip(bars2, model_losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(config.RESULTS_DIR, 'defense_comparison.png')
    plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
    print(f"Saved defense comparison plot: {filepath}")
    plt.close()


def plot_ldp_tradeoff(results_dict):
    """Plot privacy-utility tradeoff for LDP with different epsilon values"""
    # Extract LDP results
    ldp_results = {}
    for name, results in results_dict.items():
        if 'LDP' in name and name != 'No Defense':
            try:
                epsilon = float(name.split('ε=')[1].split(')')[0])
                ldp_results[epsilon] = results
            except:
                pass
    
    if len(ldp_results) == 0:
        print("No LDP results found for tradeoff plot")
        return
    
    # Sort by epsilon
    epsilons = sorted(ldp_results.keys())
    attack_accs = [ldp_results[eps]['attack_results']['accuracy'] for eps in epsilons]
    model_losses = [ldp_results[eps]['test_loss'] for eps in epsilons]
    
    # Add baseline (infinite epsilon)
    baseline = results_dict['No Defense']
    epsilons_plot = epsilons + [max(epsilons) * 2]  # Put baseline at end visually
    attack_accs_plot = attack_accs + [baseline['attack_results']['accuracy']]
    model_losses_plot = model_losses + [baseline['test_loss']]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot attack accuracy
    color = 'tab:red'
    ax1.set_xlabel('Privacy Budget (ε)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Attack Accuracy', color=color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(epsilons_plot, attack_accs_plot, 'o-', color=color, 
                     linewidth=2, markersize=8, label='Attack Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0.4, 1.0])
    ax1.grid(True, alpha=0.3)
    
    # Add baseline label
    ax1.axvline(x=epsilons_plot[-1], color='gray', linestyle='--', alpha=0.5)
    ax1.text(epsilons_plot[-1], 0.42, 'Baseline\n(No Defense)', 
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot model accuracy on second y-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Model Test Loss', color=color, fontsize=12, fontweight='bold')
    line2 = ax2.plot(epsilons_plot, model_losses_plot, 's-', color=color, 
                     linewidth=2, markersize=8, label='Model Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Set x-axis labels
    x_labels = [str(eps) for eps in epsilons] + ['∞']
    ax1.set_xticks(epsilons_plot)
    ax1.set_xticklabels(x_labels)
    
    # Title and legend
    plt.title('Privacy-Utility Tradeoff: Local Differential Privacy', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(config.RESULTS_DIR, 'ldp_tradeoff.png')
    plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
    print(f"Saved LDP tradeoff plot: {filepath}")
    plt.close()


def generate_all_plots():
    """Generate all visualization plots from saved results"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Load baseline results
    baseline_file = os.path.join(config.RESULTS_DIR, 'baseline_results.pkl')
    if os.path.exists(baseline_file):
        with open(baseline_file, 'rb') as f:
            baseline_results = pickle.load(f)
        plot_baseline_results(baseline_results)
    else:
        print(f"Baseline results not found: {baseline_file}")
    
    # Load defense evaluation results
    defense_file = os.path.join(config.RESULTS_DIR, 'defense_evaluation_results.pkl')
    if os.path.exists(defense_file):
        with open(defense_file, 'rb') as f:
            defense_results = pickle.load(f)
        plot_defense_comparison(defense_results)
        plot_ldp_tradeoff(defense_results)
    else:
        print(f"Defense evaluation results not found: {defense_file}")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print(f"All plots saved to: {config.RESULTS_DIR}")
    print("="*80 + "\n")


if __name__ == '__main__':
    generate_all_plots()
