import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm.auto import tqdm
from collections import defaultdict
from typing import Dict, Any, List

from model import GAT, HeteroGAT, HRM
from data_loader import (
    get_homogeneous_loaders,
    get_queens_loaders
)
from solver import Solver
from evaluation_util import evaluate_full_puzzle_capability

CHECKPOINT_BASE_DIR = 'checkpoints/comparison'
RESULTS_DIR = 'results'

DATA_PATHS = {
    'val_data': 'data/StateTrainingSet.json',
    'full_solve_data': 'data/StateValSet.json',
}

MODELS_TO_COMPARE = ['gat', 'hetero_gat', 'hrm']
VAL_RATIO = 0.10
SEED = 42
BATCH_SIZE = 256

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠ Using CPU")
    return device

def load_model_checkpoint(model_name: str, device):
    checkpoint_path = Path(CHECKPOINT_BASE_DIR) / model_name / 'best_model.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint['config_dict']
    if model_name == 'gat':
        model = GAT(
            input_dim=config_dict['input_dim'],
            hidden_dim=config_dict['hidden_dim'],
            layer_count=config_dict['layer_count'],
            dropout=config_dict['dropout'],
            heads=config_dict.get('gat_heads', 2)
        )
        is_heterogeneous = False
    elif model_name == 'hetero_gat':
        model = HeteroGAT(
            input_dim=config_dict['input_dim'],
            hidden_dim=config_dict['hidden_dim'],
            layer_count=config_dict['layer_count'],
            dropout=config_dict['dropout'],
            gat_heads=config_dict.get('gat_heads', 2),
            hgt_heads=config_dict.get('hgt_heads', 6),
            use_batch_norm=True
        )
        is_heterogeneous = True
    elif model_name == 'hrm':
        model = HRM(
            input_dim=config_dict['input_dim'],
            hidden_dim=config_dict['hidden_dim'],
            gat_heads=config_dict.get('gat_heads', 2),
            hgt_heads=config_dict.get('hgt_heads', 6),
            dropout=config_dict.get('dropout', 0.2),
            use_batch_norm=True,
            n_cycles=config_dict.get('n_cycles', 3),
            t_micro=config_dict.get('t_micro', 2),
            use_input_injection=config_dict.get('use_input_injection', True),
            z_init=config_dict.get('z_init', 'zeros'),
        )
        is_heterogeneous = True
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✓ Loaded {model_name.upper()} from {checkpoint_path}")
    return model, is_heterogeneous, config_dict

def evaluate_single_step(model, model_name: str, is_heterogeneous: bool, device) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Single-Step Evaluation: {model_name.upper()}")
    print(f"{'='*60}")
    if is_heterogeneous:
        _, val_loader = get_queens_loaders(
            DATA_PATHS['val_data'],
            batch_size=BATCH_SIZE,
            val_ratio=VAL_RATIO,
            seed=SEED,
            num_workers=0,
            pin_memory=True,
            shuffle_train=False
        )
    else:
        _, val_loader = get_homogeneous_loaders(
            DATA_PATHS['val_data'],
            batch_size=BATCH_SIZE,
            val_ratio=VAL_RATIO,
            seed=SEED,
            num_workers=0,
            pin_memory=True,
            shuffle_train=False
        )
    print(f"Validation samples: {len(val_loader.dataset)}")
    all_results = []
    total_graphs = 0
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = batch.to(device)
            if is_heterogeneous:
                x_dict = {'cell': batch['cell'].x}
                edge_index_dict = {
                    ('cell', 'line_constraint', 'cell'): batch[('cell', 'line_constraint', 'cell')].edge_index,
                    ('cell', 'region_constraint', 'cell'): batch[('cell', 'region_constraint', 'cell')].edge_index,
                    ('cell', 'diagonal_constraint', 'cell'): batch[('cell', 'diagonal_constraint', 'cell')].edge_index,
                }
                logits = model(x_dict, edge_index_dict)
                labels = batch['cell'].y
                batch_indices = get_batch_indices_hetero(batch, device)
                n_values = batch.n
            else:
                logits = model(batch.x, batch.edge_index)
                labels = batch.y
                batch_indices = batch.batch
                n_values = batch.n
            unique_batches = torch.unique(batch_indices)
            for i in unique_batches:
                mask = (batch_indices == i)
                graph_logits = logits[mask]
                graph_labels = labels[mask]
                top_k_indices = torch.topk(graph_logits, k=min(3, len(graph_logits)), largest=True).indices
                is_top1 = graph_labels[top_k_indices[0]].item() == 1
                is_top2 = (graph_labels[top_k_indices[:min(2, len(top_k_indices))]].sum().item() > 0)
                is_top3 = (graph_labels[top_k_indices].sum().item() > 0)
                top1_correct += int(is_top1)
                top2_correct += int(is_top2)
                top3_correct += int(is_top3)
                total_graphs += 1
    top1_acc = top1_correct / total_graphs if total_graphs > 0 else 0
    top2_acc = top2_correct / total_graphs if total_graphs > 0 else 0
    top3_acc = top3_correct / total_graphs if total_graphs > 0 else 0
    results = {
        'total_graphs': total_graphs,
        'top1_accuracy': top1_acc,
        'top2_accuracy': top2_acc,
        'top3_accuracy': top3_acc,
    }
    print(f"\nResults:")
    print(f"  Top-1 Accuracy: {top1_acc:.1%}")
    print(f"  Top-2 Accuracy: {top2_acc:.1%}")
    print(f"  Top-3 Accuracy: {top3_acc:.1%}")
    return results

def get_batch_indices_hetero(batch, device):
    if hasattr(batch, 'batch_dict') and 'cell' in batch.batch_dict:
        return batch.batch_dict['cell']
    elif hasattr(batch, '_slice_dict') and 'cell' in batch._slice_dict:
        slices = batch._slice_dict['cell']['x']
        batch_indices = torch.zeros(len(batch['cell'].y), dtype=torch.long, device=device)
        for i in range(len(slices) - 1):
            batch_indices[slices[i]:slices[i+1]] = i
        return batch_indices
    else:
        return torch.zeros(len(batch['cell'].y), dtype=torch.long, device=device)

def evaluate_full_solve_wrapper(checkpoint_path: str, data_path: str, device) -> Dict[str, Any]:
    print(f"\nLoading solver from {checkpoint_path}...")
    solver = Solver(str(checkpoint_path), device=str(device))
    stats = evaluate_full_puzzle_capability(
        solver,
        data_path,
        verbose=True
    )
    return stats

def plot_comparison(all_results: Dict[str, Dict], save_path: Path):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    model_names = list(all_results.keys())
    display_names = {'gat': 'GAT\n(Homogeneous)', 'hetero_gat': 'HeteroGAT', 'hrm': 'HRM'}
    colors = {'gat': '#1f77b4', 'hetero_gat': '#ff7f0e', 'hrm': '#2ca02c'}
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(model_names))
    width = 0.25
    top1_vals = [all_results[m]['single_step']['top1_accuracy'] * 100 for m in model_names]
    top2_vals = [all_results[m]['single_step']['top2_accuracy'] * 100 for m in model_names]
    top3_vals = [all_results[m]['single_step']['top3_accuracy'] * 100 for m in model_names]
    ax1.bar(x - width, top1_vals, width, label='Top-1', color='#d62728', alpha=0.8)
    ax1.bar(x, top2_vals, width, label='Top-2', color='#ff7f0e', alpha=0.8)
    ax1.bar(x + width, top3_vals, width, label='Top-3', color='#2ca02c', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Single-Step Prediction Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([display_names[m] for m in model_names])
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 105])
    for i, (t1, t2, t3) in enumerate(zip(top1_vals, top2_vals, top3_vals)):
        ax1.text(i - width, t1 + 2, f'{t1:.1f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i, t2 + 2, f'{t2:.1f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width, t3 + 2, f'{t3:.1f}', ha='center', va='bottom', fontsize=9)
    ax2 = fig.add_subplot(gs[0, 1])
    solve_rates = [all_results[m]['full_solve']['success_rate'] * 100 for m in model_names]
    bars = ax2.bar([display_names[m] for m in model_names], solve_rates,
                   color=[colors[m] for m in model_names], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Full Puzzle Perfect Solve Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 105])
    for bar, rate in zip(bars, solve_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3 = fig.add_subplot(gs[0, 2])
    error_data = []
    for m in model_names:
        error_by_step = all_results[m]['full_solve'].get('error_by_step', {})
        if error_by_step:
            total_errors = sum(error_by_step.values())
            error_pcts = {step: count/total_errors*100 for step, count in error_by_step.items()}
            error_data.append(error_pcts)
        else:
            error_data.append({})
    if any(error_data):
        all_steps = sorted(set(step for ed in error_data for step in ed.keys()))
        x_pos = np.arange(len(model_names))
        bottom = np.zeros(len(model_names))
        step_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(all_steps)))
        for step_idx, step in enumerate(all_steps):
            values = [error_data[i].get(step, 0) for i in range(len(model_names))]
            ax3.bar(x_pos, values, bottom=bottom, label=f'Step {step}',
                   color=step_colors[step_idx], alpha=0.8, edgecolor='white', linewidth=0.5)
            bottom += values
        ax3.set_ylabel('Error Distribution (%)', fontsize=11, fontweight='bold')
        ax3.set_title('First Error by Step', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([display_names[m] for m in model_names])
        ax3.legend(loc='upper right', fontsize=9, ncol=2)
        ax3.grid(True, alpha=0.3, axis='y')
    ax4 = fig.add_subplot(gs[1, :])
    board_sizes = sorted(set(
        size for m in model_names
        for size in all_results[m]['full_solve'].get('error_by_board_size', {}).keys()
    ))
    x_pos = np.arange(len(board_sizes))
    width = 0.25
    for i, model in enumerate(model_names):
        success_rates = []
        for size in board_sizes:
            size_data = all_results[model]['full_solve'].get('error_by_board_size', {}).get(size, {})
            if size_data:
                total = size_data['total']
                errors = size_data['errors']
                success = (total - errors) / total * 100 if total > 0 else 0
                success_rates.append(success)
            else:
                success_rates.append(0)
        offset = (i - 1) * width
        ax4.bar(x_pos + offset, success_rates, width, label=display_names[model],
               color=colors[model], alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_xlabel('Board Size', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Full Solve Success Rate by Board Size', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{size}×{size}' for size in board_sizes])
    ax4.legend(loc='lower left', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 105])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plots to {save_path}")
    plt.close()

def save_text_summary(all_results: Dict[str, Dict], save_path: Path):
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("QUEENS PUZZLE MODEL COMPARISON SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write("Models Evaluated:\n")
        f.write("  1. GAT (Homogeneous Graph Attention Network)\n")
        f.write("  2. HeteroGAT (Heterogeneous GAT with constraint-specific edges)\n")
        f.write("  3. HRM (Hierarchical Reasoning Model)\n\n")
        f.write("="*70 + "\n")
        f.write("SINGLE-STEP PREDICTION ACCURACY\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Model':<15} {'Top-1':<10} {'Top-2':<10} {'Top-3':<10}\n")
        f.write("-" * 45 + "\n")
        for model_name in ['gat', 'hetero_gat', 'hrm']:
            ss = all_results[model_name]['single_step']
            f.write(f"{model_name.upper():<15} "
                   f"{ss['top1_accuracy']:>8.1%}  "
                   f"{ss['top2_accuracy']:>8.1%}  "
                   f"{ss['top3_accuracy']:>8.1%}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("FULL PUZZLE SOLVING (STATE-0 PUZZLES)\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Model':<15} {'Success Rate':<15} {'Solved':<15} {'Failed':<10}\n")
        f.write("-" * 55 + "\n")
        for model_name in ['gat', 'hetero_gat', 'hrm']:
            fs = all_results[model_name]['full_solve']
            f.write(f"{model_name.upper():<15} "
                   f"{fs['success_rate']:>13.1%}  "
                   f"{fs['successful_solves']:>6}/{fs['total_puzzles']:<6} "
                   f"{fs['failed_solves']:>8}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("PERFORMANCE BY BOARD SIZE\n")
        f.write("="*70 + "\n\n")
        all_sizes = sorted(set(
            size for model_name in ['gat', 'hetero_gat', 'hrm']
            for size in all_results[model_name]['full_solve'].get('error_by_board_size', {}).keys()
        ))
        for size in all_sizes:
            f.write(f"\nBoard Size: {size}×{size}\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Model':<15} {'Success':<20} {'Rate':<10}\n")
            for model_name in ['gat', 'hetero_gat', 'hrm']:
                size_data = all_results[model_name]['full_solve'].get('error_by_board_size', {}).get(size, {})
                if size_data:
                    total = size_data['total']
                    errors = size_data['errors']
                    success = total - errors
                    rate = success / total if total > 0 else 0
                    f.write(f"{model_name.upper():<15} {success:>3}/{total:<15} {rate:>8.1%}\n")
    print(f"✓ Saved text summary to {save_path}")

def main():
    print("\n" + "="*70)
    print("QUEENS PUZZLE - MODEL COMPARISON EVALUATION")
    print("="*70)
    print(f"Models: {', '.join([m.upper() for m in MODELS_TO_COMPARE])}")
    print("="*70 + "\n")
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    device = get_device()
    all_results = {}
    for model_name in MODELS_TO_COMPARE:
        print(f"\n{'#'*70}")
        print(f"# Evaluating: {model_name.upper()}")
        print(f"{'#'*70}\n")
        results = {}
        try:
            model, is_heterogeneous, config_dict = load_model_checkpoint(model_name, device)
            single_step_results = evaluate_single_step(model, model_name, is_heterogeneous, device)
            results['single_step'] = single_step_results
            checkpoint_path = Path(CHECKPOINT_BASE_DIR) / model_name / 'best_model.pt'
            full_solve_results = evaluate_full_solve_wrapper(
                str(checkpoint_path),
                DATA_PATHS['full_solve_data'],
                device
            )
            results['full_solve'] = full_solve_results
            all_results[model_name] = results
        except Exception as e:
            print(f"⚠ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    if all_results:
        json_path = Path(RESULTS_DIR) / 'comparison_report.json'
        with open(json_path, 'w') as f:
            serializable_results = {}
            for model_name, results in all_results.items():
                serializable_results[model_name] = {
                    'single_step': results['single_step'],
                    'full_solve': {
                        k: (dict(v) if isinstance(v, defaultdict) else v)
                        for k, v in results['full_solve'].items()
                    }
                }
                json.dump(serializable_results, f, indent=2, default=str)
        print(f"\n✓ Saved JSON report to {json_path}")
        plot_path = Path(RESULTS_DIR) / 'comparison_plots.png'
        plot_comparison(all_results, plot_path)
        summary_path = Path(RESULTS_DIR) / 'comparison_summary.txt'
        save_text_summary(all_results, summary_path)
        print("\n" + "="*70)
        print("COMPARISON COMPLETE!")
        print("="*70)
        print(f"\nResults saved to:")
        print(f"  - {json_path}")
        print(f"  - {plot_path}")
        print(f"  - {summary_path}")
        print("\n" + "="*70)

if __name__ == "__main__":
    main()
