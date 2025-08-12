import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
import json

from data_loader import get_queens_loaders
from model import GAT, HeteroGAT
from config import Config

def load_model(model_path: str, config: Config, use_heterogeneous: bool = True):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=config.system.device, weights_only=False)
    
    if use_heterogeneous:
        model = HeteroGAT(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            layer_count=config.model.layer_count,
            dropout=config.model.dropout,
            heads=config.model.heads
        )
    else:
        model = GAT(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            layer_count=config.model.layer_count,
            dropout=config.model.dropout,
            heads=config.model.heads
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.system.device)
    model.eval()
    
    print(f"Loaded {'Heterogeneous' if use_heterogeneous else 'Homogeneous'} GAT model")
    return model

def get_batch_indices_hetero(batch, device):
    """Extract batch indices for heterogeneous data."""
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

def run_predictions(model, val_loader, use_heterogeneous: bool, device: str):
    """Run model on validation set and collect predictions."""
    print("Running predictions on validation set...")
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = batch.to(device)
            
            # Get predictions based on model type
            if use_heterogeneous:
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
                step_values = batch.step
            else:
                logits = model(batch.x, batch.edge_index)
                labels = batch.y
                batch_indices = batch.batch
                n_values = batch.n
                step_values = batch.step
            
            probs = torch.sigmoid(logits)
            
            # Process each graph in the batch
            unique_batches = torch.unique(batch_indices)
            
            for graph_idx in unique_batches:
                mask = (batch_indices == graph_idx)
                graph_logits = logits[mask]
                graph_labels = labels[mask]
                graph_probs = probs[mask]
                
                # Get metadata
                board_size = int(n_values[graph_idx].item())
                step = int(step_values[graph_idx].item())
                queens_placed = board_size - step
                
                # Get top predictions
                sorted_indices = torch.argsort(graph_logits, descending=True)
                sorted_labels = graph_labels[sorted_indices]
                sorted_probs = graph_probs[sorted_indices]
                
                # Calculate metrics
                queen_positions = torch.where(graph_labels == 1)[0].cpu().numpy()
                queens_remaining = len(queen_positions)
                
                top1_correct = int(sorted_labels[0].item()) if len(sorted_labels) > 0 else 0
                top2_correct = int(torch.any(sorted_labels[:2] == 1).item()) if len(sorted_labels) > 1 else top1_correct
                top3_correct = int(torch.any(sorted_labels[:3] == 1).item()) if len(sorted_labels) > 2 else top2_correct
                
                # Convert positions to 2D coordinates
                top1_pos = sorted_indices[0].item()
                top1_row, top1_col = top1_pos // board_size, top1_pos % board_size
                
                results.append({
                    'board_size': board_size,
                    'step': step,
                    'queens_placed': queens_placed,
                    'queens_remaining': queens_remaining,
                    'top1_correct': top1_correct,
                    'top2_correct': top2_correct,
                    'top3_correct': top3_correct,
                    'top1_confidence': float(sorted_probs[0].item()),
                    'top1_row': top1_row,
                    'top1_col': top1_col,
                    'queen_positions': queen_positions.tolist()
                })
    
    print(f"Collected {len(results)} predictions")
    return results

def analyze_board_size_effects(results):
    """Analyze performance by board size."""
    print("Analyzing board size effects...")
    
    size_stats = []
    for size in sorted(set(r['board_size'] for r in results)):
        size_results = [r for r in results if r['board_size'] == size]
        
        stats = {
            'board_size': size,
            'count': len(size_results),
            'top1_accuracy': np.mean([r['top1_correct'] for r in size_results]),
            'top2_accuracy': np.mean([r['top2_correct'] for r in size_results]),
            'top3_accuracy': np.mean([r['top3_correct'] for r in size_results]),
            'avg_confidence': np.mean([r['top1_confidence'] for r in size_results])
        }
        size_stats.append(stats)
    
    return pd.DataFrame(size_stats)

def analyze_game_state_effects(results):
    """Analyze performance by game state (queens remaining)."""
    print("Analyzing game state effects...")
    
    state_stats = []
    for queens_remaining in sorted(set(r['queens_remaining'] for r in results)):
        state_results = [r for r in results if r['queens_remaining'] == queens_remaining]
        
        stats = {
            'queens_remaining': queens_remaining,
            'count': len(state_results),
            'top1_accuracy': np.mean([r['top1_correct'] for r in state_results]),
            'top2_accuracy': np.mean([r['top2_correct'] for r in state_results]),
            'top3_accuracy': np.mean([r['top3_correct'] for r in state_results]),
            'avg_confidence': np.mean([r['top1_confidence'] for r in state_results])
        }
        state_stats.append(stats)
    
    return pd.DataFrame(state_stats)

def analyze_spatial_patterns(results):
    """Analyze spatial patterns of predictions and errors."""
    print("Analyzing spatial patterns...")
    
    spatial_data = {}
    
    for board_size in sorted(set(r['board_size'] for r in results)):
        size_results = [r for r in results if r['board_size'] == board_size]
        
        # Create heatmaps
        prediction_map = np.zeros((board_size, board_size))
        error_map = np.zeros((board_size, board_size))
        
        for result in size_results:
            row, col = result['top1_row'], result['top1_col']
            prediction_map[row, col] += 1
            if not result['top1_correct']:
                error_map[row, col] += 1
        
        # Calculate error rates
        error_rate_map = np.divide(error_map, prediction_map, 
                                 out=np.zeros_like(error_map), 
                                 where=prediction_map!=0)
        
        # Analyze position types (center, edge, corner)
        center_errors, edge_errors, corner_errors = [], [], []
        
        for result in size_results:
            row, col = result['top1_row'], result['top1_col']
            is_correct = result['top1_correct']
            
            is_corner = (row in [0, board_size-1]) and (col in [0, board_size-1])
            is_edge = (row in [0, board_size-1]) or (col in [0, board_size-1])
            
            if is_corner:
                corner_errors.append(1 - is_correct)
            elif is_edge:
                edge_errors.append(1 - is_correct)
            else:
                center_errors.append(1 - is_correct)
        
        spatial_data[board_size] = {
            'prediction_map': prediction_map,
            'error_map': error_map,
            'error_rate_map': error_rate_map,
            'center_error_rate': np.mean(center_errors) if center_errors else 0,
            'edge_error_rate': np.mean(edge_errors) if edge_errors else 0,
            'corner_error_rate': np.mean(corner_errors) if corner_errors else 0
        }
    
    return spatial_data

def create_visualizations(results, size_df, state_df, spatial_data, output_dir="evaluation_plots"):
    """Create all visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Creating visualizations in {output_path}...")
    
    # 1. Board size analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance by Board Size', fontsize=16, fontweight='bold')
    
    # Top-k accuracy
    axes[0, 0].plot(size_df['board_size'], size_df['top1_accuracy'], 'o-', label='Top-1', linewidth=2)
    axes[0, 0].plot(size_df['board_size'], size_df['top2_accuracy'], 's-', label='Top-2', linewidth=2)
    axes[0, 0].plot(size_df['board_size'], size_df['top3_accuracy'], '^-', label='Top-3', linewidth=2)
    axes[0, 0].set_xlabel('Board Size')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Top-K Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sample count
    axes[0, 1].bar(size_df['board_size'], size_df['count'], alpha=0.7)
    axes[0, 1].set_xlabel('Board Size')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_title('Sample Distribution')
    
    # Confidence
    axes[1, 0].plot(size_df['board_size'], size_df['avg_confidence'], 'o-', color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Board Size')
    axes[1, 0].set_ylabel('Average Confidence')
    axes[1, 0].set_title('Model Confidence')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error rate
    error_rate = 1 - size_df['top1_accuracy']
    axes[1, 1].bar(size_df['board_size'], error_rate, alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Board Size')
    axes[1, 1].set_ylabel('Error Rate')
    axes[1, 1].set_title('Error Rate by Size')
    
    plt.tight_layout()
    plt.savefig(output_path / 'board_size_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Game state analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance by Game State', fontsize=16, fontweight='bold')
    
    # Accuracy by queens remaining
    axes[0, 0].plot(state_df['queens_remaining'], state_df['top1_accuracy'], 'o-', linewidth=2)
    axes[0, 0].set_xlabel('Queens Remaining')
    axes[0, 0].set_ylabel('Top-1 Accuracy')
    axes[0, 0].set_title('Accuracy by Queens Remaining')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sample distribution
    axes[0, 1].bar(state_df['queens_remaining'], state_df['count'], alpha=0.7)
    axes[0, 1].set_xlabel('Queens Remaining')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_title('Sample Distribution')
    
    # Confidence by game state
    axes[1, 0].plot(state_df['queens_remaining'], state_df['avg_confidence'], 'o-', color='green', linewidth=2)
    axes[1, 0].set_xlabel('Queens Remaining')
    axes[1, 0].set_ylabel('Average Confidence')
    axes[1, 0].set_title('Confidence by Game State')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Top-k comparison
    axes[1, 1].plot(state_df['queens_remaining'], state_df['top1_accuracy'], 'o-', label='Top-1')
    axes[1, 1].plot(state_df['queens_remaining'], state_df['top2_accuracy'], 's-', label='Top-2')
    axes[1, 1].plot(state_df['queens_remaining'], state_df['top3_accuracy'], '^-', label='Top-3')
    axes[1, 1].set_xlabel('Queens Remaining')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Top-K Accuracy by Game State')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'game_state_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Spatial patterns - error rate heatmaps
    n_sizes = len(spatial_data)
    if n_sizes > 0:
        cols = min(3, n_sizes)
        rows = (n_sizes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_sizes == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('Spatial Error Patterns', fontsize=16, fontweight='bold')
        
        for idx, (board_size, data) in enumerate(spatial_data.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            im = ax.imshow(data['error_rate_map'], cmap='Reds', vmin=0, vmax=1)
            ax.set_title(f'{board_size}x{board_size} Board')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Remove unused subplots
        for idx in range(n_sizes, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(output_path / 'spatial_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Position type analysis
    if spatial_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        board_sizes = list(spatial_data.keys())
        center_errors = [spatial_data[size]['center_error_rate'] for size in board_sizes]
        edge_errors = [spatial_data[size]['edge_error_rate'] for size in board_sizes]
        corner_errors = [spatial_data[size]['corner_error_rate'] for size in board_sizes]
        
        x = np.arange(len(board_sizes))
        width = 0.25
        
        ax1.bar(x - width, center_errors, width, label='Center', alpha=0.8)
        ax1.bar(x, edge_errors, width, label='Edge', alpha=0.8)
        ax1.bar(x + width, corner_errors, width, label='Corner', alpha=0.8)
        
        ax1.set_xlabel('Board Size')
        ax1.set_ylabel('Error Rate')
        ax1.set_title('Error Rate by Position Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{size}x{size}' for size in board_sizes])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Overall position analysis
        all_center = np.mean(center_errors)
        all_edge = np.mean(edge_errors)
        all_corner = np.mean(corner_errors)
        
        ax2.bar(['Center', 'Edge', 'Corner'], [all_center, all_edge, all_corner], 
               color=['blue', 'orange', 'red'], alpha=0.7)
        ax2.set_ylabel('Average Error Rate')
        ax2.set_title('Overall Error Rate by Position Type')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Position-Based Error Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'position_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Summary dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Evaluation Summary', fontsize=18, fontweight='bold')
    
    # Overall metrics
    overall_top1 = np.mean([r['top1_correct'] for r in results])
    overall_top2 = np.mean([r['top2_correct'] for r in results])
    overall_top3 = np.mean([r['top3_correct'] for r in results])
    
    axes[0, 0].bar(['Top-1', 'Top-2', 'Top-3'], [overall_top1, overall_top2, overall_top3])
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title(f'Overall Performance\n({len(results):,} predictions)')
    axes[0, 0].set_ylim(0, 1)
    
    # Performance by size
    axes[0, 1].plot(size_df['board_size'], size_df['top1_accuracy'], 'o-', linewidth=3)
    axes[0, 1].set_xlabel('Board Size')
    axes[0, 1].set_ylabel('Top-1 Accuracy')
    axes[0, 1].set_title('Performance by Board Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Performance by game state
    axes[0, 2].plot(state_df['queens_remaining'], state_df['top1_accuracy'], 'o-', 
                   color='orange', linewidth=3)
    axes[0, 2].set_xlabel('Queens Remaining')
    axes[0, 2].set_ylabel('Top-1 Accuracy')
    axes[0, 2].set_title('Performance by Game State')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Confidence distribution
    confidences = [r['top1_confidence'] for r in results]
    axes[1, 0].hist(confidences, bins=30, alpha=0.7)
    axes[1, 0].axvline(np.mean(confidences), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(confidences):.3f}')
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Confidence Distribution')
    axes[1, 0].legend()
    
    # Confidence vs accuracy
    accuracies = [r['top1_correct'] for r in results]
    board_sizes = [r['board_size'] for r in results]
    scatter = axes[1, 1].scatter(confidences, accuracies, c=board_sizes, alpha=0.6)
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Correct (0/1)')
    axes[1, 1].set_title('Confidence vs Accuracy')
    plt.colorbar(scatter, ax=axes[1, 1], label='Board Size')
    
    # Error analysis
    error_by_size = 1 - size_df['top1_accuracy']
    axes[1, 2].bar(size_df['board_size'], error_by_size, alpha=0.7, color='red')
    axes[1, 2].set_xlabel('Board Size')
    axes[1, 2].set_ylabel('Error Rate')
    axes[1, 2].set_title('Error Rate by Board Size')
    
    plt.tight_layout()
    plt.savefig(output_path / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All visualizations saved!")

def create_summary_report(results, size_df, state_df, spatial_data, output_dir="evaluation_results"):
    """Create a text summary report."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Creating summary report...")
    
    # Calculate overall metrics
    overall_top1 = np.mean([r['top1_correct'] for r in results])
    overall_top2 = np.mean([r['top2_correct'] for r in results])
    overall_top3 = np.mean([r['top3_correct'] for r in results])
    overall_confidence = np.mean([r['top1_confidence'] for r in results])
    
    # Find best/worst performing sizes
    best_size = size_df.loc[size_df['top1_accuracy'].idxmax()]
    worst_size = size_df.loc[size_df['top1_accuracy'].idxmin()]
    
    # Find best/worst game states
    best_state = state_df.loc[state_df['top1_accuracy'].idxmax()]
    worst_state = state_df.loc[state_df['top1_accuracy'].idxmin()]
    
    report = f"""
QUEENS MODEL EVALUATION REPORT
{'='*50}

OVERALL PERFORMANCE
{'-'*20}
Total Predictions: {len(results):,}
Top-1 Accuracy: {overall_top1:.1%}
Top-2 Accuracy: {overall_top2:.1%}
Top-3 Accuracy: {overall_top3:.1%}
Average Confidence: {overall_confidence:.3f}

BOARD SIZE ANALYSIS
{'-'*20}
Best Size: {int(best_size['board_size'])}x{int(best_size['board_size'])} ({best_size['top1_accuracy']:.1%})
Worst Size: {int(worst_size['board_size'])}x{int(worst_size['board_size'])} ({worst_size['top1_accuracy']:.1%})
Performance Gap: {(best_size['top1_accuracy'] - worst_size['top1_accuracy']):.1%}

GAME STATE ANALYSIS
{'-'*20}
Best State: {int(best_state['queens_remaining'])} queens remaining ({best_state['top1_accuracy']:.1%})
Worst State: {int(worst_state['queens_remaining'])} queens remaining ({worst_state['top1_accuracy']:.1%})

SPATIAL PATTERNS
{'-'*20}
"""
    
    for board_size, data in spatial_data.items():
        report += f"{board_size}x{board_size}: Center {data['center_error_rate']:.1%}, Edge {data['edge_error_rate']:.1%}, Corner {data['corner_error_rate']:.1%} error rates\n"
    
    report += f"""
KEY INSIGHTS
{'-'*20}
• Top-2 improvement: {(overall_top2 - overall_top1):.1%}
• Model is {'overconfident' if overall_confidence > overall_top1 else 'underconfident' if overall_confidence < overall_top1 else 'well-calibrated'}
• Board size matters: {(best_size['top1_accuracy'] - worst_size['top1_accuracy']):.1%} performance range
"""
    
    # Save report and data
    report_path = output_path / 'evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save data
    size_df.to_csv(output_path / 'board_size_analysis.csv', index=False)
    state_df.to_csv(output_path / 'game_state_analysis.csv', index=False)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / 'detailed_results.csv', index=False)
    
    print(f"Report saved to {report_path}")
    return str(report_path)

def run_evaluation(model_path: str, config: Config, use_heterogeneous: bool = True, 
                  output_dir: str = "evaluation_results", test = False):
    """Run complete evaluation pipeline."""
    print("Starting model evaluation...")
    
    # Load model
    model = load_model(model_path, config, use_heterogeneous)
    
    if not test:
        # Load validation data
        _, val_loader = get_queens_loaders(
            config.data.train_json,
            batch_size=512,
            val_ratio=config.training.val_ratio,
            seed=config.data.seed,
            num_workers=2,
            shuffle_train=False
        )
    else:
        # Load test data
        _, val_loader = get_queens_loaders(
            config.data.test_json,
            batch_size=512,
            val_ratio= 1.0,
            seed=config.data.seed,
            num_workers=2,
            shuffle_train=False
        )
    
    # Run predictions
    results = run_predictions(model, val_loader, use_heterogeneous, config.system.device)
    
    # Run analyses
    size_df = analyze_board_size_effects(results)
    state_df = analyze_game_state_effects(results)
    spatial_data = analyze_spatial_patterns(results)
    
    # Create outputs
    create_visualizations(results, size_df, state_df, spatial_data, f"{output_dir}/plots")
    report_path = create_summary_report(results, size_df, state_df, spatial_data, output_dir)
    
    print(f"\nEvaluation complete!")
    print(f"Results: {output_dir}/")
    print(f"Report: {report_path}")
    
    return results, size_df, state_df, spatial_data

def main():
    """Example usage."""
    config = Config()
    config.data.train_json = "10k_training_set_with_states.json"
    
    model_path = "checkpoints/best_model.pt"  # Update this path
    
    results, size_df, state_df, spatial_data = run_evaluation(
        model_path=model_path,
        config=config,
        use_heterogeneous=True,
        output_dir="my_evaluation"
    )
    
    print("Done!")