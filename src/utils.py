"""
Utility functions for evaluation and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Dict, Tuple, List
import json
from pathlib import Path


def find_optimal_thresholds(
    y_true: pd.DataFrame,
    scores_df: pd.DataFrame,
    target_tags: List[str],
    metric: str = 'f1'
) -> Dict[str, float]:
    """
    Find optimal thresholds for each tag.
    
    Args:
        y_true: True labels
        scores_df: Predicted scores
        target_tags: List of target tags
        metric: Metric to optimize ('f1', 'precision', 'recall')
        
    Returns:
        Dictionary of optimal thresholds per tag
    """
    optimal_thresholds = {}
    
    print(f"\nOptimizing thresholds (metric: {metric})...")
    
    for tag in target_tags:
        best_threshold = 0.5
        best_score = 0
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (scores_df[tag] >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true[tag], y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true[tag], y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true[tag], y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        optimal_thresholds[tag] = best_threshold
        print(f"  {tag:20s}: threshold = {best_threshold:.3f}, {metric} = {best_score:.4f}")
    
    return optimal_thresholds


def evaluate_with_thresholds(
    y_true: pd.DataFrame,
    scores_df: pd.DataFrame,
    thresholds: Dict[str, float],
    target_tags: List[str]
) -> Tuple[Dict, Dict]:
    """
    Evaluate performance with custom thresholds.
    
    Args:
        y_true: True labels
        scores_df: Predicted scores
        thresholds: Dictionary of thresholds per tag
        target_tags: List of target tags
        
    Returns:
        Tuple of (global metrics, per-tag metrics)
    """
    y_pred = np.zeros_like(y_true.values)
    
    for i, tag in enumerate(target_tags):
        threshold = thresholds[tag]
        y_pred[:, i] = (scores_df[tag] >= threshold).astype(int)
    
    # Global metrics
    metrics = {
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'exact_match': np.mean(np.all(y_true.values == y_pred, axis=1))
    }
    
    # Per-tag metrics
    per_tag_metrics = {}
    for i, tag in enumerate(target_tags):
        per_tag_metrics[tag] = {
            'precision': precision_score(y_true.iloc[:, i], y_pred[:, i], zero_division=0),
            'recall': recall_score(y_true.iloc[:, i], y_pred[:, i], zero_division=0),
            'f1': f1_score(y_true.iloc[:, i], y_pred[:, i], zero_division=0),
            'support': int(y_true.iloc[:, i].sum())
        }
    
    return metrics, per_tag_metrics


def print_evaluation_results(
    metrics: Dict,
    per_tag_metrics: Dict,
    title: str = "EVALUATION RESULTS"
):
    """
    Print evaluation results in a formatted way.
    
    Args:
        metrics: Global metrics
        per_tag_metrics: Per-tag metrics
        title: Title for the output
    """
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    print("\nGlobal Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    print("\nPer-Tag Metrics:")
    print(f"  {'Tag':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-"*70)
    for tag, tag_metrics in per_tag_metrics.items():
        print(f"  {tag:<20} {tag_metrics['precision']:>10.4f} {tag_metrics['recall']:>10.4f} "
              f"{tag_metrics['f1']:>10.4f} {tag_metrics['support']:>10}")


def plot_results_comparison(
    val_metrics: Dict,
    test_metrics: Dict,
    val_per_tag: Dict,
    test_per_tag: Dict,
    output_path: Path
):
    """
    Plot comparison of validation and test results.
    
    Args:
        val_metrics: Validation global metrics
        test_metrics: Test global metrics
        val_per_tag: Validation per-tag metrics
        test_per_tag: Test per-tag metrics
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    tags_list = list(val_per_tag.keys())
    
    # F1 Scores per tag
    val_f1 = [val_per_tag[tag]['f1'] for tag in tags_list]
    test_f1 = [test_per_tag[tag]['f1'] for tag in tags_list]
    
    x = np.arange(len(tags_list))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, val_f1, width, label='Validation', color='steelblue', edgecolor='black')
    axes[0, 0].bar(x + width/2, test_f1, width, label='Test', color='coral', edgecolor='black')
    axes[0, 0].set_xlabel('Tags', fontweight='bold')
    axes[0, 0].set_ylabel('F1-Score', fontweight='bold')
    axes[0, 0].set_title('F1-Score per Tag - Val vs Test', fontweight='bold', fontsize=14, pad=20)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(tags_list, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Precision per tag
    val_prec = [val_per_tag[tag]['precision'] for tag in tags_list]
    test_prec = [test_per_tag[tag]['precision'] for tag in tags_list]
    
    axes[0, 1].bar(x - width/2, val_prec, width, label='Validation', color='lightgreen', edgecolor='black')
    axes[0, 1].bar(x + width/2, test_prec, width, label='Test', color='salmon', edgecolor='black')
    axes[0, 1].set_xlabel('Tags', fontweight='bold')
    axes[0, 1].set_ylabel('Precision', fontweight='bold')
    axes[0, 1].set_title('Precision per Tag - Val vs Test', fontweight='bold', fontsize=14, pad=20)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(tags_list, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Recall per tag
    val_rec = [val_per_tag[tag]['recall'] for tag in tags_list]
    test_rec = [test_per_tag[tag]['recall'] for tag in tags_list]
    
    axes[1, 0].bar(x - width/2, val_rec, width, label='Validation', color='mediumpurple', edgecolor='black')
    axes[1, 0].bar(x + width/2, test_rec, width, label='Test', color='gold', edgecolor='black')
    axes[1, 0].set_xlabel('Tags', fontweight='bold')
    axes[1, 0].set_ylabel('Recall', fontweight='bold')
    axes[1, 0].set_title('Recall per Tag - Val vs Test', fontweight='bold', fontsize=14, pad=20)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(tags_list, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Global metrics
    global_metrics_names = ['F1 Macro', 'Precision Macro', 'Recall Macro', 'Exact Match']
    global_metrics_keys = ['f1_macro', 'precision_macro', 'recall_macro', 'exact_match']
    val_global = [val_metrics[k] for k in global_metrics_keys]
    test_global = [test_metrics[k] for k in global_metrics_keys]
    
    x_global = np.arange(len(global_metrics_names))
    axes[1, 1].bar(x_global - width/2, val_global, width, label='Validation', color='cyan', edgecolor='black')
    axes[1, 1].bar(x_global + width/2, test_global, width, label='Test', color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Metrics', fontweight='bold')
    axes[1, 1].set_ylabel('Score', fontweight='bold')
    axes[1, 1].set_title('Global Metrics - Val vs Test', fontweight='bold', fontsize=14, pad=20)
    axes[1, 1].set_xticks(x_global)
    axes[1, 1].set_xticklabels(global_metrics_names, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved: {output_path}")


def save_results(
    val_metrics: Dict,
    test_metrics: Dict,
    val_per_tag: Dict,
    test_per_tag: Dict,
    optimal_thresholds: Dict,
    config_dict: Dict,
    output_path: Path
):
    """
    Save evaluation results to JSON file.
    
    Args:
        val_metrics: Validation global metrics
        test_metrics: Test global metrics
        val_per_tag: Validation per-tag metrics
        test_per_tag: Test per-tag metrics
        optimal_thresholds: Optimal thresholds per tag
        config_dict: Configuration dictionary
        output_path: Path to save results
    """
    results = {
        'val_metrics': {k: float(v) for k, v in val_metrics.items()},
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'val_per_tag': {tag: {k: float(v) for k, v in m.items()} 
                        for tag, m in val_per_tag.items()},
        'test_per_tag': {tag: {k: float(v) for k, v in m.items()} 
                         for tag, m in test_per_tag.items()},
        'optimal_thresholds': optimal_thresholds,
        'config': config_dict
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved: {output_path}")
