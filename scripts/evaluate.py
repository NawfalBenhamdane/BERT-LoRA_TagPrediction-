"""
Evaluation script for the RAG+BERT system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import faiss
import json
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

from src.config import Config
from src.data_loader import load_splits
from src.models import ContrastiveEncoder, CrossEncoder
from src.predictor import RetrievalAugmentedPredictor
from src.utils import (
    find_optimal_thresholds,
    evaluate_with_thresholds,
    print_evaluation_results,
    plot_results_comparison,
    save_results
)


def main():
    """Main evaluation function."""
    print("="*80)
    print("RAG+BERT CODE CLASSIFICATION - EVALUATION")
    print("="*80)
    
    # Load configuration
    config = Config("config.yaml")
    print(f"\nConfiguration loaded: {config}")
    
    # Load data splits
    print("\n" + "="*80)
    print("LOADING DATA SPLITS")
    print("="*80)
    
    data_splits = load_splits(config.output_dir / 'data_splits.pkl')
    df_train = data_splits['df_train']
    df_val = data_splits['df_val']
    df_test = data_splits['df_test']
    y_train = data_splits['y_train']
    y_val = data_splits['y_val']
    y_test = data_splits['y_test']
    
    print(f"Train: {len(df_train)} samples")
    print(f"Val: {len(df_val)} samples")
    print(f"Test: {len(df_test)} samples")
    
    # Load tokenizer
    print("\n" + "="*80)
    print("LOADING TOKENIZER")
    print("="*80)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    print(f"Tokenizer loaded: {config.base_model_name}")
    
    # Load models
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
    # Contrastive encoder
    contrastive_model = ContrastiveEncoder(
        base_model_name=config.base_model_name,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim
    )
    checkpoint = torch.load(config.output_dir / 'best_contrastive_encoder.pt')
    contrastive_model.load_state_dict(checkpoint['model_state_dict'])
    print("Contrastive encoder loaded")
    
    # Cross-encoder
    cross_encoder = CrossEncoder(
        base_model_name=config.base_model_name,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.cross_encoder_hidden_dim
    )
    checkpoint_cross = torch.load(config.output_dir / 'best_cross_encoder.pt')
    cross_encoder.load_state_dict(checkpoint_cross['model_state_dict'])
    print("Cross-encoder loaded")
    
    # Load FAISS index
    faiss_index = faiss.read_index(str(config.output_dir / 'faiss_index.bin'))
    print(f"FAISS index loaded: {faiss_index.ntotal} vectors")
    
    # Create predictor
    print("\n" + "="*80)
    print("CREATING PREDICTOR")
    print("="*80)
    
    predictor = RetrievalAugmentedPredictor(
        contrastive_model=contrastive_model,
        cross_encoder=cross_encoder,
        faiss_index=faiss_index,
        df_train=df_train,
        y_train=y_train,
        tokenizer=tokenizer,
        target_tags=config.TARGET_TAGS,
        tag_descriptions=config.tag_descriptions,
        max_length=config.max_length,
        top_k=config.top_k,
        alpha=config.alpha,
        beta=config.beta,
        device=config.device
    )
    print("Predictor created")
    
    # ========================================
    # VALIDATION SET EVALUATION
    # ========================================
    print("\n" + "="*80)
    print("VALIDATION SET PREDICTION")
    print("="*80)
    
    val_scores_df = predictor.predict_batch(df_val)
    val_scores_df.to_csv(config.output_dir / 'val_scores.csv', index=False)
    print(f"\nValidation scores saved: {config.output_dir / 'val_scores.csv'}")
    
    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(
        y_val, val_scores_df, config.TARGET_TAGS, metric='f1'
    )
    
    # Save thresholds
    with open(config.output_dir / 'optimal_thresholds.json', 'w') as f:
        json.dump(optimal_thresholds, f, indent=2)
    print(f"\nOptimal thresholds saved: {config.output_dir / 'optimal_thresholds.json'}")
    
    # Evaluate validation set
    val_metrics, val_per_tag = evaluate_with_thresholds(
        y_val, val_scores_df, optimal_thresholds, config.TARGET_TAGS
    )
    
    print_evaluation_results(val_metrics, val_per_tag, "VALIDATION SET RESULTS")
    
    # ========================================
    # TEST SET EVALUATION
    # ========================================
    print("\n" + "="*80)
    print("TEST SET PREDICTION")
    print("="*80)
    
    test_scores_df = predictor.predict_batch(df_test)
    test_scores_df.to_csv(config.output_dir / 'test_scores.csv', index=False)
    print(f"\nTest scores saved: {config.output_dir / 'test_scores.csv'}")
    
    # Evaluate test set
    test_metrics, test_per_tag = evaluate_with_thresholds(
        y_test, test_scores_df, optimal_thresholds, config.TARGET_TAGS
    )
    
    print_evaluation_results(test_metrics, test_per_tag, "TEST SET RESULTS")
    
    # ========================================
    # SAVE RESULTS AND PLOTS
    # ========================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save results
    save_results(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_per_tag=val_per_tag,
        test_per_tag=test_per_tag,
        optimal_thresholds=optimal_thresholds,
        config_dict=config.to_dict(),
        output_path=config.output_dir / 'final_results.json'
    )
    
    # Plot comparison
    plot_results_comparison(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_per_tag=val_per_tag,
        test_per_tag=test_per_tag,
        output_path=config.output_dir / 'final_results_comparison.png'
    )
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nDataset:")
    print(f"  Train: {len(df_train)} exercises")
    print(f"  Val: {len(df_val)} exercises")
    print(f"  Test: {len(df_test)} exercises")
    print(f"  Target tags: {config.TARGET_TAGS}")
    
    print(f"\nModels:")
    print(f"  Contrastive Encoder: {config.base_model_name}")
    print(f"  Projection dimension: {config.projection_dim}")
    print(f"  Top-K retrieval: {config.top_k}")
    print(f"  Cross-Encoder: {config.base_model_name}")
    
    print(f"\nFinal Test Performance:")
    print(f"  F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"  F1 Micro: {test_metrics['f1_micro']:.4f}")
    print(f"  Precision Macro: {test_metrics['precision_macro']:.4f}")
    print(f"  Recall Macro: {test_metrics['recall_macro']:.4f}")
    print(f"  Exact Match: {test_metrics['exact_match']:.4f}")
    
    print(f"\nBest Tags (F1-Score):")
    sorted_tags = sorted(test_per_tag.items(), key=lambda x: x[1]['f1'], reverse=True)
    for tag, metrics in sorted_tags[:5]:
        print(f"  {tag:20s}: F1={metrics['f1']:.4f}")
    
    print(f"\nAll results saved in: {config.output_dir}")
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
