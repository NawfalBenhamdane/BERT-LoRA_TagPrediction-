"""
Prediction script for single exercises.
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


def predict_exercise(predictor, exercise_text: str, optimal_thresholds: dict):
    """
    Predict tags for a single exercise.
    
    Args:
        predictor: RetrievalAugmentedPredictor instance
        exercise_text: Exercise text to classify
        optimal_thresholds: Dictionary of optimal thresholds
        
    Returns:
        Dictionary with predictions
    """
    result = predictor.predict_single(exercise_text)
    
    # Apply thresholds
    predictions = {}
    for tag, score in result['final_scores'].items():
        threshold = optimal_thresholds.get(tag, 0.5)
        predictions[tag] = {
            'score': score,
            'threshold': threshold,
            'predicted': score >= threshold
        }
    
    return {
        'predictions': predictions,
        'retrieval_scores': result['retrieval_scores'],
        'cross_encoder_scores': result['cross_encoder_scores'],
        'neighbor_indices': result['neighbor_indices'][:5],  # Top 5
        'neighbor_distances': result['neighbor_distances'][:5]
    }


def main():
    """Main prediction function."""
    print("="*80)
    print("RAG+BERT CODE CLASSIFICATION - PREDICTION")
    print("="*80)
    
    # Load configuration
    config = Config("config.yaml")
    print(f"\nConfiguration loaded: {config}")
    
    # Load data splits (for training data reference)
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    data_splits = load_splits(config.output_dir / 'data_splits.pkl')
    df_train = data_splits['df_train']
    y_train = data_splits['y_train']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    print(f"Tokenizer loaded: {config.base_model_name}")
    
    # Load models
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
    contrastive_model = ContrastiveEncoder(
        base_model_name=config.base_model_name,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim
    )
    checkpoint = torch.load(config.output_dir / 'best_contrastive_encoder.pt')
    contrastive_model.load_state_dict(checkpoint['model_state_dict'])
    print("Contrastive encoder loaded")
    
    cross_encoder = CrossEncoder(
        base_model_name=config.base_model_name,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.cross_encoder_hidden_dim
    )
    checkpoint_cross = torch.load(config.output_dir / 'best_cross_encoder.pt')
    cross_encoder.load_state_dict(checkpoint_cross['model_state_dict'])
    print("Cross-encoder loaded")
    
    faiss_index = faiss.read_index(str(config.output_dir / 'faiss_index.bin'))
    print(f"FAISS index loaded: {faiss_index.ntotal} vectors")
    
    # Load optimal thresholds
    with open(config.output_dir / 'optimal_thresholds.json', 'r') as f:
        optimal_thresholds = json.load(f)
    print("Optimal thresholds loaded")
    
    # Create predictor
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
    print("Predictor ready")
    
    # Example exercise
    print("\n" + "="*80)
    print("EXAMPLE PREDICTION")
    print("="*80)
    
    example_text = """
    Given an array of integers, find the maximum sum of a contiguous subarray.
    [SEP] Input: An array of integers nums of length n (1 <= n <= 10^5)
    [SEP] Output: Return the maximum sum of any contiguous subarray
    """
    
    print("\nExercise:")
    print(example_text.strip())
    
    print("\nPredicting...")
    result = predict_exercise(predictor, example_text, optimal_thresholds)
    
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    print("\nPredicted Tags:")
    predicted_tags = [tag for tag, info in result['predictions'].items() if info['predicted']]
    if predicted_tags:
        for tag in predicted_tags:
            info = result['predictions'][tag]
            print(f"  ✓ {tag:20s} (score: {info['score']:.4f}, threshold: {info['threshold']:.3f})")
    else:
        print("  No tags predicted")
    
    print("\nAll Tag Scores:")
    for tag, info in sorted(result['predictions'].items(), key=lambda x: x[1]['score'], reverse=True):
        status = "✓" if info['predicted'] else "✗"
        print(f"  {status} {tag:20s}: {info['score']:.4f} (threshold: {info['threshold']:.3f})")
    
    print("\nTop 5 Similar Exercises (indices):")
    for i, (idx, dist) in enumerate(zip(result['neighbor_indices'], result['neighbor_distances']), 1):
        print(f"  {i}. Index {idx} (distance: {dist:.4f})")
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nEnter your own exercise description (or 'quit' to exit):")
    print("Format: Description [SEP] Input: ... [SEP] Output: ...")
    
    while True:
        print("\n" + "-"*80)
        user_input = input("\nExercise: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nExiting...")
            break
        
        if not user_input:
            print("Please enter a valid exercise description")
            continue
        
        try:
            result = predict_exercise(predictor, user_input, optimal_thresholds)
            
            print("\nPredicted Tags:")
            predicted_tags = [tag for tag, info in result['predictions'].items() if info['predicted']]
            if predicted_tags:
                for tag in predicted_tags:
                    info = result['predictions'][tag]
                    print(f"  ✓ {tag:20s} (score: {info['score']:.4f})")
            else:
                print("  No tags predicted")
            
            print("\nTop Scores:")
            for tag, info in sorted(result['predictions'].items(), key=lambda x: x[1]['score'], reverse=True)[:5]:
                status = "✓" if info['predicted'] else "✗"
                print(f"  {status} {tag:20s}: {info['score']:.4f}")
        
        except Exception as e:
            print(f"\nError during prediction: {e}")
            continue


if __name__ == "__main__":
    main()
