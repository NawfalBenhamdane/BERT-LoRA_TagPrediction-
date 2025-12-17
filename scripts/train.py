"""
Training script for the RAG+BERT system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

from src.config import Config
from src.data_loader import (
    load_and_filter_data,
    prepare_dataframe,
    split_data,
    save_splits
)
from src.dataset import ContrastiveDataset, CrossEncoderDataset
from src.models import ContrastiveEncoder, CrossEncoder
from src.trainer import ContrastiveTrainer, CrossEncoderTrainer
from src.predictor import create_faiss_index


def main():
    """Main training function."""
    print("="*80)
    print("RAG+BERT CODE CLASSIFICATION - TRAINING")
    print("="*80)
    
    # Load configuration
    config = Config("config.yaml")
    print(f"\nConfiguration loaded: {config}")
    
    # Load and prepare data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    file_paths = glob.glob(f'{config.data_dir}/*.json')
    print(f"JSON files found: {len(file_paths)}")
    
    if len(file_paths) == 0:
        print(f"ERROR: No JSON files found in {config.data_dir}")
        print("Please update the data_dir in config.yaml")
        return
    
    data = load_and_filter_data(file_paths, config.TARGET_TAGS)
    df = prepare_dataframe(data, config.TARGET_TAGS)
    
    # Split data
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(
        df,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_state=config.random_state
    )
    
    # Save splits
    save_splits(
        df_train, df_val, df_test,
        y_train, y_val, y_test,
        config.TARGET_TAGS,
        config.output_dir / 'data_splits.pkl'
    )
    
    # Initialize tokenizer
    print("\n" + "="*80)
    print("INITIALIZING TOKENIZER")
    print("="*80)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    print(f"Tokenizer loaded: {config.base_model_name}")
    
    # ========================================
    # TRAIN CONTRASTIVE ENCODER
    # ========================================
    print("\n" + "="*80)
    print("PHASE 1: CONTRASTIVE ENCODER")
    print("="*80)
    
    # Create datasets
    train_dataset = ContrastiveDataset(
        df_train, y_train, tokenizer, config.max_length, is_train=True
    )
    val_dataset = ContrastiveDataset(
        df_val, y_val, tokenizer, config.max_length, is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Initialize model
    contrastive_model = ContrastiveEncoder(
        base_model_name=config.base_model_name,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim
    )
    
    # Train
    contrastive_trainer = ContrastiveTrainer(
        model=contrastive_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.device,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs_contrastive,
        temperature=config.temperature,
        margin=config.margin,
        gradient_clip_max_norm=config.gradient_clip_max_norm,
        output_dir=config.output_dir
    )
    
    contrastive_trainer.train()
    
    # Load best model
    checkpoint = torch.load(config.output_dir / 'best_contrastive_encoder.pt')
    contrastive_model.load_state_dict(checkpoint['model_state_dict'])
    print("\nBest contrastive model loaded")
    
    # ========================================
    # CREATE FAISS INDEX
    # ========================================
    faiss_index, train_embeddings = create_faiss_index(
        model=contrastive_model,
        tokenizer=tokenizer,
        df_train=df_train,
        y_train=y_train,
        max_length=config.max_length,
        batch_size=config.batch_size,
        device=config.device,
        output_dir=config.output_dir
    )
    
    # ========================================
    # TRAIN CROSS-ENCODER
    # ========================================
    print("\n" + "="*80)
    print("PHASE 2: CROSS-ENCODER")
    print("="*80)
    
    # Create datasets
    train_dataset_cross = CrossEncoderDataset(
        df_train, y_train, tokenizer, config.max_length,
        config.TARGET_TAGS, config.tag_descriptions
    )
    val_dataset_cross = CrossEncoderDataset(
        df_val, y_val, tokenizer, config.max_length,
        config.TARGET_TAGS, config.tag_descriptions
    )
    
    train_loader_cross = DataLoader(
        train_dataset_cross,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    val_loader_cross = DataLoader(
        val_dataset_cross,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Initialize model
    cross_encoder = CrossEncoder(
        base_model_name=config.base_model_name,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.cross_encoder_hidden_dim
    )
    
    # Train
    cross_trainer = CrossEncoderTrainer(
        model=cross_encoder,
        train_loader=train_loader_cross,
        val_loader=val_loader_cross,
        device=config.device,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs_cross,
        gradient_clip_max_norm=config.gradient_clip_max_norm,
        output_dir=config.output_dir
    )
    
    cross_trainer.train()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"\nModels saved in: {config.output_dir}")
    print("Files created:")
    print(f"  - best_contrastive_encoder.pt")
    print(f"  - best_cross_encoder.pt")
    print(f"  - faiss_index.bin")
    print(f"  - train_embeddings.npy")
    print(f"  - data_splits.pkl")
    print("\nNext step: Run 'python scripts/evaluate.py' to evaluate the models")


if __name__ == "__main__":
    main()
