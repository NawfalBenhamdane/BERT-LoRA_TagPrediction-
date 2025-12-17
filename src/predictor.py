"""
Prediction and evaluation utilities for the RAG system.
"""

import torch
import numpy as np
import pandas as pd
import faiss
from typing import Dict, List, Tuple
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .models import ContrastiveEncoder, CrossEncoder


class RetrievalAugmentedPredictor:
    """Predictor combining retrieval and cross-encoding."""
    
    def __init__(
        self,
        contrastive_model: ContrastiveEncoder,
        cross_encoder: CrossEncoder,
        faiss_index: faiss.Index,
        df_train: pd.DataFrame,
        y_train: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        target_tags: List[str],
        tag_descriptions: Dict[str, str],
        max_length: int,
        top_k: int,
        alpha: float,
        beta: float,
        device: torch.device
    ):
        """
        Initialize retrieval-augmented predictor.
        
        Args:
            contrastive_model: Trained contrastive encoder
            cross_encoder: Trained cross-encoder
            faiss_index: FAISS index for retrieval
            df_train: Training DataFrame
            y_train: Training labels
            tokenizer: Tokenizer
            target_tags: List of target tags
            tag_descriptions: Tag descriptions
            max_length: Maximum sequence length
            top_k: Number of neighbors to retrieve
            alpha: Weight for retrieval scores
            beta: Weight for cross-encoder scores
            device: Computation device
        """
        self.contrastive_model = contrastive_model.to(device)
        self.cross_encoder = cross_encoder.to(device)
        self.faiss_index = faiss_index
        self.df_train = df_train
        self.y_train = y_train
        self.tokenizer = tokenizer
        self.target_tags = target_tags
        self.tag_descriptions = tag_descriptions
        self.max_length = max_length
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
        self.contrastive_model.eval()
        self.cross_encoder.eval()
    
    def retrieve_similar_exercises(
        self,
        query_embedding: np.ndarray,
        k: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve k most similar exercises.
        
        Args:
            query_embedding: Query embedding
            k: Number of neighbors (default: self.top_k)
            
        Returns:
            Tuple of (distances, indices)
        """
        if k is None:
            k = self.top_k
        
        query_embedding = query_embedding.astype('float32')
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        return distances[0], indices[0]
    
    def aggregate_tags_from_neighbors(
        self,
        neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray
    ) -> Dict[str, float]:
        """
        Aggregate tags from neighbors with weighted voting.
        
        Args:
            neighbor_indices: Indices of neighbors
            neighbor_distances: Distances to neighbors
            
        Returns:
            Dictionary of tag scores
        """
        # Convert L2 distances to similarities
        similarities = 1.0 / (1.0 + neighbor_distances)
        similarities = similarities / similarities.sum()
        
        # Aggregate tags
        tag_scores = {}
        for tag in self.target_tags:
            neighbor_labels = self.y_train.iloc[neighbor_indices][tag].values
            score = np.sum(neighbor_labels * similarities)
            tag_scores[tag] = float(score)
        
        return tag_scores
    
    def predict_single(self, exercise_text: str, k: int = None) -> Dict:
        """
        Predict tags for a single exercise.
        
        Args:
            exercise_text: Exercise text
            k: Number of neighbors (default: self.top_k)
            
        Returns:
            Dictionary with prediction results
        """
        if k is None:
            k = self.top_k
        
        with torch.no_grad():
            # Step 1: Encode the exercise
            encoding = self.tokenizer(
                exercise_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            query_embedding = self.contrastive_model(
                encoding['input_ids'],
                encoding['attention_mask']
            ).cpu().numpy()
            
            # Step 2: Retrieve k neighbors
            distances, indices = self.retrieve_similar_exercises(query_embedding, k=k)
            
            # Step 3: Aggregate tags from neighbors
            retrieval_scores = self.aggregate_tags_from_neighbors(indices, distances)
            
            # Step 4: Cross-encoder scoring
            cross_encoder_scores = {}
            
            for tag in self.target_tags:
                tag_desc = self.tag_descriptions[tag]
                pair_text = f"{exercise_text} [SEP] Tag: {tag_desc}"
                
                pair_encoding = self.tokenizer(
                    pair_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                score = self.cross_encoder(
                    pair_encoding['input_ids'],
                    pair_encoding['attention_mask']
                ).item()
                
                cross_encoder_scores[tag] = score
            
            # Step 5: Combine scores
            final_scores = {}
            for tag in self.target_tags:
                final_scores[tag] = (
                    self.alpha * retrieval_scores[tag] +
                    self.beta * cross_encoder_scores[tag]
                )
            
            return {
                'final_scores': final_scores,
                'retrieval_scores': retrieval_scores,
                'cross_encoder_scores': cross_encoder_scores,
                'neighbor_indices': indices.tolist(),
                'neighbor_distances': distances.tolist()
            }
    
    def predict_batch(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predict tags for a batch of exercises.
        
        Args:
            df_test: Test DataFrame
            
        Returns:
            DataFrame with prediction scores
        """
        all_scores = []
        
        print("\nPredicting on dataset...")
        for idx in tqdm(range(len(df_test))):
            exercise_text = df_test.iloc[idx]['text']
            result = self.predict_single(exercise_text)
            all_scores.append(result['final_scores'])
        
        scores_df = pd.DataFrame(all_scores)
        return scores_df


def create_faiss_index(
    model: ContrastiveEncoder,
    tokenizer: PreTrainedTokenizer,
    df_train: pd.DataFrame,
    y_train: pd.DataFrame,
    max_length: int,
    batch_size: int,
    device: torch.device,
    output_dir
) -> Tuple[faiss.Index, np.ndarray]:
    """
    Create FAISS index from training data.
    
    Args:
        model: Trained contrastive encoder
        tokenizer: Tokenizer
        df_train: Training DataFrame
        y_train: Training labels
        max_length: Maximum sequence length
        batch_size: Batch size for encoding
        device: Computation device
        output_dir: Directory to save index
        
    Returns:
        Tuple of (FAISS index, embeddings array)
    """
    from torch.utils.data import DataLoader
    from .dataset import ContrastiveDataset
    
    print("\n" + "="*80)
    print("CREATING FAISS INDEX")
    print("="*80)
    
    model.eval()
    
    # Create dataset
    train_dataset = ContrastiveDataset(
        df_train, y_train, tokenizer, max_length, is_train=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Extract embeddings
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            embeddings = model(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())
    
    all_embeddings = np.vstack(all_embeddings)
    
    print(f"\nEmbeddings extracted: {all_embeddings.shape}")
    print(f"  Min: {all_embeddings.min():.4f}")
    print(f"  Max: {all_embeddings.max():.4f}")
    print(f"  Mean: {all_embeddings.mean():.4f}")
    print(f"  Std: {all_embeddings.std():.4f}")
    
    # Create FAISS index
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings.astype('float32'))
    
    print(f"\nFAISS index created:")
    print(f"  Dimension: {dimension}")
    print(f"  Number of vectors: {index.ntotal}")
    
    # Save index
    faiss.write_index(index, str(output_dir / 'faiss_index.bin'))
    np.save(output_dir / 'train_embeddings.npy', all_embeddings)
    
    print(f"\nIndex saved: {output_dir / 'faiss_index.bin'}")
    
    return index, all_embeddings
