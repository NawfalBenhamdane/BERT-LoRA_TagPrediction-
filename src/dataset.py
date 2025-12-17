"""
PyTorch Dataset classes for contrastive learning and cross-encoding.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List
from tqdm import tqdm


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with triplet samples."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        labels: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        is_train: bool = True
    ):
        """
        Initialize contrastive dataset.
        
        Args:
            df: DataFrame with text and code
            labels: DataFrame with binary labels
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            is_train: Whether this is training data (for triplet sampling)
        """
        self.df = df
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        self.texts = df['text'].tolist()
        
        if is_train:
            self._build_pairs_index()
    
    def _build_pairs_index(self):
        """Build index for finding positive/negative pairs."""
        self.positive_pairs = {}
        self.negative_pairs = {}
        
        print("Building pairs index...")
        labels_array = self.labels.values
        
        for i in tqdm(range(len(self.labels))):
            tags_i = set(self.labels.columns[labels_array[i] == 1])
            
            positives = []
            negatives = []
            
            for j in range(len(self.labels)):
                if i == j:
                    continue
                
                tags_j = set(self.labels.columns[labels_array[j] == 1])
                
                if len(tags_i & tags_j) > 0:
                    positives.append(j)
                else:
                    negatives.append(j)
            
            self.positive_pairs[i] = positives
            self.negative_pairs[i] = negatives
        
        print("Pairs index built")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        anchor_text = self.texts[idx]
        anchor_encoding = self.tokenizer(
            anchor_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        anchor_labels = torch.FloatTensor(self.labels.iloc[idx].values)
        
        if self.is_train:
            # Positive example
            if len(self.positive_pairs[idx]) > 0:
                pos_idx = np.random.choice(self.positive_pairs[idx])
                positive_text = self.texts[pos_idx]
                positive_encoding = self.tokenizer(
                    positive_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            else:
                positive_encoding = anchor_encoding
            
            # Negative example
            if len(self.negative_pairs[idx]) > 0:
                neg_idx = np.random.choice(self.negative_pairs[idx])
                negative_text = self.texts[neg_idx]
                negative_encoding = self.tokenizer(
                    negative_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            else:
                negative_encoding = anchor_encoding
            
            return {
                'anchor_input_ids': anchor_encoding['input_ids'].squeeze(0),
                'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(0),
                'positive_input_ids': positive_encoding['input_ids'].squeeze(0),
                'positive_attention_mask': positive_encoding['attention_mask'].squeeze(0),
                'negative_input_ids': negative_encoding['input_ids'].squeeze(0),
                'negative_attention_mask': negative_encoding['attention_mask'].squeeze(0),
                'labels': anchor_labels
            }
        else:
            return {
                'input_ids': anchor_encoding['input_ids'].squeeze(0),
                'attention_mask': anchor_encoding['attention_mask'].squeeze(0),
                'labels': anchor_labels
            }


class CrossEncoderDataset(Dataset):
    """Dataset for cross-encoder training with (exercise, tag) pairs."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        labels: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        target_tags: List[str],
        tag_descriptions: Dict[str, str]
    ):
        """
        Initialize cross-encoder dataset.
        
        Args:
            df: DataFrame with text and code
            labels: DataFrame with binary labels
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            target_tags: List of target tags
            tag_descriptions: Dictionary mapping tags to descriptions
        """
        self.df = df
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_tags = target_tags
        self.tag_descriptions = tag_descriptions
        
        # Create all (exercise, tag) pairs
        self.pairs = []
        
        print("Creating pairs for cross-encoder...")
        for idx in tqdm(range(len(df))):
            exercise_text = df.iloc[idx]['text']
            
            for tag in target_tags:
                tag_desc = tag_descriptions[tag]
                label = labels.iloc[idx][tag]
                
                self.pairs.append({
                    'exercise_text': exercise_text,
                    'tag_description': tag_desc,
                    'tag_name': tag,
                    'label': label
                })
        
        print(f"Total pairs created: {len(self.pairs)}")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        pair = self.pairs[idx]
        
        # Concatenate exercise + tag description
        text = f"{pair['exercise_text']} [SEP] Tag: {pair['tag_description']}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.FloatTensor([pair['label']])
        }
