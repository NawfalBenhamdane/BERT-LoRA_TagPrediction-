"""
Data loading and preprocessing utilities.
"""

import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import pickle


def load_and_filter_data(file_paths: List[str], target_tags: List[str]) -> List[Dict]:
    """
    Load all JSON files and filter those containing at least one target tag.
    
    Args:
        file_paths: List of paths to JSON files
        target_tags: List of target tags to filter by
        
    Returns:
        List of filtered examples
    """
    all_data = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)
    
    # Filter examples with target tags and source code
    filtered_data = []
    for example in all_data:
        tags = example.get('tags', [])
        source_code = example.get('source_code', '')
        if any(tag in target_tags for tag in tags) and source_code:
            filtered_data.append(example)
    
    print(f"Total examples loaded: {len(all_data)}")
    print(f"Filtered examples with target tags and source code: {len(filtered_data)}")
    
    return filtered_data


def prepare_dataframe(data: List[Dict], target_tags: List[str]) -> pd.DataFrame:
    """
    Prepare a DataFrame with text features, code, and labels.
    
    Args:
        data: List of examples
        target_tags: List of target tags
        
    Returns:
        DataFrame with processed data
    """
    records = []
    
    for example in data:
        # Combine text fields
        description = example.get('prob_desc_description', '')
        input_spec = example.get('prob_desc_input_spec', '')
        output_spec = example.get('prob_desc_output_spec', '')
        text = f"{description} [SEP] Input: {input_spec} [SEP] Output: {output_spec}"
        
        # Code source
        code = example.get('source_code', '')
        
        # Binary labels for each tag
        tags = example.get('tags', [])
        labels = [1 if tag in tags else 0 for tag in target_tags]
        
        records.append({
            'text': text,
            'code': code,
            'labels': labels,
            'raw_example': example
        })
    
    df = pd.DataFrame(records)
    
    # Statistics
    print("\n=== Tag Distribution ===")
    label_counts = np.array([r for r in df['labels']]).sum(axis=0)
    for tag, count in zip(target_tags, label_counts):
        print(f"{tag}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_state: Random seed
        
    Returns:
        Tuple of (df_train, df_val, df_test, y_train, y_val, y_test)
    """
    print("\n" + "="*80)
    print("DATA SPLIT")
    print("="*80)
    
    # Convert labels to DataFrame
    target_tags = ['math', 'graphs', 'strings', 'number theory', 
                   'trees', 'geometry', 'games', 'probabilities']
    y_all = pd.DataFrame(df['labels'].tolist(), columns=target_tags)
    
    # First split: train+val / test
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=random_state
    )
    
    # Second split: train / val
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size_adjusted,
        random_state=random_state
    )
    
    # Create splits
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    y_train = y_all.iloc[train_idx].reset_index(drop=True)
    y_val = y_all.iloc[val_idx].reset_index(drop=True)
    y_test = y_all.iloc[test_idx].reset_index(drop=True)
    
    print(f"\nSplits created:")
    print(f"  Train: {len(df_train)} exercises ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(df_val)} exercises ({len(df_val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(df_test)} exercises ({len(df_test)/len(df)*100:.1f}%)")
    
    # Print tag distribution per split
    print("\n" + "="*80)
    print("TAG DISTRIBUTION PER SPLIT")
    print("="*80)
    
    for split_name, y_split in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        print(f"\n{split_name}:")
        for tag in target_tags:
            count = y_split[tag].sum()
            pct = count / len(y_split) * 100
            print(f"  {tag:20s}: {count:5d} ({pct:5.2f}%)")
    
    return df_train, df_val, df_test, y_train, y_val, y_test


def save_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    y_test: pd.DataFrame,
    target_tags: List[str],
    output_path: Path
):
    """
    Save data splits to pickle file.
    
    Args:
        df_train, df_val, df_test: DataFrames for each split
        y_train, y_val, y_test: Label DataFrames for each split
        target_tags: List of target tags
        output_path: Path to save the pickle file
    """
    data_to_save = {
        'df_train': df_train,
        'df_val': df_val,
        'df_test': df_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'target_tags': target_tags
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"\nSplits saved: {output_path}")


def load_splits(input_path: Path) -> Dict:
    """
    Load data splits from pickle file.
    
    Args:
        input_path: Path to the pickle file
        
    Returns:
        Dictionary containing all splits
    """
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Splits loaded from: {input_path}")
    return data
