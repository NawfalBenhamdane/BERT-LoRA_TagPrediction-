"""
Configuration management for RAG+BERT system.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any
import torch


class Config:
    """Configuration class for the RAG+BERT system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self._load_config()
        self._setup_paths()
        self._setup_device()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Data paths
        self.data_dir = config_dict['data']['data_dir']
        self.output_dir = Path(config_dict['data']['output_dir'])
        
        # Target tags
        self.TARGET_TAGS = config_dict['target_tags']
        self.tag_descriptions = config_dict['tag_descriptions']
        
        # Model configuration
        model_config = config_dict['model']
        self.base_model_name = model_config['base_model_name']
        self.max_length = model_config['max_length']
        self.embedding_dim = model_config['embedding_dim']
        self.projection_dim = model_config['projection_dim']
        self.cross_encoder_hidden_dim = model_config['cross_encoder_hidden_dim']
        
        # Contrastive learning
        contrastive_config = config_dict['contrastive']
        self.temperature = contrastive_config['temperature']
        self.margin = contrastive_config['margin']
        
        # Retrieval
        retrieval_config = config_dict['retrieval']
        self.top_k = retrieval_config['top_k']
        self.alpha = retrieval_config['alpha']
        self.beta = retrieval_config['beta']
        
        # Training
        training_config = config_dict['training']
        self.batch_size = training_config['batch_size']
        self.learning_rate = training_config['learning_rate']
        self.num_epochs_contrastive = training_config['num_epochs_contrastive']
        self.num_epochs_cross = training_config['num_epochs_cross']
        self.gradient_clip_max_norm = training_config['gradient_clip_max_norm']
        self.num_workers = training_config['num_workers']
        
        # Data split
        split_config = config_dict['split']
        self.train_ratio = split_config['train_ratio']
        self.val_ratio = split_config['val_ratio']
        self.test_ratio = split_config['test_ratio']
        self.random_state = split_config['random_state']
        
        # Logging
        logging_config = config_dict['logging']
        self.log_level = logging_config['log_level']
        self.save_plots = logging_config['save_plots']
        self.plot_dpi = logging_config['plot_dpi']
        
        # Device
        self.device_name = config_dict['device']
    
    def _setup_paths(self):
        """Create necessary directories."""
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def _setup_device(self):
        """Setup computation device."""
        if self.device_name == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data_dir': self.data_dir,
            'output_dir': str(self.output_dir),
            'target_tags': self.TARGET_TAGS,
            'base_model_name': self.base_model_name,
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim,
            'projection_dim': self.projection_dim,
            'temperature': self.temperature,
            'margin': self.margin,
            'top_k': self.top_k,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs_contrastive': self.num_epochs_contrastive,
            'num_epochs_cross': self.num_epochs_cross,
        }
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(model={self.base_model_name}, device={self.device})"
