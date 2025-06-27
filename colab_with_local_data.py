#!/usr/bin/env python3
"""
GPT-2 Training in Google Colab with Local WikiText Data
"""

# ==================== IMPORTS ====================
import os
import sys

# Disable wandb logging completely
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import matplotlib.pyplot as plt
import json
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ==================== CUSTOM TRANSFORMER ARCHITECTURE ====================

# the default config is for GPT-2 Small 
class CustomTransformerConfig:
    """Configuration for custom GPT-2 model"""
    def __init__(self, vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12, n_inner=3072):
        self.vocab_size = vocab_size
        self.n_positions = n_positions  # max sequence length
        self.n_embd = n_embd           # embedding dimension
        self.n_layer = n_layer         # number of transformer layers
        self.n_head = n_head           # number of attention heads
        self.n_inner = n_inner         # feedforward inner dimension
    
    def to_dict(self):
        """Convert config to dictionary for Hugging Face compatibility"""
        return {
            'vocab_size': self.vocab_size,
            'n_positions': self.n_positions,
            'n_embd': self.n_embd,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_inner': self.n_inner,
            'model_type': 'custom_gpt2'
        }
    
    def to_json_string(self):
        """Convert config to JSON string for Hugging Face compatibility"""
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary for Hugging Face compatibility"""
        return cls(
            vocab_size=config_dict.get('vocab_size', 50257),
            n_positions=config_dict.get('n_positions', 1024),
            n_embd=config_dict.get('n_embd', 768),
            n_layer=config_dict.get('n_layer', 12),
            n_head=config_dict.get('n_head', 12),
            n_inner=config_dict.get('n_inner', 3072)
        )

class MultiHeadMaskedSelfAttention(nn.Module):
    """
    (b) Multi-head masked self-attention layers with causal (unidirectional) attention
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        # Use PyTorch's MultiheadAttention as suggested
        self.attention = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=0.1,
            batch_first=True  # Use batch_first for easier handling
        )
        
        # Create causal mask
        self.register_buffer("causal_mask", self._create_causal_mask(config.n_positions))
    
    def _create_causal_mask(self, seq_len):
        """Create causal mask to prevent attending to future positions"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool()
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Get causal mask for current sequence length
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        # Apply multi-head attention with causal mask AND SCALING
        attn_output, _ = self.attention(
            query=x,
            key=x, 
            value=x,
            attn_mask=causal_mask,
            need_weights=False
        )
        
        return attn_output

class FeedForward(nn.Module):
    """
    (c) Feedforward layers: Two linear layers with GELU non-linearity
    """
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.n_embd, config.n_inner)
        self.linear2 = nn.Linear(config.n_inner, config.n_embd)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    """
    (d) Layer normalization and residual connections with pre-layer normalization
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)  # Pre-attention layer norm
        self.attention = MultiHeadMaskedSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)  # Pre-feedforward layer norm
        self.feedforward = FeedForward(config)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Pre-layer normalization before attention + residual connection
        attn_output = self.attention(self.ln1(x))
        x = x + self.dropout(attn_output)
        
        # Pre-layer normalization before feedforward + residual connection
        ff_output = self.feedforward(self.ln2(x))
        x = x + self.dropout(ff_output)
        
        return x

class CustomGPT2Model(nn.Module):
    """
    Custom decoder-only Transformer architecture implementing GPT-2
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False  # Add gradient checkpointing support
        
        # (a) Token and position embeddings with learned position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.n_positions, config.n_embd)  # Learned positions
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False
    
    def get_input_embeddings(self):
        """Get input embeddings for compatibility with Hugging Face"""
        return self.token_embedding
    
    def set_input_embeddings(self, new_embeddings):
        """Set input embeddings for compatibility with Hugging Face"""
        self.token_embedding = new_embeddings
    
    def get_output_embeddings(self):
        """Get output embeddings for compatibility with Hugging Face"""
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings for compatibility with Hugging Face"""
        self.lm_head = new_embeddings
    
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # (a) Token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing during training for memory efficiency
                hidden_states = torch.utils.checkpoint.checkpoint(block, hidden_states, use_reentrant=False)
            else:
                hidden_states = block(hidden_states)
        
        # Final layer norm and output projection
        hidden_states = self.ln_final(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'logits': logits,
            'loss': loss
        }
    
    def generate(self, input_ids, max_length=50, temperature=1.0, do_sample=True, pad_token_id=None):
        """Simple generation method"""
        self.eval()
        device = input_ids.device
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(input_ids)
                logits = outputs['logits']
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop if we hit pad token
                if pad_token_id is not None and next_token.item() == pad_token_id:
                    break
        
        return input_ids
    
    def generate_with_strategies(self, input_ids, max_length=50, temperature=1.0, 
                             strategy='greedy', top_k=50, top_p=0.9, pad_token_id=None):
        """
        Advanced generation with multiple decoding strategies
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum sequence length
            temperature: Temperature for sampling (higher = more random)
            strategy: 'greedy', 'top_k', 'nucleus' (top_p), or 'random'
            top_k: Number of top tokens to consider for top-k sampling
            top_p: Cumulative probability threshold for nucleus sampling
            pad_token_id: Padding token ID
        
        Returns:
            Generated token IDs
        """
        self.eval()
        device = input_ids.device
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(input_ids)
                logits = outputs['logits']
                
                # Get next token logits and apply temperature
                next_token_logits = logits[:, -1, :] / temperature
                
                if strategy == 'greedy':
                    # Greedy decoding: Always select highest probability token
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                elif strategy == 'top_k':
                    # Top-k sampling: Limit choices to top k tokens
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = F.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, next_token_idx)
                
                elif strategy == 'nucleus' or strategy == 'top_p':
                    # Nucleus (top-p) sampling: Sample from smallest set with cumulative prob > p
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Set logits to -inf for tokens to remove
                    sorted_logits[sorted_indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(sorted_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, num_samples=1)
                    next_token = sorted_indices.gather(-1, next_token_idx)
                
                elif strategy == 'random':
                    # Random sampling from full distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                # Append the new token
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop if we hit pad token
                if pad_token_id is not None and next_token.item() == pad_token_id:
                    break
        
        return input_ids

# ==================== DATA LOADING FUNCTIONS ====================

def load_wikitext_from_local_files(test_mode=False):
    """Load WikiText from local parquet files in same directory"""
    
    # Define the files
    files = {
        "train": "train-00000-of-00001.parquet",
        "validation": "validation-00000-of-00001.parquet", 
        "test": "test-00000-of-00001.parquet"
    }
    
    # Load parquet files into datasets
    datasets = {}
    for split, filename in files.items():
        df = pd.read_parquet(filename)
        print(f"[OK] Loaded {split}: {len(df)} samples")
        datasets[split] = Dataset.from_pandas(df)
    
    # Create DatasetDict
    dataset = DatasetDict(datasets)
    
    # Apply test mode filtering if needed
    if test_mode:
        print("TEST MODE: Using subset of data")
        dataset['train'] = dataset['train'].select(range(min(100, len(dataset['train']))))
        dataset['validation'] = dataset['validation'].select(range(min(20, len(dataset['validation']))))
        dataset['test'] = dataset['test'].select(range(min(20, len(dataset['test']))))
    
    print(f"Final dataset sizes:")
    print(f"   Train: {len(dataset['train']):,} samples")
    print(f"   Validation: {len(dataset['validation']):,} samples")
    print(f"   Test: {len(dataset['test']):,} samples")
    
    return dataset

def tokenize_dataset(dataset, tokenizer, max_length=256):
    """Tokenize dataset for training"""
    
    def tokenize_function(examples):
        texts = [text for text in examples['text'] if text.strip()]
        if not texts:
            return {'input_ids': [], 'attention_mask': []}
        
        # Check if it's a custom BPE tokenizer (from tokenizers library)
        if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'get_vocab_size'):
            # Custom BPE tokenizer
            encodings = []
            attention_masks = []
            
            for text in texts:
                encoded = tokenizer.encode(text)
                tokens = encoded.ids[:max_length]
                
                # Pad to max_length
                if len(tokens) < max_length:
                    attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
                    tokens = tokens + [0] * (max_length - len(tokens))  # Use 0 for padding
                else:
                    attention_mask = [1] * max_length
                
                encodings.append(tokens)
                attention_masks.append(attention_mask)
            
            return {
                'input_ids': encodings,
                'attention_mask': attention_masks
            }
        else:
            # Hugging Face tokenizer
            return tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors=None
            )
    
    # Tokenize with progress bar
    tokenized_dataset = {}
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            print(f"Tokenizing {split}...")
            tokenized_dataset[split] = dataset[split].map(
                tokenize_function,
                batched=True,
                remove_columns=['text'],
                num_proc=1,  # Single process for stability
                desc=f"Tokenizing {split}"
            )
    
    return tokenized_dataset

def train_custom_bpe_tokenizer(dataset, vocab_size=32000, save_path="custom_bpe_tokenizer"):
    """
    (a) Define vocabulary size: 32,000 tokens
    (b) Train BPE tokenizer on dataset  
    (c) Save trained tokenizer
    """
    print(f"Training custom BPE tokenizer with vocab size {vocab_size}...")
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configure trainer with vocabulary size
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>"]
    )
    
    # Prepare training data
    def batch_iterator():
        for example in dataset:
            if example['text'].strip():  # Skip empty texts
                yield example['text']
    
    # Train the tokenizer on dataset
    tokenizer.train_from_iterator(batch_iterator(), trainer)
    
    # Save the trained tokenizer
    tokenizer.save(f"{save_path}.json")
    print(f"[OK] Custom BPE tokenizer saved to {save_path}.json")
    
    return tokenizer

def setup_training_environment():
    """Setup optimal environment for training"""
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Optimize for GPU type
        if "T4" in gpu_name:
            print("Optimizing for T4 GPU")
            return "small"  # Recommend small model for T4
        else:
            return "tiny"   # Conservative for other GPUs
    else:
        print("[WARNING] No GPU detected - training will be slow")
        return "tiny"

def create_custom_gpt2_model(model_size="tiny", use_custom_tokenizer=False, dataset=None):
    """Create custom GPT-2 model with specified size and tokenizer"""
    
    if use_custom_tokenizer and dataset is not None:
        # Train custom BPE tokenizer as required
        print("Training custom BPE tokenizer as per requirements...")
        tokenizer_path = "custom_bpe_tokenizer"
        
        if not os.path.exists(f"{tokenizer_path}.json"):
            custom_tokenizer = train_custom_bpe_tokenizer(
                dataset['train'], 
                vocab_size=32000,  # Requirement: 32,000 tokens
                save_path=tokenizer_path
            )
        else:
            print(f"Loading existing custom tokenizer from {tokenizer_path}.json")
            custom_tokenizer = Tokenizer.from_file(f"{tokenizer_path}.json")
        
        vocab_size = custom_tokenizer.get_vocab_size()
        tokenizer = custom_tokenizer
    else:
        # Fallback to pre-trained GPT-2 tokenizer
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
    
    if model_size == "small":  # GPT-2 Small (117M)
        config = CustomTransformerConfig(
            vocab_size=vocab_size,
            n_positions=1024, n_embd=768, n_layer=12, n_head=12, n_inner=3072
        )
    elif model_size == "medium":  # GPT-2 Medium (345M)
        config = CustomTransformerConfig(
            vocab_size=vocab_size,
            n_positions=1024, n_embd=1024, n_layer=24, n_head=16, n_inner=4096
        )
    elif model_size == "large":  # GPT-2 Large (762M)
        config = CustomTransformerConfig(
            vocab_size=vocab_size,
            n_positions=1024, n_embd=1280, n_layer=36, n_head=20, n_inner=5120
        )
    elif model_size == "xl":  # GPT-2 XL (1.5B)
        config = CustomTransformerConfig(
            vocab_size=vocab_size,
            n_positions=1024, n_embd=1600, n_layer=48, n_head=25, n_inner=6400
        )
    else:  # tiny (for testing)
        config = CustomTransformerConfig(
            vocab_size=vocab_size,
            n_positions=256, n_embd=128, n_layer=2, n_head=4, n_inner=512
        )
    
    model = CustomGPT2Model(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Custom model created: {total_params:,} parameters (~{total_params * 4 / (1024**2):.1f} MB)")
    
    return model, tokenizer

# ==================== CUSTOM TRAINER FOR CUSTOM MODEL ====================

class CustomTrainer(Trainer):
    """Custom trainer to work with our custom model and track loss curves"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss using our custom model's forward method"""
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels", input_ids)
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs, start_time=None):
        """Override log method to capture training loss"""
        # Call parent method with proper arguments
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        
        # Capture training loss
        if "train_loss" in logs:
            self.train_losses.append(logs["train_loss"])
            self.train_steps.append(self.state.global_step)
        
        # Capture evaluation loss
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            self.eval_steps.append(self.state.global_step)
    
    def get_loss_curves(self):
        """Return training and validation loss curves"""
        return {
            'train_losses': self.train_losses,
            'train_steps': self.train_steps,
            'eval_losses': self.eval_losses,
            'eval_steps': self.eval_steps
        }
    
    def save_loss_curves(self, save_path="./loss_curves.json"):
        """Save loss curves to JSON file"""
        loss_data = self.get_loss_curves()
        with open(save_path, 'w') as f:
            json.dump(loss_data, f, indent=2)
        print(f"Loss curves saved to {save_path}")
    
    def plot_loss_curves(self, save_path="./training_loss_curve.png", show_plot=True):
        """Plot and save training loss curves"""
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        if self.train_losses and self.train_steps:
            plt.plot(self.train_steps, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        
        # Plot validation loss
        if self.eval_losses and self.eval_steps:
            plt.plot(self.eval_steps, self.eval_losses, 'r-', label='Validation Loss', linewidth=2, marker='o', markersize=4)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add some styling
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curve plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return save_path

def plot_loss_curves_from_file(loss_file="./loss_curves.json", save_path="./training_loss_curve.png", show_plot=True):
    """
    Load and plot loss curves from saved JSON file
    
    Args:
        loss_file: Path to the saved loss curves JSON file
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    
    Returns:
        str: Path to saved plot
    """
    try:
        with open(loss_file, 'r') as f:
            loss_data = json.load(f)
        
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        if loss_data.get('train_losses') and loss_data.get('train_steps'):
            plt.plot(loss_data['train_steps'], loss_data['train_losses'], 
                    'b-', label='Training Loss', linewidth=2)
        
        # Plot validation loss
        if loss_data.get('eval_losses') and loss_data.get('eval_steps'):
            plt.plot(loss_data['eval_steps'], loss_data['eval_losses'], 
                    'r-', label='Validation Loss', linewidth=2, marker='o', markersize=4)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curve plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return save_path
        
    except FileNotFoundError:
        print(f"Loss curves file not found: {loss_file}")
        return None
    except Exception as e:
        print(f"Error plotting loss curves: {e}")
        return None

def analyze_training_progress(loss_file="./loss_curves.json"):
    """
    Analyze training progress from loss curves
    
    Args:
        loss_file: Path to the saved loss curves JSON file
    
    Returns:
        dict: Analysis results
    """
    try:
        with open(loss_file, 'r') as f:
            loss_data = json.load(f)
        
        analysis = {}
        
        # Analyze training loss
        if loss_data.get('train_losses'):
            train_losses = loss_data['train_losses']
            analysis['train_loss_start'] = train_losses[0] if train_losses else None
            analysis['train_loss_end'] = train_losses[-1] if train_losses else None
            analysis['train_loss_reduction'] = (train_losses[0] - train_losses[-1]) if len(train_losses) > 1 else 0
            analysis['train_loss_reduction_pct'] = (analysis['train_loss_reduction'] / train_losses[0] * 100) if train_losses[0] > 0 else 0
        
        # Analyze validation loss
        if loss_data.get('eval_losses'):
            eval_losses = loss_data['eval_losses']
            analysis['eval_loss_start'] = eval_losses[0] if eval_losses else None
            analysis['eval_loss_end'] = eval_losses[-1] if eval_losses else None
            analysis['eval_loss_reduction'] = (eval_losses[0] - eval_losses[-1]) if len(eval_losses) > 1 else 0
            analysis['eval_loss_reduction_pct'] = (analysis['eval_loss_reduction'] / eval_losses[0] * 100) if eval_losses[0] > 0 else 0
            
            # Check for overfitting (validation loss starts increasing)
            if len(eval_losses) >= 3:
                recent_trend = eval_losses[-3:]
                analysis['potential_overfitting'] = recent_trend[-1] > min(recent_trend)
            else:
                analysis['potential_overfitting'] = False
        
        # Training stability
        if loss_data.get('train_losses') and len(loss_data['train_losses']) > 10:
            recent_losses = loss_data['train_losses'][-10:]
            loss_variance = sum((x - sum(recent_losses)/len(recent_losses))**2 for x in recent_losses) / len(recent_losses)
            analysis['training_stability'] = 'stable' if loss_variance < 0.01 else 'unstable'
        
        # Print analysis
        print(f"\n{'='*50}")
        print("TRAINING PROGRESS ANALYSIS")
        print(f"{'='*50}")
        
        if 'train_loss_start' in analysis:
            print(f"Training Loss: {analysis['train_loss_start']:.4f} → {analysis['train_loss_end']:.4f}")
            print(f"Training Loss Reduction: {analysis['train_loss_reduction']:.4f} ({analysis['train_loss_reduction_pct']:.1f}%)")
        
        if 'eval_loss_start' in analysis:
            print(f"Validation Loss: {analysis['eval_loss_start']:.4f} → {analysis['eval_loss_end']:.4f}")
            print(f"Validation Loss Reduction: {analysis['eval_loss_reduction']:.4f} ({analysis['eval_loss_reduction_pct']:.1f}%)")
        
        if 'potential_overfitting' in analysis:
            if analysis['potential_overfitting']:
                print("⚠️  Potential overfitting detected (validation loss trending upward)")
            else:
                print("✅ No obvious overfitting detected")
        
        if 'training_stability' in analysis:
            print(f"Training Stability: {analysis['training_stability']}")
        
        return analysis
        
    except FileNotFoundError:
        print(f"Loss curves file not found: {loss_file}")
        return {}
    except Exception as e:
        print(f"Error analyzing training progress: {e}")
        return {}

def train_custom_model(model, tokenizer, train_dataset, eval_dataset=None, test_mode=True):
    """
    Train custom model with proper data collator handling for both tokenizer types
    
    Args:
        model: The custom GPT-2 model to train
        tokenizer: Either Hugging Face tokenizer or custom BPE tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        test_mode: Whether to use test mode (faster training)
    
    Returns:
        CustomTrainer: The trained model trainer
    """
    print("Setting up training configuration...")
    
    # Custom data collator for BPE tokenizers
    class CustomDataCollator:
        def __init__(self, pad_token_id=0):
            self.pad_token_id = pad_token_id
        
        def __call__(self, batch):
            # Handle both input_ids and labels
            input_ids = [item['input_ids'] for item in batch]
            
            # Find maximum length in batch
            max_len = max(len(ids) for ids in input_ids)
            
            # Pad sequences
            padded_input_ids = []
            attention_masks = []
            
            for ids in input_ids:
                padding_length = max_len - len(ids)
                padded_ids = ids + [self.pad_token_id] * padding_length
                attention_mask = [1] * len(ids) + [0] * padding_length
                
                padded_input_ids.append(padded_ids)
                attention_masks.append(attention_mask)
            
            return {
                'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
                'labels': torch.tensor(padded_input_ids, dtype=torch.long)  # For causal LM
            }
    
    # Choose appropriate data collator based on tokenizer type
    if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'get_vocab_size'):
        # Custom BPE tokenizer - use our custom data collator
        data_collator = CustomDataCollator(pad_token_id=0)
        print("Using custom data collator for BPE tokenizer")
    else:
        # Hugging Face tokenizer - use standard data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        print("Using Hugging Face data collator")
    
    # Configure training arguments based on test mode
    if test_mode:
        training_args = TrainingArguments(
            output_dir="./custom-gpt2-model",
            overwrite_output_dir=True,
            num_train_epochs=1,  # Quick training for testing
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            warmup_steps=50,
            logging_steps=10,
            save_steps=100,
            eval_steps=100 if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=1,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=False,  # Set to False initially for stability
            optim="adamw_torch",
            learning_rate=1e-3,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.95,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            report_to=[],  # Disable all external logging (wandb, tensorboard, etc.)
            run_name=None,  # Explicitly set to None to avoid wandb confusion
        )
    else:
        training_args = TrainingArguments(
            output_dir="./custom-gpt2-model",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=500,
            logging_steps=100,
            save_steps=1000,
            eval_steps=1000 if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=False,  # Set to False initially for stability
            optim="adamw_torch",
            learning_rate=5e-4,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.95,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            report_to=[],  # Disable all external logging (wandb, tensorboard, etc.)
            run_name=None,  # Explicitly set to None to avoid wandb confusion
        )
    
    # Initialize custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save model and loss curves
    trainer.save_model()
    trainer.save_loss_curves()
    trainer.plot_loss_curves()
    
    print("Training completed and model saved!")
    
    return trainer

# ==================== EVALUATION FUNCTIONS ====================

@torch.no_grad()
def estimate_loss(model, tokenized_datasets, tokenizer, eval_iters=100, batch_size=8):
    """
    Estimate loss on training and validation sets
    
    Args:
        model: The trained model
        tokenized_datasets: Dict with 'train' and 'validation' datasets
        tokenizer: The tokenizer used
        eval_iters: Number of batches to evaluate
        batch_size: Batch size for evaluation
    """
    model.eval()
    device = next(model.parameters()).device
    
    results = {}
    
    for split in ['train', 'validation']:
        if split not in tokenized_datasets:
            continue
            
        dataset = tokenized_datasets[split]
        losses = []
        
        # Create dataloader for evaluation
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=lambda x: {
                'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in x]),
                'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in x])
            }
        )
        
        # Evaluate for specified number of iterations
        for i, batch in enumerate(dataloader):
            if i >= eval_iters:
                break
                
            input_ids = batch['input_ids'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs['loss']
            
            if loss is not None:
                losses.append(loss.item())
        
        if losses:
            results[split] = sum(losses) / len(losses)
        else:
            results[split] = float('inf')
    
    model.train()  # Reset to training mode
    return results

@torch.no_grad()
def calculate_perplexity(model, tokenized_dataset, tokenizer, eval_iters=100, batch_size=8):
    """
    Calculate perplexity on a dataset
    
    Args:
        model: The trained model
        tokenized_dataset: Dataset to evaluate on
        tokenizer: The tokenizer used
        eval_iters: Number of batches to evaluate
        batch_size: Batch size for evaluation
    
    Returns:
        float: Perplexity score
    """
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0
    total_tokens = 0
    
    # Create dataloader for evaluation
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Don't shuffle for consistent evaluation
        collate_fn=lambda x: {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in x]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in x])
        }
    )
    
    # Evaluate for specified number of iterations
    for i, batch in enumerate(dataloader):
        if i >= eval_iters:
            break
            
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs['loss']
        
        if loss is not None:
            # Count actual tokens (excluding padding)
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    # Calculate average loss and convert to perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    model.train()  # Reset to training mode
    return perplexity

@torch.no_grad()
def evaluate_model_comprehensive(model, tokenized_datasets, tokenizer, model_name="Custom"):
    """
    Comprehensive evaluation including perplexity on all splits
    
    Args:
        model: The trained model
        tokenized_datasets: Dict with 'train', 'validation', 'test' datasets
        tokenizer: The tokenizer used
        model_name: Name for logging
    
    Returns:
        dict: Evaluation results
    """
    print(f"\n{'='*50}")
    print(f"COMPREHENSIVE EVALUATION: {model_name}")
    print(f"{'='*50}")
    
    results = {}
    
    # Evaluate on all available splits
    for split in ['train', 'validation', 'test']:
        if split not in tokenized_datasets:
            continue
            
        print(f"\nEvaluating on {split} set...")
        
        # Calculate loss
        loss_result = estimate_loss(model, {split: tokenized_datasets[split]}, tokenizer, eval_iters=50)
        loss = loss_result.get(split, float('inf'))
        
        # Calculate perplexity
        perplexity = calculate_perplexity(model, tokenized_datasets[split], tokenizer, eval_iters=50)
        
        results[split] = {
            'loss': loss,
            'perplexity': perplexity
        }
        
        print(f"{split.capitalize()} Loss: {loss:.4f}")
        print(f"{split.capitalize()} Perplexity: {perplexity:.2f}")
    
    return results

def compare_model_sizes(dataset, tokenizer, model_sizes=['tiny', 'small'], test_mode=True):
    """
    Compare different model sizes in terms of performance and efficiency
    
    Args:
        dataset: The dataset to train/evaluate on
        tokenizer: The tokenizer to use
        model_sizes: List of model sizes to compare
        test_mode: Whether to use test mode (faster training)
    
    Returns:
        dict: Comparison results
    """
    print(f"\n{'='*60}")
    print("MODEL SIZE COMPARISON")
    print(f"{'='*60}")
    
    comparison_results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for size in model_sizes:
        print(f"\n--- Training {size.upper()} model ---")
        
        # Create model with custom tokenizer
        model, custom_tokenizer = create_custom_gpt2_model(
            model_size=size, 
            use_custom_tokenizer=True, 
            dataset=dataset
        )
        model = model.to(device)
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024**2)
        
        # Tokenize data
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=128)
        
        # Train model (with timing)
        import time
        start_time = time.time()
        
        trainer = train_custom_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            test_mode=test_mode
        )
        
        training_time = time.time() - start_time
        
        # Comprehensive evaluation
        eval_results = evaluate_model_comprehensive(
            model, tokenized_dataset, tokenizer, model_name=f"{size.upper()} GPT-2"
        )
        
        # Test generation quality
        print(f"\nTesting generation quality for {size} model...")
        model.eval()
        generation_examples = []
        
        test_prompts = ["The future of AI", "Once upon a time", "Science shows that"]
        for prompt in test_prompts:
            try:
                # Handle different tokenizer types
                if hasattr(custom_tokenizer, 'encode') and hasattr(custom_tokenizer, 'get_vocab_size'):
                    # Custom BPE tokenizer
                    encoded = custom_tokenizer.encode(prompt)
                    inputs = torch.tensor([encoded.ids], dtype=torch.long).to(device)
                    pad_token_id = 0
                else:
                    # Hugging Face tokenizer
                    inputs = custom_tokenizer.encode(prompt, return_tensors="pt").to(device)
                    pad_token_id = custom_tokenizer.pad_token_id
                
                outputs = model.generate(
                    inputs, max_length=30, temperature=0.8, do_sample=True, 
                    pad_token_id=pad_token_id
                )
                
                # Decode based on tokenizer type
                if hasattr(custom_tokenizer, 'decode') and hasattr(custom_tokenizer, 'get_vocab_size'):
                    # Custom BPE tokenizer
                    generated = custom_tokenizer.decode(outputs[0].tolist())
                else:
                    # Hugging Face tokenizer
                    generated = custom_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                generation_examples.append(f"'{prompt}' → '{generated}'")
                print(f"  {generation_examples[-1]}")
            except Exception as e:
                error_msg = f"'{prompt}' → [Error: {str(e)}]"
                generation_examples.append(error_msg)
                print(f"  {error_msg}")
        
        # Store results
        comparison_results[size] = {
            'parameters': total_params,
            'size_mb': model_size_mb,
            'training_time': training_time,
            'evaluation': eval_results,
            'generation_examples': generation_examples
        }
        
        # Clean up memory
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Params':<12} {'Size(MB)':<10} {'Train Time':<12} {'Val Loss':<10} {'Val PPL':<10}")
    print("-" * 70)
    
    for size, results in comparison_results.items():
        params = f"{results['parameters']:,}"
        size_mb = f"{results['size_mb']:.1f}"
        train_time = f"{results['training_time']:.1f}s"
        val_loss = results['evaluation'].get('validation', {}).get('loss', 'N/A')
        val_ppl = results['evaluation'].get('validation', {}).get('perplexity', 'N/A')
        
        val_loss_str = f"{val_loss:.3f}" if val_loss != 'N/A' else 'N/A'
        val_ppl_str = f"{val_ppl:.1f}" if val_ppl != 'N/A' else 'N/A'
        
        print(f"{size:<10} {params:<12} {size_mb:<10} {train_time:<12} {val_loss_str:<10} {val_ppl_str:<10}")
    
    return comparison_results

# ==================== MAIN TRAINING PIPELINE ====================
def main_training_pipeline():
    """Complete training pipeline using custom Transformer and local WikiText files"""
    
    print("=" * 60)
    print("CUSTOM GPT-2 TRAINING WITH LOCAL WIKITEXT FILES")
    print("Using Custom Transformer Architecture Implementation")
    print("=" * 60)
    
    # Setup
    recommended_size = setup_training_environment()
    TEST_MODE = True  # Set to False for full training
    
    # Load data
    print("\nLoading WikiText data...")
    dataset = load_wikitext_from_local_files(test_mode=TEST_MODE)
    
    # Create custom model with BPE tokenizer as required
    print(f"\nCreating custom {recommended_size} model with BPE tokenizer...")
    model, tokenizer = create_custom_gpt2_model(
        model_size=recommended_size, 
        use_custom_tokenizer=True,  # Use BPE tokenizer as per requirements
        dataset=dataset
    )
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Tokenize
    print(f"\nTokenizing data...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=128)
    
    # Train
    print(f"\nTraining custom model...")
    trainer = train_custom_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        test_mode=TEST_MODE
    )
    
    # Comprehensive evaluation including perplexity
    print(f"\nPerforming comprehensive evaluation...")
    eval_results = evaluate_model_comprehensive(model, tokenized_dataset, tokenizer, "Custom GPT-2")
    
    # Analyze training progress from loss curves
    print(f"\nAnalyzing training progress...")
    training_analysis = analyze_training_progress("./loss_curves.json")
    
    # Evaluate different generation strategies
    print(f"\nEvaluating text generation strategies...")
    generation_results = evaluate_generation_strategies(
        model=model, 
        tokenizer=tokenizer, 
        max_length=80,
        device=device
    )
    
    # Quick generation test
    print(f"\nQuick generation examples...")
    model.eval()
    test_prompts = ["The", "Once upon", "Science"]
    
    for prompt in test_prompts:
        try:
            # Handle different tokenizer types
            if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'get_vocab_size'):
                # Custom BPE tokenizer
                encoded = tokenizer.encode(prompt)
                inputs = torch.tensor([encoded.ids], dtype=torch.long).to(device)
                pad_token_id = 0  # Use 0 for custom tokenizer
            else:
                # Hugging Face tokenizer
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
                pad_token_id = tokenizer.pad_token_id
            
            outputs = model.generate_with_strategies(
                inputs, max_length=50, strategy='nucleus', top_p=0.9, temperature=0.8,
                pad_token_id=pad_token_id
            )
            
            # Decode the output
            if hasattr(tokenizer, 'decode') and hasattr(tokenizer, 'get_vocab_size'):
                # Custom BPE tokenizer
                generated = tokenizer.decode(outputs[0].tolist())
            else:
                # Hugging Face tokenizer
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"'{prompt}' → '{generated}'")
        except Exception as e:
            print(f"Error generating text for prompt '{prompt}': {e}")
            continue
    
    print(f"\n" + "=" * 60)
    print("CUSTOM TRAINING COMPLETED!")
    print("Custom model saved to: ./custom-gpt2-model")
    print("Architecture includes:")
    print("(a) Token and learned position embeddings")
    print("(b) Multi-head masked self-attention layers") 
    print("(c) Feedforward layers with GELU activation")
    print("(d) Pre-layer normalization and residual connections")
    print("=" * 60)

def run_model_comparison():
    """Run comparison between different model sizes"""
    print("=" * 60)
    print("MODEL SIZE COMPARISON EXPERIMENT")
    print("=" * 60)
    
    # Load data
    dataset = load_wikitext_from_local_files(test_mode=True)
    
    # Create tokenizer
    _, tokenizer = create_custom_gpt2_model(model_size="tiny")
    
    # Compare different model sizes
    comparison_results = compare_model_sizes(
        dataset=dataset,
        tokenizer=tokenizer,
        model_sizes=['tiny', 'small'],  # Add more sizes as needed: ['tiny', 'small', 'medium']
        test_mode=True
    )
    
    return comparison_results

def evaluate_generation_strategies(model, tokenizer, prompts=None, max_length=100, device='cpu'):
    """
    Evaluate different text generation strategies and analyze fluency/diversity
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompts: List of prompts to test (default uses standard prompts)
        max_length: Maximum generation length
        device: Device to run on
    
    Returns:
        dict: Results for each strategy
    """
    if prompts is None:
        prompts = [
            "The future of artificial intelligence",
            "Once upon a time in a distant galaxy",
            "Climate change is affecting",
            "The most important discovery in science",
            "In the year 2050, technology will"
        ]
    
    strategies = {
        'greedy': {'strategy': 'greedy'},
        'top_k_10': {'strategy': 'top_k', 'top_k': 10},
        'top_k_50': {'strategy': 'top_k', 'top_k': 50},
        'nucleus_0.7': {'strategy': 'nucleus', 'top_p': 0.7},
        'nucleus_0.9': {'strategy': 'nucleus', 'top_p': 0.9},
        'nucleus_0.95': {'strategy': 'nucleus', 'top_p': 0.95},
        'random_low_temp': {'strategy': 'random', 'temperature': 0.5},
        'random_high_temp': {'strategy': 'random', 'temperature': 1.5}
    }
    
    print(f"\n{'='*80}")
    print("TEXT GENERATION STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    results = {}
    model.eval()
    
    for strategy_name, params in strategies.items():
        print(f"\n--- {strategy_name.upper().replace('_', ' ')} ---")
        strategy_results = {
            'generations': [],
            'unique_generations': set(),
            'avg_length': 0,
            'repetition_scores': []
        }
        
        total_length = 0
        
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}: '{prompt}'")
            
            # Generate multiple samples for diversity analysis
            for sample in range(3):  # Generate 3 samples per prompt
                try:
                    # Handle different tokenizer types
                    if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'get_vocab_size'):
                        # Custom BPE tokenizer
                        encoded = tokenizer.encode(prompt)
                        inputs = torch.tensor([encoded.ids], dtype=torch.long).to(device)
                        pad_token_id = 0
                    else:
                        # Hugging Face tokenizer
                        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
                        pad_token_id = tokenizer.pad_token_id
                    
                    # Set default parameters
                    generation_params = {
                        'input_ids': inputs,
                        'max_length': max_length,
                        'temperature': params.get('temperature', 1.0),
                        'strategy': params.get('strategy', 'greedy'),
                        'top_k': params.get('top_k', 50),
                        'top_p': params.get('top_p', 0.9),
                        'pad_token_id': pad_token_id
                    }
                    
                    outputs = model.generate_with_strategies(**generation_params)
                    
                    # Decode based on tokenizer type
                    if hasattr(tokenizer, 'decode') and hasattr(tokenizer, 'get_vocab_size'):
                        # Custom BPE tokenizer
                        generated_text = tokenizer.decode(outputs[0].tolist())
                    else:
                        # Hugging Face tokenizer
                        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                except Exception as e:
                    print(f"  Error generating text: {e}")
                    generated_text = f"[Error: {str(e)}]"
                
                # Calculate metrics
                gen_length = len(generated_text.split())
                total_length += gen_length
                
                # Check for repetition (simple metric: repeated trigrams)
                words = generated_text.lower().split()
                trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
                unique_trigrams = len(set(trigrams))
                total_trigrams = len(trigrams)
                repetition_score = 1 - (unique_trigrams / max(total_trigrams, 1))
                
                strategy_results['generations'].append(generated_text)
                strategy_results['unique_generations'].add(generated_text)
                strategy_results['repetition_scores'].append(repetition_score)
                
                if sample == 0:  # Show first sample
                    print(f"  → {generated_text}")
        
        # Calculate averages
        strategy_results['avg_length'] = total_length / len(strategy_results['generations'])
        strategy_results['avg_repetition'] = sum(strategy_results['repetition_scores']) / len(strategy_results['repetition_scores'])
        strategy_results['diversity_ratio'] = len(strategy_results['unique_generations']) / len(strategy_results['generations'])
        
        results[strategy_name] = strategy_results
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<15} {'Avg Length':<12} {'Diversity':<10} {'Repetition':<12} {'Quality Notes'}")
    print("-" * 80)
    
    for strategy_name, result in results.items():
        avg_len = f"{result['avg_length']:.1f}"
        diversity = f"{result['diversity_ratio']:.2f}"
        repetition = f"{result['avg_repetition']:.3f}"
        
        # Quality assessment
        if strategy_name == 'greedy':
            quality = "Deterministic, may repeat"
        elif 'top_k_10' in strategy_name:
            quality = "Conservative, coherent"
        elif 'top_k_50' in strategy_name:
            quality = "Balanced creativity"
        elif 'nucleus_0.7' in strategy_name:
            quality = "Focused, coherent"
        elif 'nucleus_0.9' in strategy_name:
            quality = "Good balance"
        elif 'nucleus_0.95' in strategy_name:
            quality = "More diverse"
        elif 'low_temp' in strategy_name:
            quality = "Conservative"
        else:
            quality = "Creative, may ramble"
        
        print(f"{strategy_name:<15} {avg_len:<12} {diversity:<10} {repetition:<12} {quality}")
    
    return results

def analyze_generation_quality(generations, tokenizer):
    """
    Analyze quality metrics for generated text
    
    Args:
        generations: List of generated texts
        tokenizer: Tokenizer for analysis
    
    Returns:
        dict: Quality metrics
    """
    metrics = {
        'avg_length': 0,
        'vocabulary_diversity': 0,
        'repetition_penalty': 0,
        'fluency_score': 0
    }
    
    if not generations:
        return metrics
    
    total_words = 0
    all_words = []
    total_repetition = 0
    
    for text in generations:
        words = text.lower().split()
        total_words += len(words)
        all_words.extend(words)
        
        # Calculate repetition (consecutive word repetition)
        repetitions = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
        total_repetition += repetitions / max(len(words), 1)
    
    # Calculate metrics
    metrics['avg_length'] = total_words / len(generations)
    metrics['vocabulary_diversity'] = len(set(all_words)) / len(all_words) if all_words else 0
    metrics['repetition_penalty'] = total_repetition / len(generations)
    
    # Simple fluency score (based on sentence structure)
    fluency_scores = []
    for text in generations:
        sentences = text.split('.')
        complete_sentences = sum(1 for s in sentences if len(s.strip().split()) >= 3)
        fluency_scores.append(complete_sentences / max(len(sentences), 1))
    
    metrics['fluency_score'] = sum(fluency_scores) / len(fluency_scores)
    
    return metrics

# ==================== EVALUATION FUNCTIONS ====================

def run_generation_analysis():
    """Run detailed analysis of text generation strategies"""
    print("=" * 60)
    print("TEXT GENERATION STRATEGY ANALYSIS")
    print("=" * 60)
    
    # Load a pre-trained model (or train a simple one)
    print("Setting up model for generation analysis...")
    model, tokenizer = create_custom_gpt2_model(model_size="tiny")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # For demonstration, you might want to load a pre-trained model here
    # or use a model that has been trained already
    
    print("Analyzing different generation strategies...")
    generation_results = evaluate_generation_strategies(
        model=model,
        tokenizer=tokenizer,
        prompts=[
            "The future of AI",
            "In a world where",
            "Scientists discovered",
            "The ancient mystery",
            "Technology has changed"
        ],
        max_length=60,
        device=device
    )
    
    # Detailed analysis
    print(f"\n{'='*60}")
    print("DETAILED STRATEGY ANALYSIS")
    print(f"{'='*60}")
    
    for strategy_name, results in generation_results.items():
        print(f"\n{strategy_name.upper().replace('_', ' ')}:")
        print(f"  Average Length: {results['avg_length']:.1f} words")
        print(f"  Diversity Ratio: {results['diversity_ratio']:.2f}")
        print(f"  Repetition Score: {results['avg_repetition']:.3f}")
        
        # Quality analysis
        quality_metrics = analyze_generation_quality(results['generations'], tokenizer)
        print(f"  Vocabulary Diversity: {quality_metrics['vocabulary_diversity']:.3f}")
        print(f"  Fluency Score: {quality_metrics['fluency_score']:.3f}")
    
    return generation_results

def run_loss_curve_analysis():
    """Run analysis of existing loss curves (useful if you have saved training data)"""
    print("=" * 60)
    print("TRAINING LOSS CURVE ANALYSIS")
    print("=" * 60)
    
    # Try to load and analyze existing loss curves
    loss_file = "./loss_curves.json"
    
    if os.path.exists(loss_file):
        print("Found existing loss curves data. Analyzing...")
        
        # Analyze training progress
        analysis = analyze_training_progress(loss_file)
        
        # Plot the curves
        plot_loss_curves_from_file(loss_file, save_path="./training_analysis.png", show_plot=False)
        
        print("\nLoss curve analysis completed!")
        print("Files generated:")
        print("  - training_analysis.png (loss curve plot)")
        print("  - Detailed analysis printed above")
        
        return analysis
    else:
        print(f"No loss curves file found at {loss_file}")
        print("Train a model first to generate loss curves.")
        return None

# ==================== RUN TRAINING ====================
if __name__ == "__main__":
    # Choose what to run
    RUN_TRAINING = True
    RUN_MODEL_COMPARISON = False  # Set to True to compare model sizes
    RUN_GENERATION_ANALYSIS = False  # Set to True to analyze generation strategies
    RUN_LOSS_ANALYSIS = False  # Set to True to analyze existing loss curves
    
    if RUN_TRAINING:
        main_training_pipeline()
    
    if RUN_MODEL_COMPARISON:
        run_model_comparison()
    
    if RUN_GENERATION_ANALYSIS:
        run_generation_analysis()
    
    if RUN_LOSS_ANALYSIS:
        run_loss_curve_analysis()