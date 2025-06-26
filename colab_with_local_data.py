#!/usr/bin/env python3
"""
GPT-2 Training in Google Colab with Local WikiText Data
"""

# ==================== IMPORTS ====================
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
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

# ==================== TRAINING FUNCTIONS ====================
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

def create_custom_gpt2_model(model_size="tiny"):
    """Create custom GPT-2 model with specified size"""
    
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if model_size == "small":  # GPT-2 Small (117M)
        config = CustomTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=1024, n_embd=768, n_layer=12, n_head=12, n_inner=3072
        )
    elif model_size == "medium":  # GPT-2 Medium (345M)
        config = CustomTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=1024, n_embd=1024, n_layer=24, n_head=16, n_inner=4096
        )
    elif model_size == "large":  # GPT-2 Large (762M)
        config = CustomTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=1024, n_embd=1280, n_layer=36, n_head=20, n_inner=5120
        )
    elif model_size == "xl":  # GPT-2 XL (1.5B)
        config = CustomTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=1024, n_embd=1600, n_layer=48, n_head=25, n_inner=6400
        )
    else:  # tiny (for testing)
        config = CustomTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=256, n_embd=128, n_layer=2, n_head=4, n_inner=512
        )
    
    model = CustomGPT2Model(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Custom model created: {total_params:,} parameters (~{total_params * 4 / (1024**2):.1f} MB)")
    
    return model, tokenizer

# ==================== CUSTOM TRAINER FOR CUSTOM MODEL ====================

class CustomTrainer(Trainer):
    """Custom trainer to work with our custom model"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss using our custom model's forward method"""
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels", input_ids)
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']
        
        return (loss, outputs) if return_outputs else loss

def tokenize_dataset(dataset, tokenizer, max_length=256):
    """Tokenize dataset for training"""
    
    def tokenize_function(examples):
        texts = [text for text in examples['text'] if text.strip()]
        if not texts:
            return {'input_ids': [], 'attention_mask': []}
        
        return tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
    
    # Tokenize with progress bar
    tokenized_dataset = {}
    for split in ['train', 'validation']:
        print(f"Tokenizing {split}...")
        tokenized_dataset[split] = dataset[split].map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            num_proc=1,  # Single process for stability
            desc=f"Tokenizing {split}"
        )
    
    return tokenized_dataset

def train_custom_model(model, tokenizer, train_dataset, eval_dataset, test_mode=True):
    """Train custom model with all specified requirements"""
    
    # Calculate total training steps for scheduler
    total_steps = len(train_dataset) // (8 * 2) * 1  # batch_size * grad_accum * epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    # Training arguments with all requirements
    if test_mode:
        training_args = TrainingArguments(
            output_dir="./custom-gpt2-model",
            overwrite_output_dir=True,
            num_train_epochs=1,
            
            # ✓ Choose batch size that fits GPU memory (8 or 16)
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            
            # ✓ Use gradient accumulation to simulate larger batches
            gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
            
            # ✓ Learning rate scheduling: warmup + cosine decay
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",  # Cosine decay after warmup
            learning_rate=5e-5,
            
            # ✓ AdamW optimizer with weight decay
            optim="adamw_torch",
            weight_decay=0.01,
            
            # ✓ Report loss at regular intervals
            logging_steps=10,  # Every X gradient updates
            eval_steps=50,
            eval_strategy="steps",
            
            # ✓ Model checkpointing
            save_steps=100,
            save_total_limit=3,  # Keep last 3 checkpoints
            
            prediction_loss_only=True,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=[],
        )
    else:
        # Full training configuration
        total_steps = len(train_dataset) // (16 * 4) * 2  # batch_size * grad_accum * epochs
        warmup_steps = int(0.1 * total_steps)
        
        training_args = TrainingArguments(
            output_dir="./custom-gpt2-model",
            overwrite_output_dir=True,
            num_train_epochs=2,
            
            # ✓ Larger batch size for full training
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            
            # ✓ Gradient accumulation
            gradient_accumulation_steps=4,  # Effective batch size = 16 * 4 = 64
            
            # ✓ Learning rate scheduling
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            learning_rate=3e-5,
            
            # ✓ AdamW with weight decay
            optim="adamw_torch",
            weight_decay=0.01,
            
            # ✓ Regular reporting and checkpointing
            logging_steps=50,
            eval_steps=200,
            eval_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            
            prediction_loss_only=True,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=[],
        )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Custom Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    trainer.save_model()
    
    print("[OK] Training completed! Model saved to ./custom-gpt2-model")
    return trainer

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
            if example['text'].strip():
                yield example['text']
    
    # Train the tokenizer on dataset
    tokenizer.train_from_iterator(batch_iterator(), trainer)
    
    # Save the trained tokenizer
    tokenizer.save(f"{save_path}.json")
    print(f"[OK] Custom BPE tokenizer saved to {save_path}.json")
    
    return tokenizer

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
        
        # Create model
        model, _ = create_custom_gpt2_model(model_size=size)
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
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                inputs, max_length=30, temperature=0.8, do_sample=True, 
                pad_token_id=tokenizer.pad_token_id
            )
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generation_examples.append(f"'{prompt}' → '{generated}'")
            print(f"  {generation_examples[-1]}")
        
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
    
    # Create custom model
    print(f"\nCreating custom {recommended_size} model...")
    model, tokenizer = create_custom_gpt2_model(model_size=recommended_size)
    
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
    
    # Test generation
    print(f"\nTesting text generation...")
    model.eval()
    test_prompts = ["The", "Once upon", "Science"]
    
    for prompt in test_prompts:
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs, max_length=50, temperature=0.8, do_sample=True, 
            pad_token_id=tokenizer.pad_token_id
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"'{prompt}' → '{generated}'")
    
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

# ==================== RUN TRAINING ====================
if __name__ == "__main__":
    # Choose what to run
    RUN_TRAINING = True
    RUN_MODEL_COMPARISON = False  # Set to True to compare model sizes
    
    if RUN_TRAINING:
        main_training_pipeline()
    
    if RUN_MODEL_COMPARISON:
        run_model_comparison()