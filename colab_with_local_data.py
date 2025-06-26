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
        
        # Apply multi-head attention with causal mask
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
    """Train custom model with optimized settings"""
    
    # Training arguments
    if test_mode:
        training_args = TrainingArguments(
            output_dir="./custom-gpt2-model",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=10,
            logging_steps=5,
            save_steps=50,
            eval_steps=50,
            eval_strategy="steps",
            save_total_limit=1,
            prediction_loss_only=True,
            fp16=True,  # Use mixed precision for speed
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=[],  # Disable wandb and other logging
        )
    else:
        training_args = TrainingArguments(
            output_dir="./custom-gpt2-model",
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            eval_strategy="steps",
            save_total_limit=2,
            prediction_loss_only=True,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=[],  # Disable wandb and other logging
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
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=256)
    
    # Train
    print(f"\nTraining custom model...")
    trainer = train_custom_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        test_mode=TEST_MODE
    )
    
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

# ==================== RUN TRAINING ====================
if __name__ == "__main__":
    main_training_pipeline() 