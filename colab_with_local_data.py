#!/usr/bin/env python3
"""
GPT-2 Training in Google Colab with Local WikiText Data
"""



# ==================== IMPORTS ====================
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    GPT2Config, GPT2LMHeadModel, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

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

def create_gpt2_model(model_size="tiny"):
    """Create GPT-2 model with specified size"""
    
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if model_size == "tiny":
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=256, n_embd=128, n_layer=2, n_head=4, n_inner=512,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    elif model_size == "small":
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=512, n_embd=256, n_layer=4, n_head=8, n_inner=1024,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    else:  # medium
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=1024, n_embd=512, n_layer=6, n_head=8, n_inner=2048,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    model = GPT2LMHeadModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created: {total_params:,} parameters (~{total_params * 4 / (1024**2):.1f} MB)")
    
    return model, tokenizer

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

def train_model(model, tokenizer, train_dataset, eval_dataset, test_mode=True):
    """Train model with optimized settings"""
    
    # Training arguments
    if test_mode:
        training_args = TrainingArguments(
            output_dir="./gpt2-model",
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
            output_dir="./gpt2-model",
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
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    trainer.save_model()
    
    print("[OK] Training completed! Model saved to ./gpt2-model")
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
    """Complete training pipeline using local WikiText files"""
    
    print("=" * 60)
    print("GPT-2 TRAINING WITH LOCAL WIKITEXT FILES")
    print("Using WikiText-2-raw-v1 Data from Current Directory")
    print("=" * 60)
    
    # Setup
    recommended_size = setup_training_environment()
    TEST_MODE = True  # Set to False for full training
    
    # Load data
    print("\nLoading WikiText data...")
    dataset = load_wikitext_from_local_files(test_mode=TEST_MODE)
    
    # Create model
    print(f"\nCreating {recommended_size} model...")
    model, tokenizer = create_gpt2_model(model_size=recommended_size)
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Tokenize
    print(f"\nTokenizing data...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=256)
    
    # Train
    print(f"\nTraining model...")
    trainer = train_model(
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
        with torch.no_grad():
            outputs = model.generate(
                inputs, max_length=50, num_return_sequences=1,
                temperature=0.8, do_sample=True, pad_token_id=tokenizer.pad_token_id
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"'{prompt}' → '{generated}'")
    
    print(f"\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("Model saved to: ./gpt2-model")
    print("You can find the trained model in the gpt2-model directory")
    print("=" * 60)

# ==================== RUN TRAINING ====================
if __name__ == "__main__":
    main_training_pipeline() 