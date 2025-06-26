"""
GPT-2 Training - Google Colab


Features:
- Package installation
- GPU detection and optimization
- Multiple model sizes (tiny, small, original)
- Test mode for quick verification
- Complete training pipeline in one file
"""

# ==================== PACKAGE INSTALLATION ====================
# Install required packages for Colab
import os
import sys

# Run this pip install command first:
# !pip install datasets transformers tokenizers torch tqdm accelerate
print("Make sure packages are installed before running this script!")

# ==================== IMPORTS ====================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
try:
    from datasets import load_dataset
    print("datasets library loaded successfully")
except ImportError as e:
    print(f"Failed to import datasets: {e}")
    print("Run: !pip install datasets")
    sys.exit(1)

try:
    from transformers import (
        GPT2Config, 
        GPT2LMHeadModel, 
        GPT2Tokenizer,
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling
    )
    print("transformers library loaded successfully")
except ImportError as e:
    print(f"Failed to import transformers: {e}")
    print("Run: !pip install transformers")
    sys.exit(1)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import json
import math
from tqdm import tqdm
import numpy as np

# Set environment variables to handle dataset loading issues
def setup_cache_directories():
    """Setup cache directories safely"""
    cache_dirs = {
        "HF_DATASETS_CACHE": "/tmp/hf_datasets_cache",
        "TRANSFORMERS_CACHE": "/tmp/transformers_cache", 
        "HF_HOME": "/tmp/hf_home"
    }
    
    for env_var, path in cache_dirs.items():
        try:
            os.makedirs(path, exist_ok=True)
            os.environ[env_var] = path
            print(f"Set {env_var} to {path}")
        except Exception as e:
            print(f"Could not set {env_var}: {e}")
            # Try to use default cache instead
            if env_var in os.environ:
                del os.environ[env_var]

# Setup cache directories (optional - will be bypassed if problematic)
try:
    setup_cache_directories()
except Exception as e:
    print(f"Cache setup failed (will use fallback): {e}")

# Additional environment settings to prevent pattern issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism
os.environ["HF_DATASETS_OFFLINE"] = "0"  # Ensure online mode
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"  # Allow remote code execution

# ==================== ENVIRONMENT SETUP ====================
def check_environment():
    """Check GPU availability and environment"""
    try:
        import google.colab
        print("Running in Google Colab")
        in_colab = True
    except ImportError:
        print(" Running locally")
        in_colab = False
    
    print(f" Python version: {sys.version}")
    print(f" PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check GPU memory
        if gpu_memory < 8:
            print("Limited GPU memory detected. Using 'tiny' model is recommended.")
        elif gpu_memory < 16:
            print("Moderate GPU memory. 'small' model recommended.")
        else:
            print("High GPU memory. Any model size should work.")
    else:
        print("No GPU available - training will be slow on CPU")
    
    return in_colab

# ==================== DATASET CLASS ====================
class WikiTextDataset(Dataset):
    def __init__(self, encodings, max_length=1024):
        self.encodings = encodings
        self.max_length = max_length
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx], dtype=torch.long)
        }

# ==================== TOKENIZER FUNCTIONS ====================
def train_custom_bpe_tokenizer(dataset, vocab_size=8000, save_path="custom_tokenizer"):
    """Train a custom BPE tokenizer on the dataset"""
    print(f"Training custom BPE tokenizer with vocab size {vocab_size}...")

    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"]
    )

    def batch_iterator():
        for example in dataset:
            if example['text'].strip():
                yield example['text']
    
    tokenizer.train_from_iterator(batch_iterator(), trainer)
    tokenizer.save(f"{save_path}.json")
    print(f"Custom tokenizer saved to {save_path}.json")
    
    return tokenizer

# ==================== DATASET FUNCTIONS ====================

def create_simple_text_dataset(test_mode=False):
    """Create a simple synthetic dataset as fallback"""
    print("📝 Creating synthetic fallback dataset...")
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In machine learning, neural networks are computational models.",
        "Python is a versatile programming language for data science.",
        "Artificial intelligence is transforming various industries.",
        "Deep learning models require large amounts of training data.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret visual information.",
        "Reinforcement learning helps agents learn through interaction.",
        "Data preprocessing is crucial for machine learning success.",
        "Feature engineering improves model performance significantly."
    ] * (10 if test_mode else 100)
    
    from datasets import Dataset, DatasetDict
    
    # Create train/validation/test splits
    train_size = len(sample_texts) // 2
    val_size = len(sample_texts) // 4
    
    train_data = [{"text": text} for text in sample_texts[:train_size]]
    val_data = [{"text": text} for text in sample_texts[train_size:train_size + val_size]]
    test_data = [{"text": text} for text in sample_texts[train_size + val_size:]]
    
    return DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })

def load_from_local_parquet(data_dir=".", test_mode=False):
    """Load WikiText dataset from local parquet files (same directory as notebook)"""
    import pandas as pd
    from datasets import Dataset, DatasetDict
    
    print(f"📂 Loading WikiText from local parquet files in {os.path.abspath(data_dir)}...")
    
    files = {
        "train": "train-00000-of-00001.parquet",
        "validation": "validation-00000-of-00001.parquet", 
        "test": "test-00000-of-00001.parquet"
    }
    
    datasets = {}
    for split, filename in files.items():
        file_path = os.path.join(data_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            print("💡 Run the download step first or use load_wikitext2_dataset() instead")
            return None
        
        try:
            # Read parquet file
            df = pd.read_parquet(file_path)
            print(f"✓ Loaded {split}: {len(df)} samples")
            
            # Convert to Hugging Face Dataset
            datasets[split] = Dataset.from_pandas(df)
            
        except Exception as e:
            print(f"❌ Failed to load {split}: {e}")
            return None
    
    # Create DatasetDict
    dataset = DatasetDict(datasets)
    
    # Apply test mode filtering if needed
    if test_mode:
        print("🧪 TEST MODE: Using subset of data")
        dataset['train'] = dataset['train'].select(range(min(100, len(dataset['train']))))
        dataset['validation'] = dataset['validation'].select(range(min(20, len(dataset['validation']))))
    
    print(f"📊 Final dataset sizes:")
    print(f"   Train: {len(dataset['train']):,} samples")
    print(f"   Validation: {len(dataset['validation']):,} samples")
    print(f"   Test: {len(dataset['test']):,} samples")
    
    return dataset

def download_wikitext_parquet_files(data_dir="."):
    """Download WikiText-2-raw-v1 parquet files directly from Hugging Face"""
    import urllib.request
    import subprocess
    
    base_url = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/"
    files = {
        "train": "train-00000-of-00001.parquet",
        "validation": "validation-00000-of-00001.parquet", 
        "test": "test-00000-of-00001.parquet"
    }
    
    print(f"📥 Downloading WikiText-2-raw-v1 dataset files to {os.path.abspath(data_dir)}...")
    
    downloaded_files = {}
    for split, filename in files.items():
        url = base_url + filename
        local_path = os.path.join(data_dir, filename)
        
        if os.path.exists(local_path):
            print(f"✓ {filename} already exists")
        else:
            print(f"⬇️ Downloading {filename}...")
            try:
                # Try to detect if we're in Colab
                try:
                    import google.colab
                    # Use subprocess to run wget in Colab
                    subprocess.run(['wget', '-q', url, '-O', local_path], check=True)
                except ImportError:
                    # Use urllib for local environments
                    urllib.request.urlretrieve(url, local_path)
                print(f"✓ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
                return None
        
        downloaded_files[split] = local_path
    
    return downloaded_files

def load_wikitext2_dataset(test_mode=False, use_local_files=True):
    """
    Load WikiText-2-raw-v1 dataset with robust error handling
    
    Tries multiple loading methods in order:
    1. Local parquet files (fastest, most reliable)
    2. Salesforce/wikitext with cache
    3. Legacy wikitext with cache  
    4. Salesforce/wikitext without cache
    5. Legacy wikitext without cache
    6. Streaming mode
    7. Synthetic fallback dataset
    
    Args:
        test_mode (bool): If True, uses only subset of data for quick testing
        use_local_files (bool): If True, try local parquet files first
        
    Returns:
        DatasetDict: Loaded dataset with train/validation/test splits
    """
    print("🔄 Loading WikiText-2-raw-v1 dataset...")
    
    # Method 1: Try local parquet files first (Colab-optimized)
    if use_local_files:
        print("🔍 Checking for local parquet files...")
        
        # Try to download if files don't exist in current directory
        data_dir = "."  # Current directory (same as notebook)
        parquet_files = ["train-00000-of-00001.parquet", "validation-00000-of-00001.parquet", "test-00000-of-00001.parquet"]
        
        if not any(os.path.exists(f) for f in parquet_files):
            print("📥 Local files not found. Downloading...")
            download_result = download_wikitext_parquet_files(data_dir)
            if download_result is None:
                print("❌ Download failed, falling back to Hugging Face datasets...")
            else:
                dataset = load_from_local_parquet(data_dir, test_mode)
                if dataset is not None:
                    return dataset
        else:
            print("📁 Found local files, loading...")
            dataset = load_from_local_parquet(data_dir, test_mode)
            if dataset is not None:
                return dataset
    
    print("🔄 Local files unavailable, trying Hugging Face datasets library...")

    # Clear any problematic environment variables
    problematic_vars = ['HF_DATASETS_CACHE', 'TRANSFORMERS_CACHE', 'HF_HOME']
    original_values = {}
    
    try:
        # Temporarily remove cache environment variables
        for var in problematic_vars:
            if var in os.environ:
                original_values[var] = os.environ[var]
                del os.environ[var]
        
        print("Attempting to load dataset...")
        
        # Import here to avoid caching issues
        import tempfile
        import shutil
        
        # Create a clean temporary directory
        temp_cache = tempfile.mkdtemp(prefix="hf_cache_")
        print(f"Using temporary cache: {temp_cache}")
        
        # Try multiple loading methods in order of preference
        loading_methods = [
            ("Salesforce/wikitext", temp_cache, "Salesforce path with cache"),
            ("wikitext", temp_cache, "legacy path with cache"),
            ("Salesforce/wikitext", None, "Salesforce path without cache"),
            ("wikitext", None, "legacy path without cache")
        ]
        
        dataset = None
        for dataset_name, cache_dir, method_desc in loading_methods:
            try:
                print(f"Trying {method_desc}...")
                dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", cache_dir=cache_dir)
                print(f"✓ Dataset loaded successfully with {method_desc}")
                break
            except Exception as e:
                print(f"Failed {method_desc}: {e}")
                continue
        
        # If all regular methods failed, try streaming
        if dataset is None:
            try:
                print("Trying streaming mode...")
                dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", streaming=True)
                # Convert streaming to regular dataset
                train_data = list(dataset['train'].take(1000 if not test_mode else 100))
                val_data = list(dataset['validation'].take(200 if not test_mode else 20))
                test_data = list(dataset['test'].take(200 if not test_mode else 20))
                
                # Create a simple dataset structure
                from datasets import Dataset, DatasetDict
                dataset = DatasetDict({
                    'train': Dataset.from_list(train_data),
                    'validation': Dataset.from_list(val_data),
                    'test': Dataset.from_list(test_data)
                })
                print("✓ Dataset loaded successfully with streaming")
            except Exception as e:
                print(f"Streaming failed: {e}")
                # Final fallback: create a simple synthetic dataset
                print(f"All loading methods failed. Using fallback synthetic dataset...")
                dataset = create_simple_text_dataset(test_mode)
        
        # Clean up temp directory
        try:
            if 'temp_cache' in locals():
                shutil.rmtree(temp_cache, ignore_errors=True)
        except:
            pass
            
    finally:
        # Restore original environment variables
        for var, value in original_values.items():
            os.environ[var] = value
    
    if test_mode and dataset is not None and hasattr(dataset['train'], 'select'):
        print("TEST MODE: Using small subset of data for quick verification")
        train_size = min(100, len(dataset['train']))
        val_size = min(20, len(dataset['validation']))
        
        dataset['train'] = dataset['train'].select(range(train_size))
        dataset['validation'] = dataset['validation'].select(range(val_size))

    print(f"Train: {len(dataset['train'])} samples")
    print(f"Validation: {len(dataset['validation'])} samples") 
    print(f"Test: {len(dataset['test'])} samples")

    return dataset

def prepare_dataset_for_training(dataset, tokenizer, max_length=512, use_custom_tokenizer=False):
    """Prepare dataset for training with proper tokenization"""
    def tokenize_function(examples):
        texts = [text for text in examples['text'] if text.strip()]
        
        if not texts:
            return {'input_ids': [], 'attention_mask': []}

        if use_custom_tokenizer:
            encodings = []
            attention_masks = []

            for text in texts:
                encoded = tokenizer.encode(text)
                tokens = encoded.ids[:max_length]   
            
                if len(tokens) < max_length:
                    attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
                    tokens = tokens + [1] * (max_length - len(tokens))
                else:
                    attention_mask = [1] * max_length
                
                encodings.append(tokens)
                attention_masks.append(attention_mask)
        
            return {
                'input_ids': encodings,
                'attention_mask': attention_masks
            }
        else:
            return tokenizer(
                texts, 
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors=None
            )
    
    # Simple dataset mapping without multiprocessing  
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1,  # Force single process to avoid pattern issues
        desc="Tokenizing dataset"
    )

    return tokenized_dataset

# ==================== MODEL FUNCTIONS ====================
def create_gpt2_model(vocab_size=None, use_pretrained=False, model_size="tiny"):
    """Create GPT-2 model with different sizes"""
    if use_pretrained:
        print("📥 Loading pre-trained GPT-2 model...")
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
        return model, tokenizer

    else:
        print(f"Creating new GPT-2 model ({model_size}) with vocab size {vocab_size}")

        if model_size == "tiny":
            config = GPT2Config(
                vocab_size=vocab_size,
                n_positions=256,   # Smaller context window
                n_embd=128,        # Much smaller embedding
                n_layer=2,         # Only 2 layers
                n_head=4,          # 4 attention heads
                n_inner=512,       # Smaller feed-forward
                activation_function='gelu_new',
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
                layer_norm_epsilon=1e-5,
                initializer_range=0.02,
                use_cache=True,
                bos_token_id=0,
                eos_token_id=0,
                pad_token_id=1
            )
        elif model_size == "small":
            config = GPT2Config(
                vocab_size=vocab_size,
                n_positions=512,
                n_embd=256,
                n_layer=4,
                n_head=8,
                n_inner=1024,
                activation_function='gelu_new',
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
                layer_norm_epsilon=1e-5,
                initializer_range=0.02,
                use_cache=True,
                bos_token_id=0,
                eos_token_id=0,
                pad_token_id=1
            )
        else:  # original size
            config = GPT2Config(
                vocab_size=vocab_size,
                n_positions=1024,
                n_embd=512,
                n_layer=6,
                n_head=8,
                n_inner=2048,
                activation_function='gelu_new',
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
                layer_norm_epsilon=1e-5,
                initializer_range=0.02,
                use_cache=True,
                bos_token_id=0,
                eos_token_id=0,
                pad_token_id=1
            )

        model = GPT2LMHeadModel(config)
        return model, config

# ==================== TRAINING FUNCTIONS ====================
def train_model(model, tokenizer, train_dataset, eval_dataset=None, output_dir="./gpt2-wikitext2", colab_optimized=True):
    """Train model using Hugging Face Trainer"""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer if hasattr(tokenizer, 'pad_token') else None,
        mlm=False,
        pad_to_multiple_of=8
    )

    if colab_optimized:
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=1, 
            per_device_train_batch_size=2, # small batch size
            per_device_eval_batch_size=2, 
            gradient_accumulation_steps=2, 
            warmup_steps=50, #quck warmup
            logging_steps=10, # Frequent logging
            save_steps=100, # Save frequently
            eval_steps=100,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=1,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            optim="adamw_torch",
            learning_rate=1e-3, # Higher learning rate
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.95,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            report_to=None,
        )
    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=500,
            logging_steps=100,
            save_steps=1000,
            eval_steps=1000,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            optim="adamw_torch",
            learning_rate=5e-4,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.95,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            report_to=None,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
        
    print("🏃‍♂️ Starting training...")
    trainer.train()
    trainer.save_model()
    print("Training completed and model saved!")

    return trainer

# ==================== TEXT GENERATION ====================
def generate_text(model, tokenizer, prompt="The", max_length=50):
    """Generate text using the trained model"""
    model.eval()
    device = next(model.parameters()).device

    if hasattr(tokenizer, 'encode'):
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
        )

    if hasattr(tokenizer, 'decode'):
        generated_text = tokenizer.decode(outputs[0].tolist())
    else:
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

# ==================== MAIN TRAINING PIPELINE ====================
def main_training_pipeline(test_mode=True, model_size="tiny", use_custom_tokenizer=False):
    """Complete training pipeline"""
    
    print("=" * 60)
    print("GPT-2 TRAINING PIPELINE")
    print("=" * 60)
    
    # Check environment
    in_colab = check_environment()
    
    # Configuration
    if test_mode:
        VOCAB_SIZE = 5000
        MAX_LENGTH = 128
        USE_PRETRAINED = False
        USE_CUSTOM_TOKENIZER = False  # Simplified for testing
        print("RUNNING IN TEST MODE")
        print(f"Vocab size: {VOCAB_SIZE}, Max length: {MAX_LENGTH}")
    else:
        VOCAB_SIZE = 16000
        MAX_LENGTH = 256
        USE_PRETRAINED = False
        USE_CUSTOM_TOKENIZER = use_custom_tokenizer
        print("RUNNING IN FULL MODE")
        print(f"Vocab size: {VOCAB_SIZE}, Max length: {MAX_LENGTH}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_wikitext2_dataset(test_mode=test_mode)

    # Setup tokenizer and model
    if USE_CUSTOM_TOKENIZER and not USE_PRETRAINED:
        tokenizer_path = "wikitext2_custom_tokenizer"
        if not os.path.exists(f"{tokenizer_path}.json"):
            custom_tokenizer = train_custom_bpe_tokenizer(
                dataset['train'],
                VOCAB_SIZE,
                tokenizer_path
            )
        else:
            print(f"Loading existing custom tokenizer from {tokenizer_path}.json")
            custom_tokenizer = Tokenizer.from_file(f"{tokenizer_path}.json")

        actual_vocab_size = custom_tokenizer.get_vocab_size()
        print(f"Custom tokenizer vocab size: {actual_vocab_size}")

        model, config = create_gpt2_model(vocab_size=actual_vocab_size, use_pretrained=False, model_size=model_size)

        train_dataset = prepare_dataset_for_training(
            dataset['train'],
            custom_tokenizer,
            MAX_LENGTH,
            use_custom_tokenizer=True
        )
        eval_dataset = prepare_dataset_for_training(
            dataset['validation'],
            custom_tokenizer,
            MAX_LENGTH,
            use_custom_tokenizer=True
        )

        tokenizer_for_training = None
        final_tokenizer = custom_tokenizer

    else:
        if USE_PRETRAINED:
            model, hf_tokenizer = create_gpt2_model(use_pretrained=True)
        else:
            hf_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
            if hf_tokenizer.pad_token is None:
                hf_tokenizer.pad_token = hf_tokenizer.eos_token
            
            model, config = create_gpt2_model(
                vocab_size=hf_tokenizer.vocab_size,
                use_pretrained=False,
                model_size=model_size
            )
        
        train_dataset = prepare_dataset_for_training(
            dataset['train'],
            hf_tokenizer,
            MAX_LENGTH,
            use_custom_tokenizer=False
        )
        eval_dataset = prepare_dataset_for_training(
            dataset['validation'],
            hf_tokenizer,
            MAX_LENGTH,
            use_custom_tokenizer=False
        )
        tokenizer_for_training = hf_tokenizer
        final_tokenizer = hf_tokenizer
    
    # Move model to device
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Estimated model size: ~{total_params * 4 / (1024**2):.1f} MB")

    # Train model
    trainer = train_model(
        model=model,
        tokenizer=tokenizer_for_training,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        colab_optimized=in_colab or test_mode
    )

    # Test text generation
    print("\n" + "=" * 60)
    print("TESTING TEXT GENERATION")
    print("=" * 60)
    
    test_prompts = ["The", "Once upon", "In the", "Today", "Science"]

    for prompt in test_prompts:
        try:
            generated = generate_text(model, final_tokenizer, prompt, max_length=30)
            print(f"'{prompt}' → '{generated}'")
        except Exception as e:
            print(f"Error with prompt '{prompt}': {e}")

    # Save info
    os.makedirs("./gpt2-wikitext2", exist_ok=True)
    tokenizer_info = {
        "use_custom_tokenizer": USE_CUSTOM_TOKENIZER,
        "use_pretrained": USE_PRETRAINED,
        "vocab_size": actual_vocab_size if USE_CUSTOM_TOKENIZER else hf_tokenizer.vocab_size,
        "max_length": MAX_LENGTH,
        "model_size": model_size,
        "test_mode": test_mode,
        "total_parameters": total_params
    }
    
    with open("./gpt2-wikitext2/training_info.json", "w") as f:
        json.dump(tokenizer_info, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("Model saved in './gpt2-wikitext2/' directory")
    print("=" * 60)
    
    return model, final_tokenizer

# ==================== RUN THE TRAINING ====================
if __name__ == "__main__":
    print("Choose your training mode:")
    print("1. Quick Test (tiny model, 100 samples, ~2-3 minutes)")
    print("2. Small Model (full dataset, ~10-15 minutes)")
    print("3. Full Model (requires good GPU, ~30+ minutes)")
    
    # QUICK TEST MODE (Recommended for first run)
    print("\nStarting QUICK TEST mode...")
    print("This trains a tiny model on 100 samples to verify everything works.")
    print("Expected time: 2-3 minutes\n")
    
    try:
        model, tokenizer = main_training_pipeline(
            test_mode=True, 
            model_size="tiny",
            use_custom_tokenizer=False
        )
        
        print("\nQuick test completed successfully!")
        print("If this worked, you can try larger models by changing the parameters below:")
        print("   - For small model: test_mode=False, model_size='small'")
        print("   - For full model: test_mode=False, model_size='original'")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Try reducing model size or enabling test mode")
        
    # Uncomment below for other modes:
    
    # SMALL MODEL MODE 
    # model, tokenizer = main_training_pipeline(
    #     test_mode=False, 
    #     model_size="small",
    #     use_custom_tokenizer=False
    # )
    
    # FULL MODEL MODE (Requires good GPU)
    # model, tokenizer = main_training_pipeline(
    #     test_mode=False, 
    #     model_size="original",
    #     use_custom_tokenizer=True
    # ) 