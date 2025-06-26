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

# Setup cache directories
setup_cache_directories()

# Additional environment settings to prevent pattern issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism
os.environ["HF_DATASETS_OFFLINE"] = "0"  # Ensure online mode

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
def load_wikitext2_dataset(test_mode=False):
    """Load WikiText-2-raw-v1 dataset"""
    print("Loading WikiText-2-raw-v1 dataset...")

    try:
        # Try loading with explicit cache directory
        cache_dir = "/tmp/hf_datasets_cache"
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)
    except Exception as e:
        print(f"First attempt failed: {e}")
        try:
            # Try without any cache directory
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=None)
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            try:
                # Try with default cache
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
            except Exception as e3:
                print(f"All attempts failed. Last error: {e3}")
                raise e3
    
    if test_mode:
        print("TEST MODE: Using small subset of data for quick verification")
        train_size = min(100, len(dataset['train']))  # Only 100 training samples
        val_size = min(20, len(dataset['validation'])) # Only 20 validation samples
        
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