import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
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
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
import json
import math
from tqdm import tqdm
import numpy as np
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
    
"""Train a custom BPE tokenizer on the dataset"""
def train_custom_bpe_tokenizer(dataset, vocab_size=32000, save_path="custom_tokenizer"):
    print(f"Training custom BPE tokenizer with vocab size {vocab_size}...")

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = Whitespace()

    # Configure trainer with GPT-2 style special tokens
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"]
    )

    # Extract texts from dataset
    def batch_iterator():
        for example in dataset:
            if example['text'].strip(): # Skip empty texts
                yield example['text']
    
    # Train tokenizer
    tokenizer.train_from_iterator(batch_iterator(), trainer)

    # Save tokenizer
    tokenizer.save(f"{save_path}.json")
    print(f"Custom tokenizer saved to {save_path}.json")
    
    return tokenizer

"""Load WikiText-2-raw-v1 dataset using Hugging Face datasets"""
def load_wikitext2_dataset():
    print("Loading WikiText-2-raw-v1 dataset...")

     # Load dataset from Hugging Face
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    print(f"Train: {len(dataset['train'])} samples")
    print(f"Validation: {len(dataset['validation'])} samples")
    print(f"Test:{len(dataset['test'])} samples")

    return dataset

"""Prepare dataset for training with proper tokenization"""
def prepare_dataset_for_training(dataset, tokenizer, max_length=512, use_custom_tokenizer=False):
    def tokenize_function(examples):
        # Filter out empty texts
        texts = [text for text in examples['text'] if text.strip()]

        if use_custom_tokenizer:
            # For custom tokenizer
            encodings = []
            attention_masks = []

            for text in texts:
                encoded = tokenizer.encode(text)
                tokens = encoded.ids[:max_length]   
            
                # Pad to max_length
                if len(tokens) < max_length:
                    # Create attention mask: 1s for real tokens, 0s for padding (ignore)
                    attention_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
                    # Pad tokens: add pad_token_id (usually 0 or 1) to reach max_length
                    tokens = tokens + [1] * (max_length - len(tokens))
                else:
                    # No padding needed, all tokens are real
                    attention_mask = [1] * max_length
                
                encodings.append(tokens)
                attention_masks.append(attention_mask)
        
            return {
                'input_ids': encodings,
                'attention_mask': attention_masks
        }
        else:
            # For Hugging Face tokenizer
            return tokenizer(
                texts, 
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors=None
            )
        
        # Apply tokenization
        tokenzed_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_dataset
    
"""Create GPT-2 model using Hugging Face transformers"""
def create_gpt2_model_hf(vocab_size=None, use_pretrained=False):
    if use_pretrained:
        print("Loading pre-trained GPT-2 model...")
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
        return model, tokenizer

    else:
        print("Creating new GPT-2 model with vocaab size {vocab_size}")

        # Configure model similar to GPT-2 small but with custom vocab
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=1024,  # Sets the maximum context window (how many tokens the model can "see" at once
            n_embd=512,        # Controls the size of the model (hidden dimension)
            n_layer=6,         # Number of transformer blocks (layers), More layers = deeper model
            n_head=8,          # Number of parallel attention heads per layer. Should divide evenly into n_embd. 
            n_inner=2048,      # Size of the feed-forward layer (usually 4 * n_embd)
            activation_function='gelu_new', # Activation function (GELU is standard for GPT-2, provides non-linearity)
            resid_pdrop=0.1, # Dropout rate for residual connections (prevents overfitting)
            embd_pdrop=0.1, # Dropout rate for embedding layer (prevents overfitting)
            attn_pdrop=0.1, # Dropout rate for attention weights (prevents overfitting)
            layer_norm_epsilon=1e-5, # Epsilon for layer normalization (for numerical stability)
            initializer_range=0.02, # Controls the random initialization of weights
            use_cache=True, # Enable caching for faster inference (for faster generation)
            bos_token_id=0, # Beginning-of-sequence token ID
            eos_token_id=0, # End-of-sequence token ID
            pad_token_id=1 # Padding token ID (for batching)
        )

        # Create model with custom vocab
        model = GPT2LMHeadModel(config) #Instantiating a model from a configuration file does not load the weights associated with the model, only the configuration. It will initialize the model weights randomly.

        # Initialize weights properly
        #model.apply(model._init_weights) # manually re-initialize the weights after model creation

        return model, config
    
"""Train model using Hugging Face Trainer"""
def train_with_hf_trainer(model, tokenizer, train_dataset, eval_dataset=None, output_dir="./gpt2-wikitext2"):
    # Data collator for language modeling: Batch Preparation, It handles padding, attention masks, ensures all sequences in a batch are the same length (by padding), which is required for efficient GPU processing. 
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer if hasattr(tokenizer, 'pad_toke') else None,
        mlm=False, # Masked Language Modeling (MLM): false: Used for causal language models like GPT-2 which predict the next token in a sequence (no random masking).
        pad_to_multiple_of=8 # Padding to a multiple of 8 or 16 can improve training speed and efficiency on modern GPUs

    )

    # Training arguments optimized for single GPU
    training_args = TrainingArguments(
        output_dir=output_dir,  # Where to save model/checkpoints
        overwrite_output_dir=True, # Overwrite previous runs in the output dir
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Accumulate gradients to simulate a larger batch (4*4=16)
        warmup_steps=500, # Gradually ramp up learning rate for first 500 steps
        logging_steps=100, # Log metrics every 100 steps
        save_steps=1000, # Save model every 1000 steps
        eval_steps=1000, # Evaluate every 1000 steps
        evaluation_strategy="steps" if eval_dataset else "no", # Only evaluate if eval set is provided
        save_total_limit=2,  # Only keep the 2 most recent checkpoints (saves disk space)
        prediction_loss_only=True, # Only compute and log loss (not predictions)
        remove_unused_columns=False, # Keep all columns (important for custom datasets)
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        gradient_checkpointing=True,     # Save memory
        optim="adamw_torch", # Use AdamW optimizer 
        learning_rate=5e-4,
        weight_decay=0.01, # Regularization to prevent overfitting
        adam_beta1=0.9, # AdamW beta1 (momentum)
        adam_beta2=0.95, # AdamW beta2 (variance smoothing)
        max_grad_norm=1.0, # Gradient clipping (prevents exploding gradients)
        lr_scheduler_type="cosine", # Cosine learning rate schedule (smooth decay)
        report_to=None, #Disables external logging
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
        
    print("Starting training with Hugging Face Trainer...")

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model()

    return trainer

"""Generate text using the trained model"""
def generate_text_sample(model, tokenizer, prompt="The", max_length=100):
    # If the user does not provide a prompt, the function will use "The" as the starting text for the model to generate from.
    model.eval()
    device = next(model.parameters()).device

    # Encode prompt
    if hasattr(tokenizer, 'encode'):
        # custom tokenizer
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)
    else:
        # hugging faxe tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids, #The encoded prompt
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8, #Controls randomness; higher = more random.
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
        )

    # Decode
    if hasattr(tokenizer, 'decode'):
        # Custom tokenizer
        generated_text = tokenizer.decode(outputs[0].tolist())
    else:
        # Hugging Face tokenizer
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

"""Main training function following Hugging Face """
def main():
    # Configuration
    VOCAB_SIZE =32000
    MAX_LENGTH = 512
    USE_PRETRAINED = False # Set to True to fine-tune pre-trained GPT-2
    USE_CUSTOME_TOKENIZER = True # Set to False to use GPT-2's tokenizer

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using deviced: {device}")

    # Load dataset
    dataset = load_wikitext2_dataset()

    # User customer BPE tokenizer
    if USE_CUSTOME_TOKENIZER and not USE_PRETRAINED:
        tokenizer_path = "wikitext2_custom_tokenizer"
        if not os.path.exists(f"{tokenizer_path}.json"):
            customer_tokenzier= train_custom_bpe_tokenizer(
                dataset['train'],
                VOCAB_SIZE,
                tokenizer_path
            )
        else:
            print(f"Loading existing custom tokenizer from {tokenizer_path}.json")
            custom_tokenizer = Tokenizer.from_file(f"{tokenizer_path}.json ")

        # Get actual vocab size
        actual_vocab_size = custom_tokenizer.get_vocab_size()
        print(f"Custom tokenizer vocab size: {actual_vocab_size}")

        # Create model
        model, config = create_gpt2_model_hf(vocab_size=actual_vocab_size, use_pretrained=False)

        # Prepare datasets
        train_dataset = prepare_dataset_for_training(
            dataset['train'],
            custom_tokenizer,
            MAX_LENGTH,
            use_custom_tokenzier=True
        )
        eval_dataset = prepare_dataset_for_training(
            dataset['validation'],
            custom_tokenizer,
            MAX_LENGTH,
            use_custom_tokenizer=True
        )

        tokenizer_for_training = None # Custom tokenizer doesn't work with DataCollator

    else:
        # Use pre-trained GPT-2 or GPT-2 tokenizer
        if USE_PRETRAINED:
            model, hf_tokenizer = create_gpt2_model_hf(use_pretrained=True)
        else:
            # Use GPT-2 tokenizer with new model
            hf_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
            if hf_tokenizer.pad_token is None:
                hf_tokenizer.pad_token = hf_tokenizer.eos_token # set the tokenizer's padding token (pad_token) to the same value as its end-of-sequence token (eso_token).
            
            model, config = create_gpt2_model_hf(
                vocab_size=hf_tokenizer.vocab_size,
                use_pretrained=False
            )
        # Prepare datasets
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
        custom_tokenizer = None
    
    # Move model to device
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    # A trainable parameter is a model parameter (weight or bias) that will be updated during training via backpropagation and the optimizer.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # numel() is a PyTorch tensor method that returns the number of elements in the tensor.
    print(f"Model initialized with {total_params:,} total parameters") #{total_params:,}: string formatting to include thousands separators (commas) for better readability.
    print(f"trainable parameters: {trainable_params:,}")

    # Train model using Hugging Face Trainer
    trainer = train_with_hf_trainer(
        model=model,
        tokenizer=tokenizer_for_training,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    print("Training completed")

    # Generate sample text
    print("\nGenerate sample text")
    sample_prompts = ["In the beginning", "Once upon a time", "This is a story about"]

    for prompt in sample_prompts:
        try:
            if USE_CUSTOME_TOKENIZER and not USE_PRETRAINED:
                generated =generate_text_sample(model, custom_tokenizer, prompt)
            else:
                generated = generate_text_sample(model, hf_tokenizer, prompt)
            print(f"Prompt: '{prompt}' -> Generated: '{generated}'")
        except Exception as e:
            print(f"Error generating text for prompt '{prompt}': {e}")

    # Save tokenizer info
    tokenizer_info = {
        "user_custom_tokenizer": USE_CUSTOME_TOKENIZER,
        "use_pretrained": USE_PRETRAINED,
        "vocab_size": actual_vocab_size if USE_CUSTOME_TOKENIZER and not USE_PRETRAINED else hf_tokenizer.vocab_size,
        "max_length": MAX_LENGTH
    }
    with open("./gpt2-wikitext2/tokenizer_infor.json", "w") as f:
        json.dump(tokenizer_info, f, indet=2)

    print("\nTraining pipeline completed")
    print("Model and tokenizer saved in './gpt2-wikitext2/' directory")
    
if __name__ == "__main__":
    main()





    











    
   




