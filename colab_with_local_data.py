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
import time
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

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    print("[INFO] FlashAttention available - will use for faster training")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("[INFO] FlashAttention not available - using standard attention")

# ==================== BONUS: EFFICIENT ATTENTION INSTALLATION ====================
"""
To enable FlashAttention (Bonus requirement), install it with:

pip install flash-attn --no-build-isolation

Or for older systems:
pip install flash-attn==1.0.9 --no-build-isolation

Note: FlashAttention requires:
- CUDA-compatible GPU
- PyTorch with CUDA support
- Sufficient GPU memory

Benefits of FlashAttention:
- 2-4x faster training on long sequences
- Reduced memory usage (O(N) instead of O(N²))
- Mathematically equivalent to standard attention
- Better GPU utilization

Alternative: Sparse Attention Methods
- More complex to implement
- May affect model quality
- Less mature library support
- Better for very long sequences (1K+ tokens)

For sequences of 128 tokens, FlashAttention provides the best trade-off
between performance gains and implementation complexity.
"""

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
    Supports FlashAttention for memory efficiency (Bonus requirement)
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        # Try to use FlashAttention if available (Bonus optimization)
        if FLASH_ATTENTION_AVAILABLE:
            try:
                from flash_attn import FlashMHA
                self.flash_mha = FlashMHA(
                    embed_dim=config.n_embd,
                    num_heads=config.n_head,
                    causal=True,  # Enable causal attention
                    dropout=0.1
                )
                self.use_flash_attention = True
                print(f"[BONUS] Using FlashAttention for layer with {config.n_head} heads")
            except Exception as e:
                print(f"[WARNING] FlashAttention initialization failed: {e}")
                self.use_flash_attention = False
        else:
            self.use_flash_attention = False
        
        # Fallback to standard PyTorch attention
        if not self.use_flash_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.n_embd,
                num_heads=config.n_head,
                dropout=0.1,
                batch_first=True  # Use batch_first for easier handling
            )
            # Create causal mask for standard attention
            self.register_buffer("causal_mask", self._create_causal_mask(config.n_positions))
    
    def _create_causal_mask(self, seq_len):
        """Create causal mask to prevent attending to future positions"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool()
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Use FlashAttention if available (Bonus optimization)
        if self.use_flash_attention:
            # FlashAttention automatically handles causal masking
            attn_output = self.flash_mha(x)
            return attn_output
        else:
            # Fallback to standard attention with manual causal mask
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
    
    def flash_forward(self, x):
        """FlashAttention forward pass (if available)"""
        from flash_attn import FlashMHA
        batch_size, seq_len, _ = x.shape
        
        # Get causal mask for current sequence length
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        # Use FlashAttention for faster training
        attn_output = FlashMHA.apply(
            x, x, x, causal_mask, 0.1, False, False, False
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
    
    # Initialize BPE tokenizer with better configuration
    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    
    # Use ByteLevel pre-tokenizer instead of Whitespace to reduce fragmentation
    from tokenizers.pre_tokenizers import ByteLevel
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    
    # Configure trainer with vocabulary size and better parameters
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        min_frequency=2,  # Require minimum frequency for merges
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet()  # Use byte-level alphabet
    )
    
    # Prepare training data with more text for better learning
    def batch_iterator():
        texts = []
        for example in dataset:
            if example['text'].strip():  # Skip empty texts
                texts.append(example['text'])
                # Yield in batches for memory efficiency
                if len(texts) >= 1000:
                    for text in texts:
                        yield text
                    texts = []
        # Yield remaining texts
        for text in texts:
            yield text
    
    # Train the tokenizer on dataset
    tokenizer.train_from_iterator(batch_iterator(), trainer)
    
    # Add proper post-processor
    from tokenizers.processors import ByteLevel as ByteLevelProcessor
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
    
    # Save the trained tokenizer
    tokenizer.save(f"{save_path}.json")
    print(f"[OK] Custom BPE tokenizer saved to {save_path}.json")
    
    # Test the tokenizer
    test_text = "The future of artificial intelligence"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"[TEST] Original: '{test_text}'")
    print(f"[TEST] Decoded: '{decoded}'")
    print(f"[TEST] Tokens: {len(encoded.ids)} tokens")
    
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
    
    def training_step(self, model, inputs):
        """Override training step to capture loss at every step (like reference)"""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        # Capture training loss at every step like the reference implementation
        self.train_losses.append(loss.item())
        self.train_steps.append(self.state.global_step)
        print(f"[DEBUG] Captured training loss: {loss.item():.4f} at step {self.state.global_step}")
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        loss.backward()
        return loss.detach()
    
    def log(self, logs, start_time=None):
        """Override log method to capture evaluation loss"""
        # Call parent method with proper arguments
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        
        # Capture evaluation loss only (training loss is captured in training_step)
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            self.eval_steps.append(self.state.global_step)
            print(f"[DEBUG] Captured eval loss: {logs['eval_loss']:.4f} at step {self.state.global_step}")
    
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
                print("Potential overfitting detected (validation loss trending upward)")
            else:
                print("No obvious overfitting detected")
        
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
            logging_steps=50,  # More frequent logging to capture training loss
            logging_strategy="steps",
            save_steps=1000,
            eval_steps=500 if eval_dataset else None,  # More frequent evaluation
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=2,
            prediction_loss_only=False,  # Changed to False to get more detailed logs
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
            load_best_model_at_end=True if eval_dataset else False,  # Load best model
            metric_for_best_model="eval_loss" if eval_dataset else None,
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
    Estimate loss on all available dataset splits
    
    Args:
        model: The trained model
        tokenized_datasets: Dict with dataset splits (train/validation/test)
        tokenizer: The tokenizer used
        eval_iters: Number of batches to evaluate
        batch_size: Batch size for evaluation
    """
    model.eval()
    device = next(model.parameters()).device
    
    results = {}
    
    # Handle all available splits, not just train/validation
    available_splits = ['train', 'validation', 'test']
    for split in available_splits:
        if split not in tokenized_datasets:
            continue
            
        dataset = tokenized_datasets[split]
        
        # Check if dataset is empty
        if len(dataset) == 0:
            print(f"Warning: {split} dataset is empty, setting loss to inf")
            results[split] = float('inf')
            continue
            
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

# ==================== MANUAL TRAINING FUNCTIONS (Reference Style) ====================

def get_batch(split, dataset, tokenizer, batch_size=8, block_size=128, device='cpu'):
    """
    Generate a batch of training data like in the reference implementation
    
    Args:
        split: 'train' or 'val'
        dataset: The tokenized dataset
        tokenizer: The tokenizer
        batch_size: Batch size
        block_size: Sequence length
        device: Device to move tensors to
    """
    data = dataset[split] if split in dataset else dataset['train']
    
    # Get random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Create input sequences
    x_batch = []
    y_batch = []
    
    for i in ix:
        # Get sequence of block_size tokens
        if i + block_size + 1 < len(data):
            input_ids = data[i]['input_ids'][:block_size]
            target_ids = data[i]['input_ids'][1:block_size+1]
        else:
            # Handle edge case
            input_ids = data[i]['input_ids'][:block_size]
            target_ids = input_ids[1:] + [0]  # Pad with 0
        
        # Ensure correct length
        if len(input_ids) < block_size:
            input_ids = input_ids + [0] * (block_size - len(input_ids))
        if len(target_ids) < block_size:
            target_ids = target_ids + [0] * (block_size - len(target_ids))
        
        x_batch.append(input_ids[:block_size])
        y_batch.append(target_ids[:block_size])
    
    x = torch.tensor(x_batch, dtype=torch.long).to(device)
    y = torch.tensor(y_batch, dtype=torch.long).to(device)
    
    return x, y

@torch.no_grad()
def estimate_loss_manual(model, dataset, tokenizer, eval_iters=100, batch_size=8, block_size=128, device='cpu'):
    """
    Estimate loss on train and validation sets like in the reference
    
    Args:
        model: The model to evaluate
        dataset: The tokenized dataset
        tokenizer: The tokenizer
        eval_iters: Number of evaluation iterations
        batch_size: Batch size
        block_size: Sequence length
        device: Device
    
    Returns:
        dict: Loss estimates for train and validation
    """
    out = {}
    model.eval()
    
    for split in ['train', 'validation']:
        if split not in dataset:
            continue
            
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            try:
                X, Y = get_batch(split, dataset, tokenizer, batch_size, block_size, device)
                outputs = model(input_ids=X, labels=Y)
                loss = outputs['loss']
                losses[k] = loss.item()
            except Exception as e:
                print(f"Error in batch {k}: {e}")
                losses[k] = float('inf')
        
        out[split] = losses.mean().item()
    
    model.train()
    return out

def train_model_manual(model, dataset, tokenizer, max_iters=1000, eval_interval=100, 
                       learning_rate=1e-3, batch_size=8, block_size=128, device='cpu'):
    """
    Manual training loop inspired by the reference implementation
    
    Args:
        model: The model to train
        dataset: The tokenized dataset
        tokenizer: The tokenizer
        max_iters: Maximum training iterations
        eval_interval: How often to evaluate
        learning_rate: Learning rate
        batch_size: Batch size
        block_size: Sequence length
        device: Device
    
    Returns:
        dict: Training history with losses
    """
    print(f"Starting manual training for {max_iters} iterations...")
    
    # Create optimizer like the reference
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Track losses
    train_losses = []
    eval_losses = []
    steps = []
    eval_steps = []
    
    for iter in range(max_iters):
        # Evaluate loss periodically like the reference
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss_manual(model, dataset, tokenizer, 
                                        eval_iters=50, batch_size=batch_size, 
                                        block_size=block_size, device=device)
            
            train_loss = losses.get('train', float('inf'))
            val_loss = losses.get('validation', float('inf'))
            
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            
            eval_losses.append(val_loss)
            eval_steps.append(iter)
        
        # Sample a batch of data like the reference
        try:
            xb, yb = get_batch('train', dataset, tokenizer, batch_size, block_size, device)
            
            # Evaluate the loss like the reference
            outputs = model(input_ids=xb, labels=yb)
            loss = outputs['loss']
            
            # Track training loss at every step
            train_losses.append(loss.item())
            steps.append(iter)
            
            # Backpropagation like the reference
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
        except Exception as e:
            print(f"Error in training step {iter}: {e}")
            continue
    
    print("Manual training completed!")
    
    return {
        'train_losses': train_losses,
        'train_steps': steps,
        'eval_losses': eval_losses,
        'eval_steps': eval_steps
    }

# ==================== IMPROVED TRAINING FUNCTIONS ====================

def test_tokenizer_quality(tokenizer, test_texts=None):
    """
    Test tokenizer quality and fragmentation
    
    Args:
        tokenizer: The tokenizer to test
        test_texts: List of test texts (optional)
    
    Returns:
        dict: Quality metrics
    """
    if test_texts is None:
        test_texts = [
            "The future of artificial intelligence",
            "Once upon a time in a distant galaxy",
            "Neural networks are powerful machine learning models",
            "Climate change affects global weather patterns",
            "Programming languages enable software development"
        ]
    
    print(f"\n{'='*60}")
    print("TOKENIZER QUALITY TEST")
    print(f"{'='*60}")
    
    results = {
        'texts': [],
        'original_lengths': [],
        'token_counts': [],
        'fragmentation_scores': [],
        'examples': []
    }
    
    for text in test_texts:
        # Handle different tokenizer types
        if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'get_vocab_size'):
            # Custom BPE tokenizer
            encoded = tokenizer.encode(text)
            tokens = encoded.tokens
            decoded = tokenizer.decode(encoded.ids)
            token_count = len(encoded.ids)
        else:
            # Hugging Face tokenizer
            encoded = tokenizer.encode(text)
            tokens = tokenizer.convert_ids_to_tokens(encoded)
            decoded = tokenizer.decode(encoded)
            token_count = len(encoded)
        
        # Calculate fragmentation score (words vs tokens)
        words = text.split()
        fragmentation_score = token_count / len(words) if words else 1.0
        
        # Store results
        results['texts'].append(text)
        results['original_lengths'].append(len(words))
        results['token_counts'].append(token_count)
        results['fragmentation_scores'].append(fragmentation_score)
        
        example = {
            'original': text,
            'decoded': decoded,
            'tokens': tokens[:10],  # First 10 tokens
            'word_count': len(words),
            'token_count': token_count,
            'fragmentation': fragmentation_score
        }
        results['examples'].append(example)
        
        print(f"\nOriginal: '{text}'")
        print(f"Decoded:  '{decoded}'")
        print(f"Tokens:   {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"Words: {len(words)}, Tokens: {token_count}, Fragmentation: {fragmentation_score:.2f}")
    
    # Calculate overall metrics
    avg_fragmentation = sum(results['fragmentation_scores']) / len(results['fragmentation_scores'])
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Average fragmentation: {avg_fragmentation:.2f}")
    print(f"Quality assessment: {'Good' if avg_fragmentation < 2.0 else 'Needs improvement'}")
    
    if avg_fragmentation > 2.5:
        print(" High fragmentation detected! Consider:")
        print("   - Using ByteLevel pre-tokenizer")
        print("   - Adjusting min_frequency parameter")
        print("   - Using more training data")
    
    return results

def debug_loss_curves(loss_file="./loss_curves.json"):
    """Debug what's actually saved in the loss curves file"""
    try:
        import json
        with open(loss_file, 'r') as f:
            loss_data = json.load(f)
        
        print(f"\n{'='*50}")
        print("LOSS CURVES DEBUG")
        print(f"{'='*50}")
        
        for key, value in loss_data.items():
            if isinstance(value, list):
                print(f"{key}: {len(value)} entries")
                if value:
                    print(f"  First: {value[0]}")
                    print(f"  Last: {value[-1]}")
                    print(f"  Sample: {value[:3]}")
                else:
                    print(f"  Empty list!")
            else:
                print(f"{key}: {value}")
        
        return loss_data
        
    except FileNotFoundError:
        print(f"Loss curves file not found: {loss_file}")
        return None
    except Exception as e:
        print(f"Error reading loss curves: {e}")
        return None

# ==================== MAIN FUNCTIONS WITH FIXES ====================

def train_with_better_loss_tracking(model, tokenizer, train_dataset, eval_dataset=None, 
                                   use_manual_training=False, test_mode=True):
    """
    Enhanced training with better loss tracking options
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        use_manual_training: If True, use manual training loop like reference
        test_mode: If True, use test mode settings
    
    Returns:
        dict: Training results with loss curves
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if use_manual_training:
        print("Using manual training loop (reference style) for better loss tracking...")
        
        # Convert dataset to proper format for manual training
        tokenized_data = {
            'train': train_dataset,
            'validation': eval_dataset if eval_dataset else train_dataset
        }
        
        # Training parameters
        max_iters = 100 if test_mode else 1000
        eval_interval = 20 if test_mode else 100
        batch_size = 4 if test_mode else 8
        block_size = 128
        learning_rate = 1e-3 if test_mode else 5e-4
        
        # Manual training
        results = train_model_manual(
            model=model,
            dataset=tokenized_data,
            tokenizer=tokenizer,
            max_iters=max_iters,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            batch_size=batch_size,
            block_size=block_size,
            device=device
        )
        
        # Save results
        import json
        with open('./loss_curves.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        
        if results['train_losses'] and results['train_steps']:
            plt.plot(results['train_steps'], results['train_losses'], 'b-', label='Training Loss', linewidth=2)
        
        if results['eval_losses'] and results['eval_steps']:
            plt.plot(results['eval_steps'], results['eval_losses'], 'r-', label='Validation Loss', linewidth=2, marker='o')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves (Manual Training)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./training_loss_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Loss curves saved to loss_curves.json and training_loss_curve.png")
        
        return results
        
    else:
        print("Using improved CustomTrainer with fixed loss tracking...")
        
        # Use the improved CustomTrainer
        trainer = train_custom_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_mode=test_mode
        )
        
        return trainer

# ==================== UPDATED MAIN PIPELINE ====================