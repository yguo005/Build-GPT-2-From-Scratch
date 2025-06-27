# -*- coding: utf-8 -*-
"""
GPT-2 Implementation from Scratch
Based on reference code with additional components to meet all requirements
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import os
import json
import pickle
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# =============================================================================
# 1. HYPERPARAMETERS (GPT-2 Standard Configurations)
# =============================================================================

# GPT-2 Small configuration
CONFIG = {
    'vocab_size': 50257,  # Will be set after tokenizer training
    'block_size': 128,    # Sequence length as required
    'n_embd': 768,        # Embedding dimension
    'n_head': 12,         # Number of attention heads
    'n_layer': 12,        # Number of transformer blocks
    'dropout': 0.1,
    'bias': True
}

# Training hyperparameters
TRAIN_CONFIG = {
    'batch_size': 8,      # Adjust based on GPU memory
    'gradient_accumulation_steps': 4,  # Simulate larger batch size
    'max_iters': 5000,
    'eval_interval': 500,
    'eval_iters': 200,
    'learning_rate': 6e-4,
    'weight_decay': 1e-1,
    'warmup_iters': 500,
    'lr_decay_iters': 5000,
    'min_lr': 6e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'compile': True,
    'checkpoint_interval': 1000
}

print(f"Using device: {TRAIN_CONFIG['device']}")

# =============================================================================
# 2. BPE TOKENIZER IMPLEMENTATION
# =============================================================================

class BPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        
    def train_tokenizer(self, texts):
        """Train BPE tokenizer on provided texts"""
        print("Training BPE tokenizer...")
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Setup trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]
        )
        
        # Train on texts
        self.tokenizer.train_from_iterator(texts, trainer)
        
        # Add post-processor
        self.tokenizer.post_processor = TemplateProcessing(
            single="<|bos|> $A <|eos|>",
            special_tokens=[("<|bos|>", 2), ("<|eos|>", 3)]
        )
        
        print(f"Tokenizer trained with vocab size: {self.tokenizer.get_vocab_size()}")
        
    def encode(self, text):
        """Encode text to token ids"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet")
        return self.tokenizer.encode(text).ids
    
    def decode(self, token_ids):
        """Decode token ids to text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet")
        return self.tokenizer.decode(token_ids)
    
    def save(self, path):
        """Save tokenizer"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet")
        self.tokenizer.save(path)
        
    def load(self, path):
        """Load tokenizer"""
        self.tokenizer = Tokenizer.from_file(path)

# =============================================================================
# 3. DATASET LOADING AND PREPROCESSING (LOCAL FILES)
# =============================================================================

def load_and_prepare_dataset():
    """Load WikiText-2 dataset from local files and prepare for training"""
    print("Loading WikiText-2 dataset from local files...")

    train_path = "wiki.train.txt"
    val_path = "wiki.valid.txt"
    test_path = "wiki.test.txt"

    def read_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    train_texts = read_file(train_path)
    val_texts = read_file(val_path)
    test_texts = read_file(test_path)

    print(f"Train texts: {len(train_texts)}")
    print(f"Validation texts: {len(val_texts)}")
    print(f"Test texts: {len(test_texts)}")

    return train_texts, val_texts, test_texts

def prepare_data(texts, tokenizer, block_size):
    """Tokenize and chunk texts into sequences"""
    all_tokens = []
    
    print("Tokenizing texts...")
    for text in tqdm(texts):
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
    
    # Chunk into sequences
    sequences = []
    for i in range(0, len(all_tokens) - block_size, block_size):
        sequences.append(all_tokens[i:i + block_size])
    
    return torch.tensor(sequences, dtype=torch.long)

# =============================================================================
# 4. TRANSFORMER COMPONENTS (From reference with modifications)
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention using PyTorch's implementation as required"""
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        
        # Use PyTorch's MultiheadAttention as specified in requirements
        self.attention = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            bias=CONFIG['bias'],
            batch_first=True
        )
        
        # Causal mask for autoregressive generation
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(block_size, block_size)).view(1, block_size, block_size)
        )
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Create causal attention mask
        attn_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        
        # Apply attention
        attn_output, _ = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=False
        )
        
        return attn_output

class FeedForward(nn.Module):
    """Feedforward network with GELU activation as required"""
    
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=CONFIG['bias']),
            nn.GELU(),  # GELU instead of ReLU as required
            nn.Linear(4 * n_embd, n_embd, bias=CONFIG['bias']),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer block with pre-layer normalization"""
    
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, bias=CONFIG['bias'])
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd, bias=CONFIG['bias'])
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        # Pre-layer norm as in GPT-2
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT2Model(nn.Module):
    """GPT-2 model implementation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.position_embedding = nn.Embedding(config['block_size'], config['n_embd'])
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config['n_embd'], 
                config['n_head'], 
                config['block_size'], 
                config['dropout']
            ) for _ in range(config['n_layer'])
        ])
        
        # Final layer norm and projection
        self.ln_f = nn.LayerNorm(config['n_embd'], bias=config['bias'])
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        
        # Weight tying (share weights between embedding and output projection)
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.shape
        assert T <= self.config['block_size'], f"Cannot forward sequence of length {T}, block size is only {self.config['block_size']}"
        
        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(pos)  # (T, n_embd)
        x = tok_emb + pos_emb
        
        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text with different sampling strategies:
        - Greedy decoding (temperature=0)
        - Top-k sampling
        - Nucleus (top-p) sampling
        """
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            if temperature == 0:
                # Greedy decoding
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# =============================================================================
# 5. TRAINING UTILITIES
# =============================================================================

def get_batch(data, batch_size, block_size, device):
    """Generate a batch of data"""
    ix = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i] for i in ix])
    y = torch.stack([data[i] for i in ix])  # Targets are the same as inputs (shifted in loss calculation)
    
    # Create input and target sequences
    x = x[:, :-1].contiguous()  # Input: all tokens except last
    y = y[:, 1:].contiguous()   # Target: all tokens except first (shifted by 1)
    
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            with torch.cuda.amp.autocast():
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def calculate_perplexity(model, data, batch_size, block_size, device):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(data) - batch_size, batch_size):
            X, Y = get_batch(data, batch_size, block_size, device)
            with torch.cuda.amp.autocast():
                logits, loss = model(X, Y)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    return perplexity

def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Learning rate schedule: warmup + cosine decay"""
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# =============================================================================
# 6. MAIN TRAINING FUNCTION
# =============================================================================

def train_model():
    """Main training function"""
    
    # Load and prepare dataset
    train_texts, val_texts, test_texts = load_and_prepare_dataset()
    
    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=32000)
    tokenizer.train_tokenizer(train_texts[:1000])  # Use subset for faster training
    
    # Update config with actual vocab size
    CONFIG['vocab_size'] = tokenizer.tokenizer.get_vocab_size()
    
    # Prepare data
    print("Preparing training data...")
    train_data = prepare_data(train_texts, tokenizer, CONFIG['block_size'])
    val_data = prepare_data(val_texts, tokenizer, CONFIG['block_size'])
    test_data = prepare_data(test_texts, tokenizer, CONFIG['block_size'])
    
    print(f"Train sequences: {len(train_data)}")
    print(f"Val sequences: {len(val_data)}")
    print(f"Test sequences: {len(test_data)}")
    
    # Initialize model
    model = GPT2Model(CONFIG).to(TRAIN_CONFIG['device'])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("Starting training...")
    
    for iter_num in range(TRAIN_CONFIG['max_iters']):
        # Update learning rate
        lr = get_lr(
            iter_num, 
            TRAIN_CONFIG['warmup_iters'], 
            TRAIN_CONFIG['lr_decay_iters'], 
            TRAIN_CONFIG['learning_rate'], 
            TRAIN_CONFIG['min_lr']
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate the loss on train/val sets
        if iter_num % TRAIN_CONFIG['eval_interval'] == 0 or iter_num == TRAIN_CONFIG['max_iters'] - 1:
            losses = estimate_loss(
                model, train_data, val_data, 
                TRAIN_CONFIG['eval_iters'], 
                TRAIN_CONFIG['batch_size'], 
                CONFIG['block_size'], 
                TRAIN_CONFIG['device']
            )
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")
            
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': CONFIG,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }, 'best_model.pt')
        
        # Save checkpoint
        if iter_num % TRAIN_CONFIG['checkpoint_interval'] == 0 and iter_num > 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': CONFIG,
                'iter_num': iter_num,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, f'checkpoint_{iter_num}.pt')
        
        # Training step with gradient accumulation
        optimizer.zero_grad()
        for micro_step in range(TRAIN_CONFIG['gradient_accumulation_steps']):
            X, Y = get_batch(train_data, TRAIN_CONFIG['batch_size'], CONFIG['block_size'], TRAIN_CONFIG['device'])
            
            with torch.cuda.amp.autocast():
                logits, loss = model(X, Y)
                loss = loss / TRAIN_CONFIG['gradient_accumulation_steps']  # Scale loss
            
            scaler.scale(loss).backward()
        
        # Clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
    
    # Final evaluation
    print("\nFinal Evaluation:")
    
    # Calculate perplexity on test set
    test_perplexity = calculate_perplexity(
        model, test_data, TRAIN_CONFIG['batch_size'], 
        CONFIG['block_size'], TRAIN_CONFIG['device']
    )
    print(f"Test Perplexity: {test_perplexity:.2f}")
    
    # Generate text samples with different strategies
    print("\nGenerating text samples...")
    model.eval()
    
    prompt = "The quick brown"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=TRAIN_CONFIG['device'])
    
    # Greedy decoding
    with torch.no_grad():
        greedy_output = model.generate(prompt_ids, max_new_tokens=50, temperature=0)
        greedy_text = tokenizer.decode(greedy_output[0].tolist())
        print(f"\nGreedy: {greedy_text}")
    
    # Top-k sampling
    with torch.no_grad():
        topk_output = model.generate(prompt_ids, max_new_tokens=50, temperature=0.8, top_k=50)
        topk_text = tokenizer.decode(topk_output[0].tolist())
        print(f"\nTop-k (k=50): {topk_text}")
    
    # Nucleus sampling
    with torch.no_grad():
        nucleus_output = model.generate(prompt_ids, max_new_tokens=50, temperature=0.8, top_p=0.9)
        nucleus_text = tokenizer.decode(nucleus_output[0].tolist())
        print(f"\nNucleus (p=0.9): {nucleus_text}")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    steps = list(range(0, len(train_losses) * TRAIN_CONFIG['eval_interval'], TRAIN_CONFIG['eval_interval']))
    plt.plot(steps, train_losses, label='Training Loss')
    plt.plot(steps, val_losses, label='Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.show()
    
    # Save final results
    results = {
        'test_perplexity': test_perplexity,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': CONFIG,
        'train_config': TRAIN_CONFIG,
        'generated_samples': {
            'greedy': greedy_text,
            'top_k': topk_text,
            'nucleus': nucleus_text
        }
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save tokenizer
    tokenizer.save('tokenizer.json')
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test perplexity: {test_perplexity:.2f}")

if __name__ == "__main__":
    train_model()
