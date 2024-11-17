import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os
from tqdm import tqdm

# Download and load the Shakespeare text
def download_shakespeare_text():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "shakespeare.txt"
    
    if not os.path.exists(filename):
        print("Downloading Shakespeare text...")
        response = requests.get(url)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Download complete!")
    
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

# Hyperparameters
class Config:
    batch_size = 64
    block_size = 256  # Context length
    max_iters = 500
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 200
    embedding_dim = 384
    hidden_dim = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_data_loader(text, config):
    # Create character level tokens
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Create train/val split
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([data[i:i+config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
        x, y = x.to(config.device), y.to(config.device)
        return x, y
    
    return get_batch, vocab_size, encode, decode

def train_model(model, config, get_batch):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    for iter in tqdm(range(config.max_iters), desc="Training"):
        # Every once in a while evaluate the loss on train and val sets
        if iter % config.eval_interval == 0:
            losses = estimate_loss(model, config, get_batch)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Sample a batch of data
        xb, yb = get_batch('train')
        
        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    return model

@torch.no_grad()
def estimate_loss(model, config, get_batch):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def generate_samples(model, encode, decode, initial_text="ROMEO:", num_samples=3):
    print(f"\nGenerating samples with different strategies (initial text: '{initial_text}')\n")
    context = torch.tensor([encode(initial_text)], dtype=torch.long, device=model.token_embedding_table.weight.device)
    
    # 1. Beam Search
    print("1. Beam Search Generation:")
    beam_results = model.beam_search(
        context,
        beam_width=5,
        max_length=200,
        length_penalty=0.8,
        repetition_penalty=1.2
    )
    for i, (text, score) in enumerate(beam_results[:num_samples], 1):
        print(f"\nBeam {i} (score: {score:.2f}):")
        print(text)
    
    # 2. Nucleus Sampling
    print("\n2. Nucleus Sampling Generation:")
    for i in range(num_samples):
        nucleus_text = model.nucleus_sampling(
            context,
            max_length=200,
            temperature=0.8,
            top_p=0.9
        )
        print(f"\nSample {i+1}:")
        print(nucleus_text)
    
    # 3. Contrastive Search
    print("\n3. Contrastive Search Generation:")
    contrastive_text = model.contrastive_search(
        context,
        max_length=200,
        k=4,
        alpha=0.6
    )
    print(contrastive_text)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(1337)
    
    # Initialize configuration
    config = Config()
    
    # Download and load text
    text = download_shakespeare_text()
    print(f"Total length of dataset in characters: {len(text)}")
    
    # Create data loader and get vocabulary size
    get_batch, vocab_size, encode, decode = create_data_loader(text, config)
    
    # Initialize model
    model = AdvancedBigramLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim
    ).to(config.device)
    
    # Print model size
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Train model
    print("Training model...")
    model = train_model(model, config, get_batch)
    print("Training complete!")
    
    # Generate samples
    generate_samples(model, encode, decode)
    
    # Save model
    torch.save(model.state_dict(), 'shakespeare_model.pt')
    print("\nModel saved as 'shakespeare_model.pt'")

if __name__ == '__main__':
    main()