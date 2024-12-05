#  created by ET on 12/4/2024

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from collections import defaultdict

class PokemonDescriptionDataset(Dataset):
    def __init__(self, descriptions, pattern_groups, word_toIndex, window_size=2):
        self.data_pairs = []
        self.word_to_idx = word_toIndex
        
        # Process standard descriptions
        for desc in descriptions:
            words = desc.lower().split()
            self.add_pairs_from_sequence(words, window_size)
        
        # Process pattern groups to create additional semantic pairs
        for category, group_dict in pattern_groups.items():
            for group_name, words in group_dict.items():
                # Convert words to lowercase and filter valid words
                valid_words = [w.lower() for w in words if w.lower() in word_toIndex]
                # Create pairs within each group to capture semantic relationships
                self.add_semantic_pairs(valid_words)
    
    def add_pairs_from_sequence(self, words, window_size):
        for i, target in enumerate(words):
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i != j and target in self.word_to_idx and words[j] in self.word_to_idx:
                    target_idx = self.word_to_idx[target]
                    context_idx = self.word_to_idx[words[j]]
                    self.data_pairs.append((target_idx, context_idx))
    
    def add_semantic_pairs(self, words):
        # Create pairs between all words in the same semantic group
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i != j and word1 in self.word_to_idx and word2 in self.word_to_idx:
                    self.data_pairs.append((
                        self.word_to_idx[word1],
                        self.word_to_idx[word2]
                    ))
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        target_idx, context_idx = self.data_pairs[idx]
        return torch.tensor(target_idx), torch.tensor(context_idx)

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbedding, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, inputs):
        return self.embeddings(inputs)

def load_and_preprocess_data(csv_file, json_file):
    """Load and preprocess Pokemon descriptions and pattern groups"""
    # Load CSV descriptions
    df = pd.read_csv(csv_file)
    descriptions = df['text'].tolist()
    
    # Load JSON pattern groups
    with open(json_file, 'r') as f:
        pattern_groups = json.load(f)
    
    # Create vocabulary from both sources
    words = set()
    # Add words from descriptions
    for desc in descriptions:
        words.update(desc.lower().split())
    
    # Add words from pattern groups
    for category, group_dict in pattern_groups.items():
        for group_name, word_list in group_dict.items():
            words.update(word.lower() for word in word_list)
    
    # Create word mappings
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    return descriptions, pattern_groups, word_to_idx, idx_to_word

def train_embeddings(descriptions, pattern_groups, word_to_idx, embedding_dim=100, batch_size=32, num_epochs=10):
    """Train word embeddings using both descriptions and pattern groups"""
    # Create dataset and dataloader
    dataset = PokemonDescriptionDataset(descriptions, pattern_groups, word_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WordEmbedding(len(word_to_idx), embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for target, context in dataloader:
            target = target.to(device)
            context = context.to(device)
            
            # Forward pass
            embeddings = model(target)
            output = torch.matmul(embeddings, model.embeddings.weight.T)
            
            # Calculate loss and backpropagate
            loss = criterion(output, context)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return model

def encode_description(description, model, word_to_idx):
    """Encode a text description using the trained embeddings"""
    words = description.lower().split()
    word_vectors = []
    
    for word in words:
        if word in word_to_idx:
            idx = torch.tensor([word_to_idx[word]])
            with torch.no_grad():
                word_vector = model(idx).squeeze().cpu().numpy()
                word_vectors.append(word_vector)
    
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.embeddings.weight.shape[1])

def main():
    # Load and preprocess the data
    descriptions, pattern_groups, word_to_idx, idx_to_word = load_and_preprocess_data(
        'output_descriptions.csv',
        'pattern_groups.json'
    )
    
    # Train the embeddings
    model = train_embeddings(descriptions, pattern_groups, word_to_idx)
    
    # Example: Encode a new description
    test_description = "A blue dragon with red wings"
    encoded_vector = encode_description(test_description, model, word_to_idx)
    
    # Print some example information
    print(f"\nVocabulary size: {len(word_to_idx)}")
    print(f"Example encoding for '{test_description}':")
    print(f"Vector shape: {encoded_vector.shape}")
    print(f"Vector preview: {encoded_vector[:5]}...")
    
    # Save the model and mappings
    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word
    }, 'pokemon_embeddings.pt')

if __name__ == "__main__":
    main()