import pandas as pd
import numpy as np 
from collections import Counter
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json

def analyze_descriptions(csv_path, min_frequency=10, min_correlation=0.3):
    df = pd.read_csv(csv_path)
    descriptions = df['text'].tolist()
    
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')
    
    # Tokenize and clean
    stop_words = set(stopwords.words('english'))
    tokens_list = []
    for desc in descriptions:
        tokens = word_tokenize(desc.lower())
        tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
        tokens_list.append(tokens)
    
    # Build vocabulary with frequency threshold
    word_freq = Counter([t for tokens in tokens_list for t in tokens])
    vocab = {word for word, freq in word_freq.items() if freq >= min_frequency}
    
    # Create co-occurrence matrix
    vocab_size = len(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    cooc_matrix = np.zeros((vocab_size, vocab_size))

    for tokens in tokens_list:
        tokens = [t for t in tokens if t in vocab]
        for i, token1 in enumerate(tokens):
            for token2 in tokens[i+1:]:
                idx1, idx2 = word_to_idx[token1], word_to_idx[token2]
                cooc_matrix[idx1, idx2] += 1
                cooc_matrix[idx2, idx1] += 1
    
    # Normalize matrix
    word_counts = np.array([word_freq[word] for word in vocab])
    expected_cooc = np.outer(word_counts, word_counts) / len(tokens_list)
    correlation_matrix = cooc_matrix / expected_cooc
    
    # Find related terms
    patterns = defaultdict(list)
    for i, word in enumerate(vocab):
        related = []
        for j, other_word in enumerate(vocab):
            if i != j and correlation_matrix[i, j] > min_correlation:
                related.append((other_word, correlation_matrix[i, j]))
        if related:
            patterns[word] = sorted(related, key=lambda x: x[1], reverse=True)
    
    # Save results
    with open('description_patterns.json', 'w') as f:
        json.dump({
            'word_frequencies': {w: c for w, c in word_freq.items() if w in vocab},
            'patterns': patterns
        }, f, indent=2)
    
    return patterns, word_freq

# Create pattern groups based on co-occurrence
def create_pattern_groups(patterns, word_freq, min_group_size=3):
    groups = {}
    
    # Group by primary categories
    categories = {
        'attributes': set(['small', 'large', 'long', 'round', 'spiky']),
        'colors': set(['blue', 'red', 'green', 'white', 'black', 'purple']),
        'features': set(['wings', 'tail', 'horns', 'markings', 'spikes'])
    }
    
    for category, seed_words in categories.items():
        groups[category] = {}
        for word in seed_words:
            if word in patterns:
                related = [w for w, _ in patterns[word] if word_freq[w] > min_group_size]
                if related:
                    groups[category][word] = related
    
    # Save grouped patterns
    with open('pattern_groups.json', 'w') as f:
        json.dump(groups, f, indent=2)
    
    return groups

# Usage
patterns, word_freq = analyze_descriptions('output_descriptions.csv')
groups = create_pattern_groups(patterns, word_freq)