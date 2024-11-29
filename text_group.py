import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')


# Load the CSV file
df = pd.read_csv('output_descriptions.csv')

# Function to preprocess text
def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    # Simplify tokenization process
    tokens = word_tokenize(text) 
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(tokens)


# Apply preprocessing to each description
df['processed_text'] = df['text'].apply(preprocess)


# Example of creating a synonym dictionary (extend this according to your data)
synonym_dict = {
    'amphibian': ['frog', 'salamander'],
    'caterpillar': ['larva', 'worm'],
    # Add more based on specific keywords and synonyms
}

# Function to assign groups based on keywords
def assign_group(text):
    for key, synonyms in synonym_dict.items():
        if any(word in text.split() for word in synonyms):
            return key
    return 'other'  # Default group if no keywords match

# Apply group assignment to the processed text
df['group'] = df['processed_text'].apply(assign_group)


df.to_csv('grouped_output.csv', index=False)
