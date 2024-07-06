from embedding import Embedding
import torch


# Sample vocabulary and sentences
vocab = ["i", "am", "learning", "transformers", "with", "mohamed", "stifi"]
vocab_size = len(vocab)
embedding_dim = 4

# Create a word to index mapping
word_to_idx = {word: i for i, word in enumerate(vocab)}

# Initialize the embedding layer
embedding = Embedding(vocab_size, embedding_dim)

# Example sentence
sentence = ["i", "am", "learning", "with", "mohamed", "stifi"]

# Convert words to indices
indices = torch.tensor([word_to_idx[word] for word in sentence])

# Get the embeddings for the sentence
embedded_sentence = embedding(indices)

print("Indices:", indices)
print("Embedded Sentence Shape:", embedded_sentence.shape)
print("Embedded Sentence:", embedded_sentence)

