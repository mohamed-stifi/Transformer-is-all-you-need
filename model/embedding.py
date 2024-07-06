'''
## Word Embeddings in Transformers

In transformers, the first step is to convert the input text into a form that the model can process. This is where word embeddings come into play. Instead of using one-hot encoding, transformers use learned word embeddings to represent words in a dense, continuous vector space. These embeddings capture semantic relationships between words, making them more efficient and effective for tasks like translation and paraphrasing.

### How Word Embeddings Work

**Learned Word Embeddings**:
   - Words are represented as dense vectors in a continuous space.
   - These vectors are learned during training.
   - They capture semantic relationships (e.g., the vectors for "queen" and "woman" are closer to each other than to "king" or "man").

### Continuous Bag of Words (CBOW)

Like images, words also have some features, for example — domain, gender, plurality or verb tense. If you can reprsent the words in terms of these features as a quantity i.e a features a quantitative matrix, then using this matrix we can compare different words with each other and based on these features we can calculate the similarity between them. This feature matrix is nothing but the weights in a neural network

### Embedding Shapes in Transformers

- **Input Embedding Matrix**: If your vocabulary size is `V` and your embedding dimension is `d`, the embedding matrix has a shape of `[V, d]`.
- **Word Embeddings**: Each word is mapped to a vector of shape `[d]`.
- **Sequence Embeddings**: For a sequence of length `L`, the embeddings have a shape of `[L, d]`.

### Example Using PyTorch

Let's implement a simple word embedding example using PyTorch.

```python
import torch
import torch.nn as nn

# Sample vocabulary and sentences
vocab = ["i", "am", "learning", "transformers"]
vocab_size = len(vocab)
embedding_dim = 8

# Create a word to index mapping
word_to_idx = {word: i for i, word in enumerate(vocab)}

# Initialize the embedding layer
embedding = nn.Embedding(vocab_size, embedding_dim)

# Example sentence
sentence = ["i", "am", "learning"]

# Convert words to indices
indices = torch.tensor([word_to_idx[word] for word in sentence])

# Get the embeddings for the sentence
embedded_sentence = embedding(indices)

print("Indices:", indices)
print("Embedded Sentence Shape:", embedded_sentence.shape)
print("Embedded Sentence:", embedded_sentence)
```

### Output

```
Indices: tensor([0, 1, 2, 4, 5, 6])
Embedded Sentence Shape: torch.Size([6, 4])
Embedded Sentence: tensor([[-1.6601,  1.3929,  2.1383, -0.5488],
        [-2.7878,  0.6419, -1.4701,  4.6438],
        [ 0.8859, -1.6612,  0.3924, -2.3862],
        [ 2.2542,  1.2763,  1.5111,  2.5506],
        [ 3.2673, -0.5550, -0.2295,  4.3048],
        [-3.8697,  2.7400, -0.8800,  0.6942]], grad_fn=<MulBackward0>)
```

### Explanation of the Code

1. **Vocabulary and Embedding Layer**:
   - We define a small vocabulary and an embedding dimension.
   - `nn.Embedding` initializes the embedding matrix.

2. **Mapping Words to Indices**:
   - We create a mapping from words to indices.
   - Convert the words in a sentence to their corresponding indices.

3. **Getting the Embeddings**:
   - The `embedding` layer converts the indices to dense vectors.
   - The resulting `embedded_sentence` tensor contains the word embeddings for the sentence.

### Shapes

- **Indices Shape**: `[6]` (since the sentence has 6 words)
- **Embedded Sentence Shape**: `[6, 4]` (6 words, each represented by an 4-dimensional vector)

By using learned embeddings, we capture the semantic relationships between words, which helps improve the performance of transformer models on various NLP tasks.
'''
from torch import nn
import math

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):      # embedding_dim = d_model in the orignale paper
        """
        Args:
            d_model (int): Embedding Dimension
            vocabulary_size (int): Vocabulary Size 
        """
        '''
        * The weights of the embedding layer are represented by a matrix that maps each word in the vocabulary to a vector in the embedding space.
        - vocab_size: is the number of words in the vocabulary
        - embedding_dim: is the dimension of the word embeddings
        * The shape of the weights matrix is (vocab_size,embedding_dim).
        '''
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        '''
        * The input to the embedding layer is typically a batch of sequences, where each sequence is a list of token indices.
        * For a batch of sequences (let's denote the batch size as 𝐵), the input shape would be (B,max_input_len).
        * The output of the embedding layer is the batch of sequences with each token index replaced by its corresponding embedding vector.
        * For a batch of sequences (with batch size B), the output shape would be (B,max_input_len,embedding_dim).
        '''
        return self.embedding(x)*math.sqrt(self.embedding_dim)