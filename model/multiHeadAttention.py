class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        assert embedding_dim % num_heads == 0, 'embedding_dim must be divisible by num_heads'
        self.head_dim = embedding_dim // num_heads
        self.W_Q = nn.Linear(embedding_dim, embedding_dim)
        self.W_K = nn.Linear(embedding_dim, embedding_dim)
        self.W_V = nn.Linear(embedding_dim, embedding_dim)
        self.W_O = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(Q, K, V, mask, dropout = None):
        """
            Q, K, V: (batch, num_heads, seq_len, head_dim)
        """
        d_k = Q.shape[-1]

        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len) --> (batch, num_heads, seq_len, seq_len)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = scores.softmax(dim = -1)
        if dropout is not None:
            attention_weights = dropout(attention_weights)

        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim) --> (batch, num_heads, seq_len, head_dim)
        return (attention_weights @ V), attention_weights

    def forward(self, Q, K, V, mask = None):
        query = self.W_Q(Q)      # (batch, seq_len, embedding_dim) --> (batch, seq_len, embedding_dim)
        key = self.W_K(K)      # (batch, seq_len, embedding_dim) --> (batch, seq_len, embedding_dim)
        value = self.W_V(V)      # (batch, seq_len, embedding_dim) --> (batch, seq_len, embedding_dim)

        # (batch, seq_len, embedding_dim) -->  # (batch, num_heads, seq_len, head_dim)
        query = query.view(query.shape[0], -1, self.num_heads, self.head_dim).permute(0,2,1,3)

        # (batch, seq_len, embedding_dim) -->  # (batch, num_heads, seq_len, head_dim)
        key = key.view(key.shape[0], -1, self.num_heads, self.head_dim).permute(0,2,1,3)

        # (batch, seq_len, embedding_dim) -->  # (batch, num_heads, seq_len, head_dim)
        value = value.view(value.shape[0], -1, self.num_heads, self.head_dim).permute(0,2,1,3)

        # x: (batch, num_heads, seq_len, head_dim)
        x, self.attention_weights = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, num_heads, seq_len, head_dim) --> (batch, seq_len, num_heads, head_dim) --> (batch, seq_len, embedding_dim)
        x = x.permute(0,2,1,3).contiguous().view(x.shape[0], -1, self.embedding_dim)

        # (batch, seq_len, embedding_dim) --> (batch, seq_len, embedding_dim)
        return self.W_O(x)