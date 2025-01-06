import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    """
    This converts words (represented as numbers) into deterministic vectors.

    Args:
        vocab_size: how many different words your model knows
        d_model: how many dimensions each word vector will have

    The vectors are multiplied by sqrt(d_model) to help with training stability.

    Example:
        If you have vocab_size=1000 and d_model=512, each word from your 
        1000-word vocabulary gets converted into a 512-dimensional vector
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    
    """
    Adds positional information to word vectors in transformer models.
    
    Since transformers process all words simultaneously, they need position info
    to know word order. This is done by:
    1. Creating unique sine/cosine patterns for each position
    2. Adding these patterns to the word vectors
    
    Intuition:
        Imagine the positional encoding as giving each position in the sequence a "signature" or "address" in a high-dimensional space, 
        where these signatures reflect both the absolute position and relative differences between positions. The sinusoids act as a unique 
        but systematic way to assign these signatures, enabling the model to "understand" sequence order without explicit recurrence or convolution mechanisms.

    Args:
        d_model: Dimension of the model
        seq_len: Maximum sequence length the model can handle
        dropout: Dropout rate to prevent overfitting
        
    Flow:
    1. Text -> Numbers (vocabulary lookup)
    2. Numbers -> Vectors (InputEmbeddings)
    3. Vectors + Position Info (PositionalEncoding)
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10e**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied  
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim= True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    Standard feed-forward network in transformer architecture.
    Consists of two linear transformations with a ReLU activation in between.
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension of the feed-forward layer (typically 4x d_model)
        dropout: Dropout rate
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


