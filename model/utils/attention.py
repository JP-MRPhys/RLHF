
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(MultiheadSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        # define linear transformations for Q, K, and V
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        # define linear transformation for output
        self.output_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]

        # linear transformations of Q, K, and V
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        # scaled dot product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size).float())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        attn_output = torch.matmul(attn, v)

        # concatenate and linear transformation of output
        concat = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.output_linear(concat)

        return output


class TransformerHead(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, ff_hidden_size):
        super(TransformerHead, self).__init__()
        self.self_attention = MultiheadSelfAttention(hidden_size, num_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, ff_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_size, hidden_size),
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.layer_norm1(x + self.dropout(self.self_attention(x, mask)))
        x = self.layer_norm2(x + self.dropout(self.feedforward(x)))
        return x + residual


import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_size = x.size()
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        out = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return out


class TransformerHead(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, ff_hidden_size):
        super(TransformerHead, self).__init__()
        self.self_attention = SelfAttention(hidden_size, num_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, ff_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_size, hidden_size),
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.layer_norm1(x + self.dropout(self.self_attention(x, mask)))
        x = self.layer_norm2(x + self.dropout(self.feedforward(x)))
        return x + residual
