import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        
        # 每个维度上的缩放因子div_term
        div_term = 1 / 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        
        position = torch.arange(0, max_len).unsqueeze(1).float()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# --- Scaled Dot-Product Attention ---
def scaled_dot_product_attention(query, key, value, mask=None):
    """计算缩放点积注意力"""

    # 计算点积得分
    # (batch_size, num_heads, query_len, head_dim) dot (batch_size, num_heads, key_len, head_dim)]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))  # d_k 即 head_dim

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    # 计算注意力权重
    attention_weights = F.softmax(scores, dim=-1)

    # 将注意力权重与值相乘，得到输出
    attention_output = torch.matmul(attention_weights, value)
    
    return attention_output

# --- Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 一次投影，分头 
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """将最后一维拆分成多个头"""
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)    #(batch_size, num_heads, seq_len, head_dim)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
         
        # 将查询、键、值投影到不同的子空间
        # (batch_size, seq_len, d_model) => (batch_size, num_heads, seq_len, head_dim)
        query = self.split_heads(self.query_linear(query), batch_size)
        key = self.split_heads(self.key_linear(key), batch_size)
        value = self.split_heads(self.value_linear(value), batch_size)

        # 计算缩放点积注意力
        attention_output = scaled_dot_product_attention(query, key, value, mask)

        # 将多头输出拼接，contiguous()确保张量的存储在内存中是连续的
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        return self.out_linear(attention_output)

# --- FeedForward Neural Network ---
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# --- 编码器层（Encoder Layer） ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        attn_output = self.attention(x, x, x, padding_mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x

# --- 解码器层（Decoder Layer） ---
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, num_heads)
        self.attention2 = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, combined_mask=None, padding_mask=None):
        attn_output1 = self.attention1(x, x, x, combined_mask)
        x = self.layer_norm1(x + self.dropout(attn_output1))
        attn_output2 = self.attention2(x, enc_output, enc_output, padding_mask)
        x = self.layer_norm2(x + self.dropout(attn_output2))
        ff_output = self.feed_forward(x)
        x = self.layer_norm3(x + self.dropout(ff_output))
        return x

# --- Transformer 编码器（Encoder） ---
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# --- Transformer 解码器（Decoder） ---
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, enc_output, combined_mask=None, padding_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, combined_mask, padding_mask)
        return x

# --- Transformer 模型 ---
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout)
        self.out_linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_padding_mask=None, tgt_combined_mask=None):
        enc_output = self.encoder(src, src_padding_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_combined_mask, src_padding_mask)
        output = self.out_linear(dec_output)
        return output