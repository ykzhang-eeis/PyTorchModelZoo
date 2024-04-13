import torch.nn as nn
from layer_norm_and_feedforward import LayerNorm, FeedForward
from multi_head_attention import MultiHeadAttention
from token_and_position_embedding import TransformerEmbedding

class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, ffn_hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_hidden_dim, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Args:
            src.shape: [batch_size, seq_len, embedding_dim]
        """
        src1 = self.norm1(src)
        src = src + self.dropout1(self.mha(src1, src1, src1, src_mask))
        src2 = self.norm2(src)
        src = src + self.dropout2(self.ffn(src2))

        return src

class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, ffn_hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.mask_mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(d_model)
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_hidden_dim, dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        
        tgt1 = self.norm1(tgt)
        tgt = tgt + self.dropout1(self.mask_mha(tgt1, tgt1, tgt1, tgt_mask))

        tgt2 = self.norm2(tgt)
        tgt = tgt + self.dropout2(self.cross_mha(tgt2, src, src, src_mask))

        tgt3 = self.norm3(tgt)
        tgt = tgt + self.dropout3(self.ffn(tgt3))

        return tgt
    
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_layers, num_heads, ffn_hidden_dim, dropout, max_len):
        super().__init__()
        self.embedding = TransformerEmbedding(src_vocab_size, d_model, dropout, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, ffn_hidden_dim, dropout) for _ in range(num_layers)
            ])
        self.norm = LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        src = self.norm(src)
        return src
    
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_layers, num_heads, ffn_hidden_dim, dropout, max_len):
        super().__init__()
        self.embedding = TransformerEmbedding(tgt_vocab_size, d_model, dropout, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ffn_hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        tgt = self.embedding(tgt)
        for layer in self.layers:
            tgt = layer(src, tgt, src_mask, tgt_mask)
        tgt = self.norm(tgt)
        tgt = self.fc(tgt)
        return tgt