import math
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.d_model = d_model
    
    def forward(self, x):
        """
        Args:
            x.shape: [batch_size, seq_len]
            embedding_dim 和 d_model 被设定为相同的值，简化了模型设计
            TokenEmbedding 类的 forward 方法接受一个形状为 [batch_size, seq_len] 的张量
            seq_len 这个参数和 token 数是相关的，seq_len 反映了处理后序列的统一长度。
            如果我们想要在一个批次中处理 token 长度不同的序列，我们可能需要将它们填充
            或截取到相同的长度，假设我们有以下三个序列，它们的 token 数量不同
            序列A: [the, cat, sat]
            序列B: [on, the, mat]
            序列C: [and, looked, at, me]
            如果我们想要在一个批次中处理这些序列，我们需要将它们填充到相同的长度，例如：
            序列A: [the, cat, sat, <pad>]
            序列B: [on, the, mat, <pad>]
            序列C: [and, looked, at, me]
        """
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model) # [batch_size, seq_len, d_model]
    

class PositionEmbedding(nn.Module):
    
    def __init__(self, seq_len, d_model):
        super().__init__()
        # pos_embedding参数是固定的，不会随着训练进行更新
        self.pos_embedding = nn.Parameter(torch.zeros(seq_len, d_model), requires_grad=False)

        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))

        self.pos_embedding[:, 0::2] = torch.sin(position * div_term)
        self.pos_embedding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        """
        Args:
            x.shape: [batch_size, seq_len, d_model]
        """
        return x + self.pos_embedding[:x.size(1), :].unsqueeze(0)


class TransformerEmbedding(nn.Module):
    
    def __init__(self, vocab_size, d_model, dropout, max_len):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionEmbedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x.shape: [batch_size, seq_len]
        """
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(tok_emb)
        return self.dropout(pos_emb)

    