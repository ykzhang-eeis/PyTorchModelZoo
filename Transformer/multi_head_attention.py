import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int = 512, num_heads: int = 8):
        """
            实现多头掩码注意力机制模块
            在很多Transformer模型的实现中，embedding_dim和d_model被设定为相同的值，这样做简化了模型设计，这段代码也是如此
            embedding_dim: 每一个token经过embedding层之后的向量长度，把token映射到embedding_dim维的空间
            d_model: W_{q}, W_{k}, W_{v}三个矩阵的维度就是 (embedding_dim, d_model)，即(d_model, d_model)
            因为W_{q}, W_{k}, W_{v}矩阵都要拆分成num_heads个子矩阵，所以d_model要能整除num_heads，维度变为(d_model, d_model/num_heads)
            num_heads: 多头注意力的头数
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_models must be multiple of num_heads"

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)

        self.head_dim = d_model // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        query_len, key_len, value_len = q.shape[1], k.shape[1], v.shape[1]

        num_heads = self.num_heads
        head_dim = self.head_dim

        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, head_dim)
        q = self.query(q).reshape(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
        k = self.key(k).reshape(batch_size, key_len, num_heads, head_dim).transpose(1, 2)
        v = self.value(v).reshape(batch_size, value_len, num_heads, head_dim).transpose(1, 2)

        # (batch_size, num_heads, query_len, head_dim) * (batch_size, num_heads, head_dim, key_len)
        alpha = torch.matmul(q, k.transpose(2, 3)) * self.scale # (batch_size, num_heads, query_len, key_len)
        
        if mask is not None:
            assert mask.shape[-1] == alpha.shape[-1]
            alpha = alpha.masked_fill(mask == 0, float("-inf")) # 会将mask等于0的位置在alpha中被替换为负无穷大
        alpha_bar = F.softmax(alpha, dim=-1)

        # (batch_size, num_heads, query_len, key_len) * (batch_size, num_heads, value_len, head_dim)
        attn = torch.matmul(alpha_bar, v) # (batch_size, num_heads, query_len, head_dim)
        attn = attn.transpose(1, 2).contiguous().reshape(batch_size, query_len, -1) # (batch_size, seq_len, d_model)

        output = self.fc(attn)
        
        return output
    
if __name__ == "__main__":
    """
        test output shape
    """
    x = torch.rand(128, 64, 512) # (batch_size, seq_len, d_model)
    mha = MultiHeadAttention(x.shape[-1])
    output = mha(x, x, x)
    print(output.shape) # (batch_size, seq_len, d_model)