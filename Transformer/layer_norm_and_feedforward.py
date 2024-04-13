import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """
        Args:
            normalized: 输入的特征维度，假设输入是(batch_size, seq_len, d_model)，
            那么normalized_shape就是d_model（一般情况下，最后一个维度表示特征维度）
        """
        super().__init__()
        self.eps = eps
        # 初始化可学习参数γ和β，它们的形状与归一化操作的特征维度相匹配
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        """
        Args:
            x.shape: [batch_size, seq_len, d_model]
        """
        mean = x.mean(dim=-1, keepdim=True) # (batch_size, seq_len, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # (batch_size, seq_len, 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 应用可学习的参数γ和β进行缩放和平移
        return self.gamma * x_norm + self.beta
    

class FeedForward(nn.Module):
    
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        """
        Args:
            x.shape: [batch_size, seq_len, d_model]
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)