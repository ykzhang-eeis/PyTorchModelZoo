import torch
from transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
src_vocab_size = 1000  # 词汇表大小
tgt_vocab_size = 2000
src_seq_len = 8  # 源序列长度
tgt_seq_len = 16  # 目标序列长度
src_pad_idx = 0  # 源序列填充索引
tgt_pad_idx = 0  # 目标序列填充索引

src = torch.randint(low=1, high=src_vocab_size, size=(batch_size, src_seq_len)).to(device)
tgt = torch.randint(low=1, high=tgt_vocab_size, size=(batch_size, tgt_seq_len)).to(device)

# 添加填充以模拟实际情况（可选）
src[0, -2:] = src_pad_idx  # 假设源序列最后两个是填充
tgt[0, -3:] = tgt_pad_idx

model = Transformer(src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, device=device).to(device)
output = model(src, tgt)

print(output.shape)
