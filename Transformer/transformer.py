import torch
import torch.nn as nn
from encoder_and_decoder import Encoder, Decoder

class Transformer(nn.Module):

    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 src_pad_idx, 
                 tgt_pad_idx, 
                 d_model=512, 
                 num_layers=3, 
                 num_heads=8, 
                 ffn_hidden_dim=2048, 
                 dropout=0.1, 
                 device="cpu", 
                 max_len=1000):
        super().__init__()

        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, ffn_hidden_dim, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, ffn_hidden_dim, dropout, max_len)

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.device = device

    """
        pad_mask: 这个掩码的作用是，由于在同一个batch中各个句子的token数不一致，被padding的部分要进行掩码操作
        causal_mask: 这个掩码的作用是掩盖掉解码时的token之后的tokens
    """
    def create_causal_mask(self, tgt):
        batch_size, len_tgt = tgt.shape
        tgt_mask = torch.tril(torch.ones(len_tgt, len_tgt)).expand(batch_size, 1, len_tgt, len_tgt)
        return tgt_mask.to(self.device)
    
    def create_pad_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def forward(self, src, tgt):
        src_mask = self.create_pad_mask(src)
        tgt_mask = self.create_causal_mask(tgt)

        enc = self.encoder(src, src_mask)
        dec = self.decoder(enc, tgt, src_mask, tgt_mask)
        return dec
