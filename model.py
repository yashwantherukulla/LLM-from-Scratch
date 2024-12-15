import torch
from torch import nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_len": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate_emb": 0.1,
    "drop_rate_sc": 0.1,
    "drop_rate_mha": 0.1,
    "qkv_bias": False
}

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor: 
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # reasoning for unbiased=False in notes
        norm_x = (x - mean)/torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5*x*(1 + torch.tanh(
            torch.sqrt(torch.tensor(2/torch.pi)) * (x + 0.044715 * torch.pow(x,3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # reduces the projection dim to match the desired output dim
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # Uses a Linear layer to combine all the outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(b, num_tokens, self.num_heads, self.head_dim) # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        K = K.view(b, num_tokens, self.num_heads, self.head_dim)
        V = V.view(b, num_tokens, self.num_heads, self.head_dim)

        Q = Q.transpose(1,2) # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        attn_scores = Q @ K.transpose(2,3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / K.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vect = (attn_weights @ V).transpose(1,2) # (b, num_tokens, num_heads, head_dim)
        context_vect = context_vect.contiguous().view(b, num_tokens, self.d_out) # combines heads
        context_vect = self.out_proj(context_vect)
        return context_vect

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        self.mha = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_len"], cfg["drop_rate_mha"], cfg["n_heads"], cfg["qkv_bias"])
        self.dropout_sc = nn.Dropout(cfg["drop_rate_sc"])
        self.ff = FeedForward(cfg)

    def forward(self, x):
        sc = x
        x = self.norm1(x)
        x = self.mha(x)
        x = self.dropout_sc(x)
        x = x + sc

        sc = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout_sc(x)
        x = x + sc

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_len"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate_emb"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]) # splatting the blocks

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits