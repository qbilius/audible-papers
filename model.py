"""
Adapted from https://github.com/karpathy/nanoGPT
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, weight=self.weight, bias=self.bias, eps=1e-5)


class SelfAttention(nn.Module):

    def __init__(self,
                 n_head: int,
                 n_embed: int,
                 dropout: float,
                 bias: bool,
                 ) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=bias)
        self.multihead_attn = nn.MultiheadAttention(n_embed, n_head, dropout=dropout, bias=bias, batch_first=True)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        y = self.multihead_attn(q, k, v, need_weights=False)[0]
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):

    def __init__(self,
                 n_embed: int,
                 dropout: float,
                 bias: bool,
                 ):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed, bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embed, n_embed, bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 n_head: int,
                 n_embed: int,
                 dropout: float,
                 bias: bool,
                 ):
        super().__init__()
        self.ln_1 = LayerNorm(n_embed, bias)
        self.attn = SelfAttention(n_head, n_embed, dropout, bias)
        self.ln_2 = LayerNorm(n_embed, bias)
        self.mlp = MLP(n_embed, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self,
                 block_size=1024,
                 vocab_size=50304,  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
                 n_layer=12,
                 n_head=12,
                 n_embed=768,
                 dropout=0,
                 bias=True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
                 ):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embed),
            wpe=nn.Embedding(block_size, n_embed),
            dropout=nn.Dropout(dropout),
            hidden=nn.ModuleList([Block(n_head, n_embed, dropout, bias) for _ in range(n_layer)]),
            ln_f=LayerNorm(n_embed, bias=bias),
        ))
        self.lm_head = nn.Linear(n_embed, 1, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0, std=.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=.02)

    def forward(self, x) -> torch.Tensor:
        t: int = x.shape[1]
        if t > self.block_size:  # noqa
            raise ValueError(f'Cannot forward sequence of length {t}, '
                             f'block size is only {self.block_size}')
        pos: torch.Tensor = torch.arange(0, t, dtype=torch.long, device=x.device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(x)  # token embeddings of shape (b, t, n_embed)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embed)
        x = self.transformer.dropout(tok_emb + pos_emb)
        for block in self.transformer.hidden:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits

    # @torch.no_grad()
    # def generate(self,
    #              idx: torch.LongTensor,
    #              max_new_tokens: int,
    #              temperature: float = 1,
    #              top_k: int | None = None
    #              ):
    #     """
    #     Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    #     the sequence max_new_tokens times, feeding the predictions back into the model each time.
    #     Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    #     """
    #     for _ in range(max_new_tokens):
    #         # if the sequence context is growing too long we must crop it at block_size
    #         idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
    #         # forward the model to get the logits for the index in the sequence
    #         logits, _ = self(idx_cond)
    #         # pluck the logits at the final step and scale by desired temperature
    #         logits = logits[:, -1, :] / temperature
    #         # optionally crop the logits to only the top k options
    #         if top_k is not None:
    #             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #             logits[logits < v[:, [-1]]] = -float('Inf')
    #         # apply softmax to convert logits to (normalized) probabilities
    #         probs = F.softmax(logits, dim=-1)
    #         # sample from the distribution
    #         idx_next = torch.multinomial(probs, num_samples=1)
    #         # append sampled index to the running sequence and continue
    #         idx = torch.cat((idx, idx_next), dim=1)

    #     return idx
