"""
re-implementing gpt model from https://github.com/karpathy/nanoGPT/blob/master/model.py
in tinygrad
"""

from tinygrad import Tensor, nn
from tinygrad.dtype import dtypes

# layernorm
# use Tensor.layernorm which has no learnable parameters


class CausalSelfAttenton:
    def __init__(self, embd_size: int, n_heads: int, bias: bool = False):
        assert embd_size % n_heads == 0
        self.embd_size = embd_size
        self.n_heads = n_heads
        # key, query, value projections for all heads, but in one batch
        self.c_attn = nn.Linear(self.embd_size, 3 * self.embd_size, bias=bias)
        # output projection
        self.c_proj = nn.Linear(self.embd_size, self.embd_size, bias=bias)

    def __call__(self, x: Tensor) -> Tensor:
        B, T, C = x.size()  # batch, sequence, embedding dim
        q, k, v = self.c_attn(x).split(self.embd_size, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        y = q.scaled_dot_product_attention(k, v, is_causal=True)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        return self.c_proj(y)


class MLP:
    def __init__(self, embd_size: int, hidden_size: int, bias: bool = False):
        self.c_fc = nn.Linear(embd_size, hidden_size, bias=bias)
        self.c_proj = nn.Linear(hidden_size, embd_size, bias=bias)

    def __call__(self, x: Tensor) -> Tensor:
        return self.c_proj(self.c_fc(x).gelu())


class Block:
    def __init__(
        self, embd_size: int, n_heads: int, hidden_size: int, bias: bool = False
    ):
        self.attn = CausalSelfAttenton(embd_size=embd_size, n_heads=n_heads, bias=bias)
        self.mlp = MLP(embd_size=embd_size, hidden_size=hidden_size, bias=bias)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(x.layernorm())
        x = x + self.mlp(x.layernorm())
        return x


class GPT:
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layers: int,
        n_heads: int,
        embed_size: int,
        hidden_size: int,
        bias: bool = False,
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tok_embed = nn.Embedding(vocab_size=vocab_size, embed_size=embed_size)
        self.pos_embed = nn.Embedding(vocab_size=block_size, embed_size=embed_size)
        self.blocks = [
            Block(
                embd_size=embed_size,
                n_heads=n_heads,
                hidden_size=hidden_size,
                bias=bias,
            )
            for _ in range(n_layers)
        ]
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=bias)
        # weight tying: https://paperswithcode.com/method/weight-tying
        self.tok_embed.weight = self.lm_head.weight

    def __call__(self, idx: Tensor, targets: Tensor = None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"cannot forward sequence of length {t}, block size only {self.block_size}"
        pos = Tensor.arange(0, t, dtype=dtypes.long, device=device)

        tok_embed = self.tok_embed(idx)
        pos_embed = self.pos_embed(pos)
        x = tok_embed + pos_embed

        for block in self.blocks:
            x = block(x)
        x = x.layernorm()

        if targets is not None:
            logits = self.lm_head(x)
            loss = logits.view(-1, logits.size(-1)).sparse_categorical_crossentropy(
                targets.view(-1)
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
