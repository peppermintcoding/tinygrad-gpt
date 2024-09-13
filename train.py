from model import GPT
from tinygrad import Tensor

model = GPT(
    vocab_size=50304, # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    block_size=128,
    n_layers=4,
    n_heads=4,
    embed_size=64,
    hidden_size=64*2,
    bias=False,
)
batch_size = 2
x = Tensor.randint(batch_size, model.block_size, low=0, high=model.vocab_size)
targets = Tensor.randint(batch_size, model.block_size, low=0, high=model.vocab_size)

logits, loss = model(x, targets=targets)
print(loss.item())