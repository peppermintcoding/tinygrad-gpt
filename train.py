import matplotlib.pyplot as plt
import numpy as np
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import CI, trange
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from utils import save_model

from model import GPT

plt.rcParams["figure.figsize"] = (16, 8)


vocab_size = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
block_size = 256
max_steps = 10
batch_size = 32

data = np.load("data/train.npy")
print(f"data set has {len(data):,} tokens")


def get_batch():
    ix = [
        np.random.randint(low=0, high=len(data) - block_size) for _ in range(batch_size)
    ]
    x = [np.array(data[i : i + block_size]) for i in ix]
    y = [np.array(data[i + 1 : i + 1 + block_size]) for i in ix]
    return Tensor(x, dtype=dtypes.int64, requires_grad=False), Tensor(
        y, dtype=dtypes.int64
    )


model = GPT(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layers=8,
    n_heads=8,
    embed_size=512,
    hidden_size=512 * 2,
    bias=False,
)
print(f"model has {sum(p.numel() for p in get_parameters(model)):,} parameters")
optim = AdamW(params=get_parameters(model), lr=4e-4, b1=0.9, b2=0.95, weight_decay=0.1)


@TinyJit
def train_step(x: Tensor, y: Tensor):
    _, loss = model(x, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.realize()


with Tensor.train():
    losses, accuracies = [], []
    x, y = get_batch()
    for i in (t := trange(max_steps, disable=CI)):
        loss = train_step(x, y)
        x, y = get_batch()
        losses.append(loss.numpy())
        t.set_description(f"loss: {loss.numpy():.4f}")

save_model(model, "model")
plt.plot(losses)
