from utils import load_state_dict_from_numpy
from tinygrad.nn.state import load_state_dict
from model import GPT
from tinygrad.nn.state import get_parameters
import tiktoken
from tinygrad import Tensor

model = GPT(
    vocab_size=50304,
    block_size=256,
    n_layers=8,
    n_heads=8,
    embed_size=512,
    hidden_size=512 * 2,
    bias=False,
)

state_dict = load_state_dict_from_numpy("model")
load_state_dict(model, state_dict)

print(f"model has {sum(p.numel() for p in get_parameters(model)):,} parameters")
tokenizer = tiktoken.get_encoding("gpt2")
input_ids = Tensor([tokenizer.encode("Love is not just")])
tokens = model.generate(idx=input_ids, max_new_tokens=32)
print(tokenizer.decode(tokens[0].numpy()))
