import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict


def save_model(model, filename: str):
    state_dict = get_state_dict(model)
    np_state_dict = {name: tensor.numpy() for name, tensor in state_dict.items()}
    np.savez(f"{filename}.npz", **np_state_dict)


def load_state_dict_from_numpy(filename: str) -> dict[str:Tensor]:
    loaded_dict = np.load(f"{filename}.npz")
    loaded_dict = {key: loaded_dict[key] for key in loaded_dict.files}
    return {name: Tensor(array) for name, array in loaded_dict.items()}
