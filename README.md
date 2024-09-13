### a simple gpt model with training in tinygrad

this is similiar to [nano-gpt](https://github.com/karpathy/nanoGPT/tree/master) from Andrej Kaparthy
but in tinygrad

### setup environment

 - `pip install -U tinygrad`
 - `pip install tiktoken` for tokenizing training setw
 - `pip install matplotlib` for plotting the training loss

### preparing the data for training

to tokenize the data for training, create a `data/text` folder and put in
as many .txt files with text you want. then run `python3 prepare_data.py`
which will generate a `train.npy` file in the `data` folder with all the tokens
as a numpy array.

### training

run `DEFAULT_FLOAT=BFLOAT16 python3 train.py` to start a training run
config of the model can be changed in `train.py`
