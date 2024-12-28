from pathlib import Path
import torch

file_path = Path(__file__).parent / "names.txt"
words = open(file_path, 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

def build_dataset(words, block_size = 8):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      context = context[1:] + [ix] # crop and append
      Y.append(context)

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y

# Create batches for training
def create_batches(X, Y, batch_size):
    for i in range(0, len(X) - batch_size, batch_size):
        yield X[i:i+batch_size], Y[i:i+batch_size]