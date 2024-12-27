import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import random
import time
import torch.nn.functional as F
import torch.optim as optim

from utils import *

if __name__ == '__main__':
    ## Set up training/testing Dataset
    random.seed(time.time_ns)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    block_size = 6  # Context size
    Xtr,  Ytr  = build_dataset(words[:n1],   block_size)     # 80%
    Xdev, Ydev = build_dataset(words[n1:n2], block_size)   # 10%
    Xte,  Yte  = build_dataset(words[n2:],   block_size)     # 10%

    # Hyperparameters
    embedding_dim = 32
    hidden_dim = 128

    # Initialize the model
    model = CharRNN(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Total elements = embedding_dim * vocab_size + (embedding_dim + 1) * hidden_dim + (hidden_dim + 1) * hidden_dim + (hidden_dim + 1) * vocab_size
    parameters = model.parameters()
    print(sum(p.nelement() for p in parameters)) # number of parameters in total

    # Training the RNN
    n_epochs = 10
    batch_size = 32

    loss_train = []
    for i in range(n_epochs):
        for X_batch, Y_batch in create_batches(Xtr, Ytr, batch_size):
            # Forward pass
            # note: this version is not correct, it treat each splited string as separate string, while we should carry over the hidden
            outputs, _ = model(X_batch)  
            # print(outputs.shape, Y_batch.shape)
            loss = F.cross_entropy(outputs, Y_batch)  # Compute loss
        
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for tracking
            loss_train.append(loss.data.item())

            # for debugging purpose
            break

        print(f"after epoch-{i}, loss is {loss.data.item()}.")
