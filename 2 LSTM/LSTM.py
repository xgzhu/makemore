import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import random
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import *

if __name__ == '__main__':
    ## Set up training/testing Dataset
    random.seed(str(time.time_ns))
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    block_size = 6  # Context size
    Xtr,  Ytr  = build_dataset(words[:n1], block_size)     # 80%
    Xdev, Ydev = build_dataset(words[n1:], block_size)   # 10%

    # Hyperparameters
    embedding_dim = 32
    hidden_dim = 128
    num_layers = 1

    # Initialize the model
    model = CharLSTM(vocab_size, embedding_dim, hidden_dim, vocab_size, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Total elements = embedding_dim * vocab_size + (embedding_dim + 1) * hidden_dim + (hidden_dim + 1) * hidden_dim + (hidden_dim + 1) * vocab_size
    parameters = model.parameters()
    print(sum(p.nelement() for p in parameters)) # number of parameters in total

    # Training the LSTM
    n_epochs = 20
    batch_size = 32

    writer = SummaryWriter(log_dir='logs')
    idx = 0
    for epoch in range(n_epochs):
        for X_batch, Y_batch in create_batches(Xtr, Ytr, batch_size):
            # Forward pass
            # note: this version is not correct, it treat each splited string as separate string, while we should carry over the hidden
            outputs, _ = model(X_batch)  
            # print(outputs.shape, Y_batch.shape)
            loss = F.cross_entropy(outputs.view(-1, vocab_size), Y_batch.view(-1))  # Compute loss
        
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for tracking
            writer.add_scalar("Training Loss", loss.data.item(), idx)
            idx += 1

        outputs_dev, _ = model(Xdev)  # Detach hidden state to prevent exploding gradients
        # print(outputs.shape, Y_batch.shape)
        loss_dev = F.cross_entropy(outputs_dev.view(-1, vocab_size), Ydev.view(-1))  # Compute loss
        writer.add_scalar("Testing Loss", loss_dev.data.item(), idx)
        print(f"after epoch-{epoch}, traing loss is {loss.data.item()}, dev loss is {loss_dev.data.item()}.")

    # Flush tensorboard
    writer.close()

    # Change mode to eval
    model.eval()

    # Give some predictions
    for _ in range(20):
        name = []
        context = [0] * block_size # initialize with all ...
        # hidden = None
        while True:
            # forward pass the neural net
            out, _ = model(torch.tensor([context]))
            logits = out[:,-1,:]
            probs = F.softmax(logits, dim=1)

            token = torch.multinomial(probs, num_samples=1, replacement=True)
            context = context[1:] + [token]
            if token == 0:
                break
            name.append(token.item())
        print("".join([itos[i] for i in name]))

    PATH = "./output/saved_lstm"
    torch.save(model, PATH)