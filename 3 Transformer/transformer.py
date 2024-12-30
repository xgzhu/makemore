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

    Xtr,  Ytr  = build_dataset(words[:n1], 1)     # 80%
    Xdev, Ydev = build_dataset(words[n1:], 1)   # 10%

    # Hyperparameters
    emb_size = 16
    head_size = 4
    max_len = 8
    layer_cnt = 2

    # Initialize the model
    model = TransformerModel(vocab_size, emb_size, max_len, layer_cnt)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Total elements = embedding_dim * vocab_size + (embedding_dim + 1) * hidden_dim + (hidden_dim + 1) * hidden_dim + (hidden_dim + 1) * vocab_size
    parameters = model.parameters()
    print(sum(p.nelement() for p in parameters)) # number of parameters in total

    # Training the LSTM
    n_epochs = 10
    batch_size = 32

    writer = SummaryWriter(log_dir='logs')
    idx = 0
    for epoch in range(n_epochs):
        for X_batch, Y_batch in create_batches(Xtr, Ytr, batch_size):
            # Forward pass
            # note: this version is not correct, it treat each splited string as separate string, while we should carry over the hidden
            logits, loss = model(X_batch, Y_batch)
        
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

    # Save model
    PATH = "./output/saved_lstm"
    torch.save(model, PATH)

    # Give some predictions
    results = model.generate(idx = torch.zeros((20, 1), dtype=torch.long), max_new_tokens=20).tolist()
    for res in results:
        res = res[1:]
        print("".join([itos[i] for i in res]))