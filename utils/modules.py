
import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers = 1):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Character embedding layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)  # Simple RNN layer
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Fully connected output layer
        self.layers = [self.embedding, self.rnn, self.fc]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden = None):
        x = self.embedding(x)  # Embed input characters
        out, hidden = self.rnn(x, hidden)  # Pass through RNN, if hidden is None, it will initialized with zeros
        out = self.fc(out)  # Pass through fully connected layer
        return out[:, -1, :], hidden

    def parameters(self):
        # get parameters of all layers and stretch them out into one list
        return [p for layer in self.layers for p in layer.parameters()]