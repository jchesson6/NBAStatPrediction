import torch
from torch import nn


class LSTM_RNN(nn.Module):
    def __init__(self, num_inputs, hidden_features, num_layers=1, num_outputs=1):
        super(LSTM_RNN, self).__init__()
        self.num_inputs = num_inputs
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.lstm = nn.LSTM(num_inputs, hidden_features, num_layers) #lstm model
        self.fc_output_layer = nn.Linear(hidden_features, num_outputs) #convert predictions to outputs
        
    def forward(self, x, h0=None, c0=None):

        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_features).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_features).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc_output_layer(out)  # Selecting the last output
        return out, hn, cn