import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Define the three RNNs (Elman, Jordan, Multi-layer RNNs)
class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(ElmanRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Initial hidden state
        out, hn = self.rnn(x, h0)  # Passing the input and initial hidden state
        out = self.fc(out[:, -1, :])  # Take the output at the last time step
        return out


class JordanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(JordanRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        
        self.rnn = nn.GRU(input_size + output_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Initialize the previous output with zeros
        prev_output = torch.zeros(batch_size, 1, self.output_size).to(x.device)
        
        outputs = []
        for t in range(seq_len):
            # Concatenate input at time t with previous output
            x_t = torch.cat((x[:, t:t+1, :], prev_output), dim=2)
            
            # Pass through RNN
            out, h0 = self.rnn(x_t, h0)
            
            # Generate output
            y_t = self.fc(out)
            y_t = self.activation(y_t)
            outputs.append(y_t)
            
            # Update previous output
            prev_output = y_t
        
        # Stack all outputs
        outputs = torch.cat(outputs, dim=1)
        return outputs[:, -1, :]  # Return only the last time step
    
class MultiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.0):
        super(MultiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Initialize RNN cells for each layer
        self.rnn_cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer takes input_size + output_size as input
                input_dim = input_size + output_size
            else:
                # Subsequent layers take hidden_size + output_size as input
                input_dim = hidden_size + output_size

            rnn_cell = nn.RNNCell(input_dim, hidden_size)
            self.rnn_cells.append(rnn_cell)

        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Initialize hidden states and previous outputs for each layer
        h_t = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        y_t_prev = [torch.zeros(batch_size, self.output_size).to(x.device) for _ in range(self.num_layers)]

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # Current input at time t

            # Process through each layer
            for i, rnn_cell in enumerate(self.rnn_cells):
                if i == 0:
                    # First layer input: concatenation of x_t and previous output of layer i
                    combined_input = torch.cat([x_t, y_t_prev[i]], dim=1)
                else:
                    # Subsequent layers input: concatenation of previous layer's hidden state and previous output of layer i
                    combined_input = torch.cat([h_t[i - 1], y_t_prev[i]], dim=1)

                # Update hidden state
                h_t[i] = rnn_cell(combined_input, h_t[i])

                # Apply dropout if needed
                h_t_drop = self.dropout(h_t[i])

                # Compute current output for this layer
                y_t = self.fc(h_t_drop)

                # Update previous output for this layer
                y_t_prev[i] = y_t

            # Append the output from the last layer
            outputs.append(y_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, seq_len, output_size)
        return outputs[:, -1, :]  # Return the last output