# Copyright 2018 Amazon.com, Inc.
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0
import torch
import torch.nn as nn
from typing import List

# Placeholder for distribution output (implement or integrate your own)
class DistributionOutput:
    pass

def make_rnn_cell(
    num_cells: int,
    num_layers: int,
    cell_type: str,
    residual: bool,
    dropout_rate: float,
) -> nn.Module:
    """
    Creates an RNN layer using PyTorch built-in modules.
    """
    if cell_type == "lstm":
        rnn_layer = nn.LSTM(
            input_size=num_cells,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            batch_first=True,
        )
    elif cell_type == "gru":
        rnn_layer = nn.GRU(
            input_size=num_cells,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            batch_first=True,
        )
    else:
        raise ValueError(f"Unsupported cell_type: {cell_type}")

    # Simple approach to mimic residual by combining outputs
    if residual:
        # Wrap the RNN so residual can be applied in forward
        class ResidualRNN(nn.Module):
            def __init__(self, rnn):
                super().__init__()
                self.rnn = rnn

            def forward(self, x, hidden=None):
                out, h = self.rnn(x, hidden)
                # Residual connection: add input to output if shapes match
                if out.shape == x.shape:
                    out += x
                return out, h

        return ResidualRNN(rnn_layer)
    else:
        return rnn_layer


class DeepVARNetwork(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        distr_output: DistributionOutput,
        dropout_rate: float,
        lags_seq: List[int],
        target_dim: int,
        cardinality: List[int] = [1],
        embedding_dimension: int = 1,
        scaling: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.target_dim = target_dim
        self.scaling = scaling

        # Example embedding if needed for categorical fields
        self.embeddings = nn.ModuleList([
            nn.Embedding(card_size, embedding_dimension)
            for card_size in cardinality
        ])

        # Simple scaler placeholders
        self.use_scaling = scaling
        if self.use_scaling:
            # Implement actual scaling logic if needed
            pass

        # RNN
        self.rnn_input_size = num_cells  # or adapt if you have extra features
        self.rnn = make_rnn_cell(
            num_cells=self.rnn_input_size,
            num_layers=num_layers,
            cell_type=cell_type,
            residual=True,  # or False, depending on usage
            dropout_rate=dropout_rate,
        )

        # Distribution output placeholder
        self.distr_output = distr_output

        # Fully connected layer for final output
        self.proj_distr_args = nn.Linear(num_cells, target_dim)

    def forward(self, x, *args, **kwargs):
        """
        x shape is expected as (batch, sequence_length, features).
        """
        # Optionally embed categorical inputs or scale data
        # ...

        # RNN forward
        rnn_out, _ = self.rnn(x)
        # Take the last time step (context end)
        last_state = rnn_out[:, -1, :]

        # Project to distribution parameters
        distr_params = self.proj_distr_args(last_state)

        return distr_params
