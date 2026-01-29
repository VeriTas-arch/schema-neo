import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter

from .osci_leaky_rnn import OsciLeakyRNN
from .rnn import LeakyRNN

cuda = torch.cuda.is_available()

if cuda:
    device = "cuda"
else:
    device = "cpu"


class RNNModel(nn.modules.Module):
    """
    Build the rnn model, for loop
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        rnn_type="RNN",
        h0_trainable=False,
        **kwargs
    ):
        super(RNNModel, self).__init__()

        if rnn_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, **kwargs)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, **kwargs)
        elif rnn_type == "GRU":
            self.rnn = nn.GRUCell(input_size, hidden_size, **kwargs)
        elif rnn_type == "LeakyRNN":
            self.rnn = LeakyRNN(input_size, hidden_size, **kwargs)
        elif rnn_type == "OsciLeakyRNN":
            self.rnn = OsciLeakyRNN(input_size, hidden_size, **kwargs)
        else:
            raise ValueError("No such RNN types!")

        self.linear = nn.Linear(hidden_size, num_classes)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.rnn_type = rnn_type
        self.h0 = None

        ### for GRU
        if h0_trainable:
            if rnn_type == "GRU":
                self.h0 = Parameter(torch.Tensor(1, self.hidden_size)).to(device)
                h_factor = 1.0 / np.sqrt(self.hidden_size)
                self.h0.data.uniform_(-h_factor, h_factor)
            elif rnn_type == "LSTM":
                self.h0 = Parameter(torch.Tensor(1, 2 * self.hidden_size)).to(device)
                h_factor = 1.0 / np.sqrt(2 * self.hidden_size)
                self.h0.data.uniform_(-h_factor, h_factor)
            # raise ValueError(self.h0)

    def init_hidden(self, batch):

        if self.rnn_type in ["RNN", "LeakyRNN"] and self.h0 is None:
            return torch.zeros((1, batch, self.hidden_size)).to(device)
        elif self.rnn_type == "LSTM" and self.h0 is None:
            # return (torch.zeros((1,batch,self.hidden_size)).to(device),torch.zeros((1,batch,self.hidden_size)).to(device))
            return torch.zeros((1, batch, 2 * self.hidden_size)).to(device)
        elif self.rnn_type == "OsciLeakyRNN" and self.h0 is None:
            return torch.zeros((1, batch, self.hidden_size)).to(device)

        else:
            return None

    def forward(self, x, hidden_t=None, return_hidden=False):
        tsq = x.shape[0]
        batch = x.shape[1]

        # if hidden_t is not None:
        #         raise ValueError(hidden_t.shape)
        #         raise ValueError(hidden_t)
        # for LSTM, the received hidden_t is not None, with the shape of (batch, hidden_size), used by GRU

        # if hidden_t is None:
        #         raise ValueError("Failed to initialize hidden_t")

        if hidden_t is None:  # for LSTM, delete h0 from hiddens = model(data_x,j0)
            hidden_t = self.init_hidden(batch)  # however, for LSTM, h0 is not None
            # if hidden_t is None:
            #     raise ValueError(self.h0)

        if self.rnn_type == "LSTM":
            hidden_t = hidden_t.unsqueeze(0)
            hidden_t = (
                hidden_t[:, :, : self.hidden_size],
                hidden_t[:, :, self.hidden_size :],
            )

        ### for LSTM
        tot_hidden = []

        if self.rnn_type == "LSTM":
            rnn_out = []
            for t in range(tsq):
                rnn_out_t, hidden_t = self.rnn(
                    x[t][None, ...], hidden_t
                )  ## (Tseq,batch,feature_dim)
                tot_hidden.append(torch.cat(hidden_t, dim=-1))

                rnn_out.append(rnn_out_t)
            rnn_out = torch.stack(rnn_out, dim=0).squeeze()

        if self.rnn_type == "LeakyRNN":
            rnn_out = self.rnn(x, hidden_t)

        if self.rnn_type == "GRU":
            rnn_out = []
            for t in range(tsq):
                rnn_out_t = self.rnn(x[t], hidden_t)
                hidden_t = rnn_out_t
                rnn_out.append(rnn_out_t)
            rnn_out = torch.stack(
                rnn_out, dim=0
            ).squeeze()  ### shape (T, batch, feature_dim)

        rnn_out_ = rnn_out.view(-1, self.hidden_size)
        linear_out = self.linear(rnn_out_)
        linear_out = linear_out.view(tsq, batch, -1)

        if self.rnn_type == "LSTM":
            tot_hidden = torch.stack(tot_hidden).squeeze(
                dim=1
            )  ### shape is [Tseq, batch, n_hidden]

        if return_hidden:
            return linear_out, rnn_out, tot_hidden
        else:
            return linear_out, rnn_out
