import torch
import torch.nn as nn

from .rnn import LeakyRNN

cuda = torch.cuda.is_available()

if cuda:
    device = "cuda"
else:
    device = "cpu"


class HierarchicalRNNModel(nn.modules.Module):
    """
    Build the rnn model, for loop
    """

    def __init__(
        self,
        input_size,
        hidden_size1,
        hidden_size2,
        num_classes1,
        num_classes2,
        rnn_type="RNN",
        **kwargs
    ):
        super(HierarchicalRNNModel, self).__init__()

        if rnn_type == "RNN":
            self.rnn1 = nn.RNN(input_size, hidden_size1, **kwargs)
            self.rnn2 = nn.RNN(hidden_size1, hidden_size2, **kwargs)
        elif rnn_type == "LSTM":
            self.rnn1 = nn.LSTM(input_size, hidden_size1, **kwargs)
            self.rnn2 = nn.LSTM(hidden_size1, hidden_size2, **kwargs)
        elif rnn_type == "GRU":
            self.rnn1 = nn.GRU(input_size, hidden_size1, **kwargs)
            self.rnn2 = nn.GRU(hidden_size1, hidden_size2, **kwargs)
        elif rnn_type == "LeakyRNN":
            self.rnn1 = LeakyRNN(input_size, hidden_size1, **kwargs)
            self.rnn2 = LeakyRNN(hidden_size1, hidden_size2, **kwargs)
        else:
            raise ValueError("No such RNN types!")

        self.linear1 = nn.Linear(hidden_size1, num_classes1)
        self.linear2 = nn.Linear(hidden_size2, num_classes2)

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.rnn_type = rnn_type

    def init_hidden(self, batch):

        if self.rnn_type in ["RNN", "LeakyRNN"]:
            return (torch.zeros((1, batch, self.hidden_size1)).to(device),), (
                torch.zeros((1, batch, self.hidden_size2)).to(device),
            )
        if self.rnn_type == "LSTM":
            return (
                torch.zeros((1, batch, self.hidden_size1)).to(device),
                torch.zeros((1, batch, self.hidden_size1)).to(device),
            ), (
                torch.zeros((1, batch, self.hidden_size2)).to(device),
                torch.zeros((1, batch, self.hidden_size2)).to(device),
            )

    def forward(self, x):
        tsq = x.shape[0]
        batch = x.shape[1]

        hidden1, hidden2 = self.init_hidden(batch)

        ### the first rnn network
        rnn_out1, _ = self.rnn1(x, hidden1)  ## (Tseq,batch,feature_dim)
        rnn_out1_ = rnn_out1.view(-1, self.hidden_size1)
        linear_out1 = self.linear1(rnn_out1_)
        linear_out1 = linear_out1.view(tsq, batch, -1)

        ### the second rnn network
        rnn_out1 = rnn_out1.squeeze()
        rnn_out2, _ = self.rnn2(rnn_out1, hidden2)
        rnn_out2_ = rnn_out2.view(-1, self.hidden_size2)
        linear_out2 = self.linear2(rnn_out2_)
        linear_out2 = linear_out2.view(tsq, batch, -1)

        rnn_out2 = rnn_out2.squeeze()

        return linear_out1, linear_out2, rnn_out1, rnn_out2
