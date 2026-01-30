import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter


class SimpleRNN(nn.Module):
    """
    Build the rnn model, for loop

    RNN dynamics:
    x(t+1) = x(t) + dt/tau * (-x(t) + Wr * tanh[x(t)] + Win * U + bias)

    g, modulate the distribution of the rnn weights.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation_fn="tanh",
        g=0.9,
        dt=0.05,
        tau=1,
        h0_trainable=False,
        **kwargs,
    ):
        super(SimpleRNN, self).__init__(**kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt = dt
        self.tau = tau
        self.g = g

        if activation_fn == "tanh":
            self.activation_fn = torch.tanh
        elif activation_fn == "relu":
            self.activation_fn = torch.relu
        else:
            raise ValueError("No such activation function !")

        self.Win = Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.Wr = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias = Parameter(torch.Tensor(1, self.hidden_size))
        self.Wout = Parameter(torch.Tensor(self.hidden_size, self.output_size))

        ### for GRU
        if h0_trainable:
            self.h0 = Parameter(torch.Tensor(1, self.hidden_size))
            h_factor = 1.0 / np.sqrt(self.hidden_size)
            self.h0.data.uniform_(-h_factor, h_factor)

        self.init_weights()

    def init_hidden(self, batch):
        #! remember to use this function when analyzing the hidden state
        #! so that the initial hidden state is always zero
        # If the model defines a trainable h0, use it (expanded to batch).
        if hasattr(self, "h0"):
            # self.h0.shape ---> (1, hidden_size)
            return self.h0.expand(batch, -1).contiguous()

        # create zeros on the same device and dtype as model parameters
        param = next(self.parameters(), None)
        device = param.device if param is not None else None
        dtype = param.dtype if param is not None else None
        return torch.zeros((batch, self.hidden_size), device=device, dtype=dtype)

    def init_weights(self):
        ## init input and output weights
        self.Win.data.normal_(0, 1 / self.hidden_size)
        self.Wout.data.normal_(0, 1 / self.hidden_size)
        self.bias.data.zero_()

        ## init rnn connections
        self.Wr.data.normal_(0, self.g**2 / self.hidden_size)

    def forward(self, x, hidden_init=None):
        """
        x.shape ---> (tsq, batch, n_input)
        """
        # Prefer vectorized path: move time-major -> batch-major and precompute
        # input projection. For models with LayerNorm enabled we fall back to the
        # original path because LayerNorm + activation is a small extra cost and
        # complicates scripting.
        tsq = x.shape[0]
        batch = x.shape[1]

        # bring x to the right device/dtype once
        x = x.to(device=self.Win.device, dtype=self.Win.dtype)

        # batch-major for easier matmuls: (batch, tsq, input_size)
        x_b = x.transpose(0, 1)

        # precompute input projection for all time steps: (batch, tsq, hidden)
        input_proj = x_b @ self.Win

        # initial hidden
        if hidden_init is None:
            h0 = self.init_hidden(batch)
        else:
            param = next(self.parameters(), None)
            if param is not None:
                device = param.device
                dtype = param.dtype
                h0 = hidden_init.to(device=device, dtype=dtype)
            else:
                raise ValueError("failed to init hidden")

        tot_input_list = []
        tot_rnnhid_list = []
        tot_output_list = []

        for t in range(tsq):
            linear_input_t = input_proj[:, t, :]
            h0 = (1 - self.dt / self.tau) * h0 + self.dt / self.tau * (
                self.activation_fn(h0) @ self.Wr + linear_input_t + self.bias
            )
            rnn_out_t = self.activation_fn(self.ln_hidden(h0))
            linear_output_t = rnn_out_t @ self.Wout

            tot_input_list.append(linear_input_t)
            tot_rnnhid_list.append(h0)
            tot_output_list.append(linear_output_t)

        tot_input_tensor = torch.stack(tot_input_list, dim=0).transpose(0, 1)
        tot_rnnhid_tensor = torch.stack(tot_rnnhid_list, dim=0).transpose(0, 1)
        tot_output_tensor = torch.stack(tot_output_list, dim=0).transpose(0, 1)
        output_tensor = torch.stack([t for t in tot_output_list], dim=0)

        return output_tensor, (tot_input_tensor, tot_rnnhid_tensor, tot_output_tensor)
