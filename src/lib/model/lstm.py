import torch.nn as nn


class LSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
