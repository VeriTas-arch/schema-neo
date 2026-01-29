import torch
import torch.nn as nn


def bi_relu(x):
    """
    if x >0:
        f(x)=x
    if x < 0:
        f(x)=-x
    """
    return torch.nn.LeakyReLU(negative_slope=-1)(x)


class LogAct(nn.Module):

    def __init__(self, alpha=1.5, beta=4.0, gamma=0.1, thr=6.0):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.thr = thr

    def forward(self, x):
        x = (x - self.thr) / self.alpha
        x = torch.exp(x)
        return self.beta / self.gamma * torch.log1p(x)


class RecLogAct(nn.Module):
    def __init__(self, alpha=1.5, beta=4.0, gamma=0.1, thr=6.0):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.thr = thr

    def forward(self, x):

        x = torch.expm1(self.gamma / self.beta * x)
        return self.thr + self.alpha * torch.log(x)


if __name__ == "__main__":
    aa = torch.FloatTensor([1])
    bb = torch.FloatTensor([-1])
    print(bi_relu(aa))
    print(bi_relu(bb))
