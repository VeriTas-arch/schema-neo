import torch
import torch.nn as nn

cuda = torch.cuda.is_available()

if cuda:
    device = "cuda"
else:
    device = "cpu"


class ConjModel(nn.modules.Module):
    """
    This model split the rule model and senseory inputs away, and the rule representation and
    the sensory input are associated togther.

    Using such a model, wo can generalize the rule based retrieval to some memory recall tasks.

    rule rnn
    [][]
    [][]--- [][]
          / [][] --- output
    ---- /  [][]
    input conj_rnn

    transform the rule based recall task as a simple memorizing and association task.
    """

    def __init__(
        self, rule_rnn_type, conj_rnn_type, rule_output_size, conj_output_size, **kwargs
    ):
        super().__init__()

        self.rule_rnn_type = rule_rnn_type
        self.conj_rnn_type = conj_rnn_type
        self.rule_output_size = rule_output_size
        self.conj_output_size = conj_output_size
        self.rule_rnn_dict = kwargs["rule_rnn_dict"]
        self.conj_rnn_dict = kwargs["conj_rnn_dict"]

        if self.rule_rnn_type == "RNN":
            self.rule_rnn = nn.RNN(**self.rule_rnn_dict)
        elif self.rule_rnn_type == "LSTM":
            self.rule_rnn = nn.LSTM(**self.rule_rnn_dict)
        elif self.rule_rnn_type == "FastWeightRNN":
            pass
        else:
            raise ValueError(
                "This rule RNN type {} is not defined !".format(self.rule_rnn_type)
            )

        if self.conj_rnn_type == "RNN":
            self.conj_rnn = nn.RNN(**self.conj_rnn_dict)
        elif self.conj_rnn_type == "LSTM":
            self.conj_rnn = nn.LSTM(**self.conj_rnn_dict)
        elif self.conj_rnn == "FastWeightRNN":
            pass
        else:
            raise ValueError(
                "This conj RNN type {} is not defined !".format(self.conj_rnn_dict)
            )

        self.linear_conj = nn.Linear(
            self.conj_rnn_dict["hidden_size"], conj_output_size
        )
        self.linear_rule = nn.Linear(
            self.rule_rnn_dict["hidden_size"], rule_output_size
        )

    def forward(self, x, rule_hd=None, conj_hd=None):

        rule_x, conj_x = x

        seq, batch, n_rule = rule_x.shape

        if rule_hd is None:
            rule_hd = torch.zeros(
                1,
                batch,
                self.rule_rnn_dict["hidden_size"],
                dtype=rule_x.dtype,
                device=rule_x.device,
            )
            if self.rule_rnn_type == "LSTM":
                conj_hd = (rule_hd, rule_hd)
        if conj_hd is None:
            conj_hd = torch.zeros(
                1,
                batch,
                self.conj_rnn_dict["hidden_size"],
                dtype=rule_x.dtype,
                device=rule_x.device,
            )
            if self.conj_rnn_type == "LSTM":
                conj_hd = (conj_hd, conj_hd)
        ### rule rnn

        rule_output, rule_hid = self.rule_rnn(rule_x, rule_hd)
        lin_rule_output = self.linear_rule(
            rule_output.view(-1, self.rule_rnn_dict["hidden_size"])
        )
        lin_rule_output = lin_rule_output.view(seq, batch, self.rule_output_size)

        conj_tot_x = torch.cat([lin_rule_output, conj_x], dim=2)

        ### conj_rnn;

        conj_output, _ = self.conj_rnn(conj_tot_x, conj_hd)
        ### conj linear readout
        lin_conj_output = self.linear_conj(
            conj_output.view(-1, self.conj_rnn_dict["hidden_size"])
        )
        lin_conj_output = lin_conj_output.view(seq, batch, self.conj_output_size)

        return lin_rule_output, lin_conj_output
