import torch
from torch import nn


class ELU(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta


class FeedForward(nn.Module):
    def __init__(self, n_inp, n_out, activation, use_residual=False, use_norm=True):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        self.norm = nn.LayerNorm(n_out) if use_norm else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        self.use_residual = use_residual and (n_inp == n_out)

    def forward(self, x):
        y = self.linear(x)
        y = self.norm(y)
        y = self.activation(y)
        if self.use_residual:
            return y + x
        return y


class Model(nn.Module):
    def __init__(self, n_inp, n_out, hidden_dims=[2048, 2048, 1024, 1024],
                 activations=None, residual=True, norm=True, out_activation=None, include_tanhshrink=False):
        super().__init__()

        num_layers = len(hidden_dims)
        if activations is None:
            activations = [nn.LeakyReLU(0.1, inplace=True)] * (num_layers - 1)
            activations.append(ELU(alpha=0.1, beta=0.1))
        elif not isinstance(activations, list):
            activations = [activations]* (num_layers - 1)
            activations.append(ELU(alpha=0.1, beta=0.1))

        if len(activations) != num_layers:
            raise ValueError(f"Expected {num_layers} activations, but got {len(activations)}")

        # 1. Build hidden layers
        layers = []
        current_dim = n_inp
        for h_dim, act in zip(hidden_dims, activations):
            layers.append(FeedForward(
                current_dim,
                h_dim,
                act,
                use_residual=residual,
                use_norm=norm
            ))
            current_dim = h_dim
        self.net_lat = nn.Sequential(*layers)

        # 2. Final Output Construction
        self.net_out = nn.Linear(current_dim, n_out)
        #self.out_elu = ELU(alpha=0.1, beta=0.1)
        #self.net_out2 = nn.Linear(n_out, n_out)
        self.out_act = out_activation if out_activation is not None else ELU(alpha=1.0, beta=1.0)
        # Tanhshrink is now placed AFTER RELU
        self.tanhshrink = nn.Tanhshrink() if include_tanhshrink else nn.Identity()

    def forward(self, x):
        x = self.net_lat(x)
        x = self.net_out(x)
        #x = self.out_elu(x)
        #x = self.net_out2(x)
        x = self.out_act(x)
        x = self.tanhshrink(x)
        return x