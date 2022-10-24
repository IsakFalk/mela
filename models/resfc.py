from torch import nn as nn


class ResFC(nn.Module):
    def __init__(self, input_dim, feat_dim, residual=False, layer_norm=True):
        super().__init__()

        self.feat_dim = feat_dim

        self.layer1 = nn.Linear(input_dim, feat_dim)
        self.layer2 = nn.Linear(feat_dim, feat_dim)
        self.layer3 = nn.Linear(feat_dim, feat_dim)
        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu = nn.ReLU()

        self.m = nn.LayerNorm(feat_dim)
        self.residual = residual
        self.layer_norm = layer_norm

    def forward(self, xs):
        h1 = self.leaky_relu(self.layer1(xs))
        h2 = self.leaky_relu(self.layer2(h1))
        out = self.layer3(h2)

        if self.residual:
            out = xs + out
        if self.layer_norm:
            out = self.m(out)

        return out


def resfc(input_dim=440, feat_dim=440, **kwargs):
    """
    Constructs a ResFC model.
    """
    model = ResFC(input_dim, feat_dim, **kwargs)
    return model
