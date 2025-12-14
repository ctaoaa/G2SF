import torch
import torch.nn as nn
from .mlp import MLPnet


class MultiModalGateNet(nn.Module):
    def __init__(self, args, dim=1920, n_hidden='4096'):
        super(MultiModalGateNet, self).__init__()
        self.img_dim = 768
        self.pcd_dim = 1152
        self.dim = dim
        self.drop_out = args.drop_out

        self.residual_transformer = MLPnet(self.dim, n_hidden=n_hidden, n_output=args.emb_dim, bias=True, dropout=args.drop_out)
        self.transformer = MLPnet(self.dim, n_hidden=n_hidden, n_output=args.emb_dim, bias=True, dropout=args.drop_out)
        final_rep_dim = 2 * args.emb_dim
        # 输出是二维
        self.regressor = nn.Sequential(nn.Linear(final_rep_dim, 256), nn.ReLU(), nn.Dropout(self.drop_out), nn.Linear(256, 2))

    def forward(self, f, f0):
        f = self.residual_transformer(f)
        f0 = self.transformer(f0)
        p = self.regressor(torch.cat([f, f0], dim=1))
        return torch.exp(torch.tanh(p))



