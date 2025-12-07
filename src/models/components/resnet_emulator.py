import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual 
        out = self.act(out)
        return out

class StokesResNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=400, hidden_dim=256, n_blocks=4):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.act(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

class PhysicsLoss(nn.Module):
    def __init__(self, derivative_weight=1.0, **kwargs):
        super().__init__()
        self.lambda_d = derivative_weight
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        loss_val = self.mse(y_pred, y_true)
        
        if self.lambda_d > 0:
            diff_true = y_true[:, 1:] - y_true[:, :-1]
            diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
            loss_deriv = self.mse(diff_pred, diff_true)
        else:
            loss_deriv = 0.0
        
        return loss_val + (self.lambda_d * loss_deriv)