import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class PolicyHead(nn.Module):
    def __init__(self, in_channels, n_moves):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32 * 8 * 8, n_moves)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

class MxModel(nn.Module):
    def __init__(self, in_channels=60, n_blocks=16, n_moves=1715, channels=192):
        super().__init__()
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for i in range(n_blocks)])
        self.conv_out = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn_out = nn.BatchNorm2d(channels)
        self.policy_head = PolicyHead(channels, n_moves)
    def forward(self, x):
        if x.dim() == 4 and x.shape[1] != self.in_channels:
            if x.shape[3] == self.in_channels:
                x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        x = F.relu(self.bn_out(self.conv_out(x)))
        policy = self.policy_head(x)
        return policy


model = MxModel(
        in_channels=60,
        n_blocks=16,  # Increased from 10 to 16
        n_moves=1715,
        channels=192  # Increased from 128 to 192
    )