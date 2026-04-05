import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """Your original basic CNN"""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


class BasicResNetBlock(nn.Module):
    """Simple residual block for stronger CNN"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNetEncoder(nn.Module):
    """Stronger CNN with residual blocks"""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = nn.Sequential(BasicResNetBlock(64, 64), BasicResNetBlock(64, 64))
        self.layer2 = nn.Sequential(BasicResNetBlock(64, 128, stride=2), BasicResNetBlock(128, 128))
        self.layer3 = nn.Sequential(BasicResNetBlock(128, 256, stride=2), BasicResNetBlock(256, 256))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)