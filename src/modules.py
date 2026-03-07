from constant import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if pool:
            layers.extend([nn.MaxPool2d((1, 2)), nn.Dropout(0.25)])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class AcousticExtractor(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        c1 = in_features // 16