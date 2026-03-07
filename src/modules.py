from .constant import *

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
        c2 = in_features // 8

        self.cnn = nn.Sequential(
            ConvBlock(1, c1),
            ConvBlock(c1, c1, pool=True),
            ConvBlock(c1, c2, pool=True)
        )

        fc_in_features = c2 * (in_features // 4)
        # 2 times pooling

        self.fc = nn.Sequential(
            nn.Linear(fc_in_features, out_features),
            nn.Dropout(0.5)
        )
    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))
        # [batch, in_features // 8, time, in_features // 4]
        x = x.transpose(1, 2)
        x = x.flatten(2)
        return self.fc(x)

class BiLSTM(nn.Module):
    def __init__(self, input_features, hidden_features):
        super().__init__()
        self.rnn = nn.LSTM(input_features, hidden_features, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.rnn(x)[0]
    
