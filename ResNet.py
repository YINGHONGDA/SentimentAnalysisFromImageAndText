import torch.nn as nn


class Resnet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        block, n_blocks, channles = config
        self.in_channles = channles[0]
        assert len(n_blocks) == len(channles) == 4

        self.conv1 = nn.Conv2d(3, self.in_channles, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channles)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channles[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channles[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channles[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channles[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channles, output_dim)

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):
        layers = []

        if self.in_channles != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        layers.append(block(self.in_channles, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))
        self.in_channles = block.expansion * channels
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x, h
