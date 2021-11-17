from nonlocalunet.blocks import *


class NonLocalUnetBuilder(nn.Module):
    def __init__(self, in_channels, out_channels, depth=3, init_filters=32, agg=True, dropout=0.5):
        """
        :param in_channels: int
        :param out_channels: int
        :param depth: number of downsampling blocks
        :param init_filters: initial number of filters in first downsampling block
        :param agg: use global aggregation block in upsampling block
        :param dropout: float
        """
        super(NonLocalUnetBuilder, self).__init__()
        self.depth = depth
        self.input_block = ResidualBlock(in_channels)
        self.conv_input = nn.Conv3d(in_channels, init_filters, 1)
        self.down_sample = nn.ModuleList()
        fl = [init_filters * 2 ** i for i in range(depth + 1)]  # filter_list

        # Down sample
        for i in range(depth):
            self.down_sample.append(DownSamplingBlock(fl[i], fl[i + 1]))

        # Bottom block
        if agg:
            self.bottom = BottomAggBlock(fl[-1], fl[-1], fl[-1], dropout=dropout)
        else:
            self.bottom = BottomBlock(fl[-1])
        # Up sample
        self.up_sample = nn.ModuleList()
        for i in reversed(range(1, depth + 1)):
            if agg:
                self.up_sample.append(UpSamplingAggBlock(fl[i], fl[i - 1], fl[i - 1], fl[i - 1]))
            else:
                self.up_sample.append(UpSamplingBlock(fl[i], fl[i - 1]))

        # Output
        self.output_block = ResidualBlock(fl[0])
        self.dropout = nn.Dropout(dropout)
        self.conv_output = nn.Conv3d(fl[0], out_channels, 1)

    def forward(self, x) -> torch.tensor:
        x = x.contiguous()
        x = self.input_block(x)
        # down sampling
        down = [self.conv_input(x)]
        for i in range(self.depth):
            down.append(self.down_sample[i](down[i]))
        # bottom block
        x = self.bottom(down[-1])
        # up sampling
        for i in range(self.depth):
            x = self.up_sample[i](x)
            x = x + down[-i - 2]
        x = self.output_block(x)
        x = self.dropout(x)
        x = self.conv_output(x)
        x = torch.sigmoid(x)
        return x


class NonLocalUnet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(NonLocalUnet, self).__init__()
        self.input_block = ResidualBlock(in_channels)
        self.conv_input = nn.Conv3d(in_channels, 32, 1)
        self.down_sample1 = DownSamplingBlock(32, 64)
        self.down_sample2 = DownSamplingBlock(64, 128)
        self.bottom = BottomAggBlock(128, 128, 128, dropout=dropout)
        self.up_sample1 = UpSamplingAggBlock(128, 64, 64, 64)
        self.up_sample2 = UpSamplingAggBlock(64, 32, 32, 32)
        self.output_block = ResidualBlock(32)
        self.dropout = nn.Dropout(dropout)
        self.conv_output = nn.Conv3d(32, out_channels, 1)

    def forward(self, x) -> torch.tensor:
        x = x.contiguous()
        x = self.input_block(x)
        x1 = self.conv_input(x)
        x2 = self.down_sample1(x1)
        x = self.down_sample2(x2)
        x = self.bottom(x)
        x = self.up_sample1(x)
        x = x + x2
        x = self.up_sample2(x)
        x = x + x1
        x = self.output_block(x)
        x = self.dropout(x)
        x = self.conv_output(x)
        x = torch.sigmoid(x)
        return x

