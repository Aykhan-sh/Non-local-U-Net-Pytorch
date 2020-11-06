from model.blocks import *


class NonLocalUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NonLocalUnet, self).__init__()
        self.input_block = InputBlock(in_channels)
        self.conv_input = nn.Conv3d(in_channels, 32, 1)
        self.down_sample1 = DownSamplingBlock(32, 64)
        self.down_sample2 = DownSamplingBlock(64, 128)
        self.bottom = BottomBlock(128, 128, 128)
        self.up_sample1 = UpSamplingBlock(128, 64, 64, 64)
        self.up_sample2 = UpSamplingBlock(64, 32, 32, 32)
        self.output_block = InputBlock(32)
        self.conv_output = nn.Conv3d(32, out_channels, 1)
        self.sg = nn.Sigmoid()

    def forward(self, x) -> torch.tensor:
        suka = []  # TODO rename suka
        x = self.input_block(x)
        print(x.is_contiguous())
        x = self.conv_input(x)
        print(x.is_contiguous())

        suka.append(x)
        x = self.down_sample1(x)
        print(x.is_contiguous())

        suka.append(x)
        x = self.down_sample2(x)
        print(x.is_contiguous())

        x = self.bottom(x)
        print(x.is_contiguous())

        x = self.up_sample1(x)
        print(x.is_contiguous())

        x = x + suka[-1]
        print(x.is_contiguous())

        x = self.up_sample2(x)
        print(x.is_contiguous())

        x = x + suka[-2]
        print(x.is_contiguous())

        x = self.output_block(x)
        print(x.is_contiguous())

        x = self.conv_output(x)
        print(x.is_contiguous())

        x = self.sg(x)
        print(x.is_contiguous())

        return x

