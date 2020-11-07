import torch
from torch import nn

from model.utils import get_conv_transform


class GlobalAggregationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ck, cv, query_transform, dropout=0.5):
        """
        :param in_channels: int
            input channels
        :param out_channels: int
            number of channels to output
        :param ck: int
            channels of keys
        :param cv: int
            channels of values
        :param query_transform: string in ['up', 'down', 'same']
        or torch.nn.Module
            if string utils.get_conv_transform is used.
            Parameter might be a torch module that handles tenzors
            with shape of (Batch, Channels, Depth, Height, Width)
            and outputs tensor (Batch, ck, NewDepth, NewHeight, NewWidth .
        """
        super(GlobalAggregationBlock, self).__init__()
        self.ck = ck
        self.cv = cv
        self.softmax = nn.Softmax(-1)
        self.conv_1_ck = nn.Conv3d(in_channels, ck, 1)
        self.conv_1_cv = nn.Conv3d(in_channels, cv, 1)
        if type(query_transform) is str:
            self.query_transform = get_conv_transform(in_channels, ck, query_transform)
        else:
            self.query_transform = query_transform
        self.conv_1_co = nn.Conv3d(cv, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: torch tensor (Batch, Channels, Depth, Height, Width)
        :return: 5d torch tensor
        """
        queryes = self.query_transform(x)
        batch, cq, dq, hq, wq = queryes.shape
        queryes = queryes.flatten(start_dim=2, end_dim=-1)
        keys = self.conv_1_ck(x).flatten(start_dim=2, end_dim=-1)
        values = self.conv_1_cv(x).flatten(start_dim=2, end_dim=-1)
        queryes = queryes.transpose(2, 1)
        attention = torch.matmul(queryes, keys) / (self.ck ** 0.5)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        values = values.transpose(2, 1)
        output = torch.matmul(attention, values)
        output = self.conv_1_co(output.view(batch, self.cv, dq, hq, wq))
        return output


class InputBlock(nn.Module):
    def __init__(self, in_channels):
        super(InputBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm3d(in_channels)
        self.batch_norm2 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU6()
        self.conv1 = nn.Conv3d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        out = self.batch_norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        x = x + out
        return x


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm3d(in_channels)
        self.batch_norm2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU6()
        self.residual_conv = nn.Conv3d(in_channels, out_channels, 1, 2)
        self.conv1 = get_conv_transform(in_channels, out_channels, 'down')
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + residual
        return x


class BottomBlock(nn.Module):
    def __init__(self, in_channels, ck, cv):
        super(BottomBlock, self).__init__()
        self.agg_block = GlobalAggregationBlock(in_channels, in_channels, ck, cv, 'same')

    def forward(self, x):
        x = self.agg_block(x)
        return x


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ck, cv):
        super(UpSamplingBlock, self).__init__()
        self.agg_block = GlobalAggregationBlock(in_channels, out_channels, ck, cv, 'up')
        self.residual_deconv = get_conv_transform(in_channels, out_channels, 'up')

    def forward(self, x):
        residual = self.residual_deconv(x)
        x = self.agg_block(x)
        x = x + residual
        return x
