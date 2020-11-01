import torch
from torch import nn


def dim(x):
    print(x.shape)


class GlobalAggregationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ck, cv, query_transform=None):
        super(GlobalAggregationBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ck = ck
        self.cv = cv
        self.softmax = nn.Softmax(-1)
        self.conv_1_ck = nn.Conv3d(in_channels=in_channels, out_channels=ck, kernel_size=1)
        self.conv_1_cv = nn.Conv3d(in_channels=in_channels, out_channels=cv, kernel_size=1)
        self.query_transform = query_transform
        if self.query_transform is None:
            self.query_transform = nn.Conv3d(in_channels=in_channels, out_channels=ck, kernel_size=1)
        self.conv_1_co = nn.Conv3d(in_channels=self.query_transform.out_channels,
                                   out_channels=self.out_channels, kernel_size=1)

    def forward(self, X) -> torch.tensor:
        Q = self.query_transform(X)
        batch, cq, dq, hq, wq = Q.shape
        dim(Q)
        Q = Q.view(batch, cq, -1)
        dim(Q)
        K = self.conv_1_ck(X).view(batch, self.ck, -1)
        dim(K)
        V = self.conv_1_cv(X).view(batch, self.cv, -1)
        dim(V)
        A = self.softmax(torch.matmul(Q.permute((0, 2, 1)), K) / (self.ck ** 0.5))
        dim(A)
        O = torch.matmul(A, V.permute(0,2,1))
        dim(O)
        Y = self.conv_1_co(O.view(batch, cq, dq, hq, wq))
        dim(Y)
        return Y
