import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_str
from torchinfo import summary
import torch.nn.functional as F

from ConvQuadraticOperation import ConvQuadraticOperation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()

# MetaAconC (Adaptive Activation)
class MetaAconC(nn.Module):
    def __init__(self, width, r=16):
        super().__init__()
        self.fc1 = nn.Conv1d(width, max(r, width // r), kernel_size=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm1d(max(r, width // r))
        self.fc2 = nn.Conv1d(max(r, width // r), width, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm1d(width)
        self.p1 = nn.Parameter(torch.randn(1, width, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1))

    def forward(self, x):
        beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(x.mean(dim=2, keepdims=True))))))
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

# CoordAtt (Attention Mechanism)
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_w = nn.AdaptiveMaxPool1d(1)
        mip = max(6, inp // reduction)
        self.conv1 = nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(mip)
        self.act = MetaAconC(mip)
        self.conv_w = nn.Conv1d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, w = x.size()
        x_w = self.pool_w(x)
        y = torch.cat([identity, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_ww, x_c = torch.split(y, [w, 1], dim=2)
        a_w = self.conv_w(x_ww)
        a_w = a_w.sigmoid()
        out = identity * a_w
        return out

class ResNeStShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=1, padding=0):
        super(ResNeStShortcut, self).__init__()
        self.conv = ConvQuadraticOperation(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class QDCA_BiGRU_Res(nn.Module):
    def __init__(self):
        super(QDCA_BiGRU_Res, self).__init__()

        # Path 1: Low-frequency (large kernel)
        self.path1_conv1 = ConvQuadraticOperation(1, 16, 64, 8, 28)
        self.path1_bn1 = nn.BatchNorm1d(16)
        self.path1_maxpool1 = nn.MaxPool1d(2, 2)
        self.path1_relu1 = nn.ReLU()

        # Path 2: High-frequency (small kernels with deeper layers)
        self.path2_resnet_shortcut1_2 = ConvQuadraticOperation(64, 32, 2, 2, 0)
        self.path2_bn1_2 = nn.BatchNorm1d(32)

        self.path2_conv2 = ConvQuadraticOperation(64, 32, 3, 1, 1)  # Changed in_channels from 16 to 64
        self.path2_bn2 = nn.BatchNorm1d(32)
        self.path2_maxpool2 = nn.MaxPool1d(2, 2)
        self.path2_relu2 = nn.ReLU()

        self.path2_resnet_shortcut2_3 = ConvQuadraticOperation(32, 64, 2, 2, 0)
        self.path2_bn2_3 = nn.BatchNorm1d(64)

        self.path2_conv3 = ConvQuadraticOperation(32, 64, 3, 1, 1)
        self.path2_bn3 = nn.BatchNorm1d(64)
        self.path2_maxpool3 = nn.MaxPool1d(2, 2)
        self.path2_relu3 = nn.ReLU()

        # Alignment layers to match dimensions before fusion
        self.path1_align = nn.Conv1d(16, 64, kernel_size=1, stride=1)
        self.path1_pool = nn.AdaptiveAvgPool1d(32)

        # Fusion and Attention
        self.fusion_resnet_shortcut = ConvQuadraticOperation(64, 64, 1, 1, 0)
        self.fusion_bn = nn.BatchNorm1d(64)
        self.attention = CoordAtt(64, 64)

        # BiGRU for temporal modeling
        self.bigru = nn.GRU(64, 64, bidirectional=True)

        # Final layers
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 100)
        self.relu1_fc = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 10)

    def get_features(self, x):
        # Path 1
        out1 = self.path1_relu1(self.path1_maxpool1(self.path1_bn1(self.path1_conv1(x))))
        out1 = self.path1_align(out1)
        out1 = self.path1_pool(out1)

        # Path 2 with ResNet shortcuts
        shortcut1_2 = self.path2_bn1_2(self.path2_resnet_shortcut1_2(out1))
        out2 = self.path2_maxpool2(self.path2_bn2(self.path2_conv2(out1)))
        out2 = self.path2_relu2(shortcut1_2 + out2)

        shortcut2_3 = self.path2_bn2_3(self.path2_resnet_shortcut2_3(out2))
        out3 = self.path2_maxpool3(self.path2_bn3(self.path2_conv3(out2)))
        out3 = self.path2_relu3(shortcut2_3 + out3)

        # Fusion with mul and shortcut
        if out1.shape[2] != out3.shape[2]:
            out1 = nn.functional.interpolate(out1, size=out3.shape[2], mode='linear', align_corners=False)
        fused = torch.mul(out1, out3)
        shortcut_fusion = self.fusion_bn(self.fusion_resnet_shortcut(fused))
        fused = fused + shortcut_fusion

        # Attention
        att = self.attention(fused)

        # BiGRU
        att = att.permute(0, 2, 1)
        gru_out, _ = self.bigru(att)
        gru_out = gru_out.permute(0, 2, 1)

        # GAP and FC
        out = self.gap(gru_out).view(gru_out.size(0), -1)
        out = self.relu1_fc(self.fc1(out))
        out = self.dp(out)
        out = self.fc2(out)

        return out

    def forward(self, x):
        # Path 1
        out1 = self.path1_relu1(self.path1_maxpool1(self.path1_bn1(self.path1_conv1(x))))
        out1 = self.path1_align(out1)
        out1 = self.path1_pool(out1)

        # Path 2 with ResNet shortcuts
        shortcut1_2 = self.path2_bn1_2(self.path2_resnet_shortcut1_2(out1))
        out2 = self.path2_maxpool2(self.path2_bn2(self.path2_conv2(out1)))
        out2 = self.path2_relu2(shortcut1_2 + out2)

        shortcut2_3 = self.path2_bn2_3(self.path2_resnet_shortcut2_3(out2))
        out3 = self.path2_maxpool3(self.path2_bn3(self.path2_conv3(out2)))
        out3 = self.path2_relu3(shortcut2_3 + out3)

        # Fusion with mul and shortcut
        if out1.shape[2] != out3.shape[2]:
            out1 = nn.functional.interpolate(out1, size=out3.shape[2], mode='linear', align_corners=False)
        fused = torch.mul(out1, out3)
        shortcut_fusion = self.fusion_bn(self.fusion_resnet_shortcut(fused))
        fused = fused + shortcut_fusion

        # Attention
        att = self.attention(fused)

        # BiGRU
        att = att.permute(0, 2, 1)
        gru_out, _ = self.bigru(att)
        gru_out = gru_out.permute(0, 2, 1)

        # GAP and FC
        out = self.gap(gru_out).view(gru_out.size(0), -1)
        out = self.relu1_fc(self.fc1(out))
        out = self.dp(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    X = torch.rand(2, 1, 2048).cuda()
    m = QDCA_BiGRU_Res().cuda()
    summary(m, (2, 1, 2048))
    print(flop_count_str(FlopCountAnalysis(m, X)))
    for name, param in m.named_parameters():
        print(f"Parameter name: {name}")

