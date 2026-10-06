import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvQuadraticOperation import ConvQuadraticOperation  # فرض می‌کنیم لایه کوانتیک تو اینجاست

class SEBlock(nn.Module):
    """بلوک Squeeze-Excitation برای توجه به کانال‌ها"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        avg_pool = F.avg_pool1d(x, x.size(2), stride=1).view(x.size(0), x.size(1))  # pooling سراسری
        out = torch.relu(self.fc1(avg_pool))
        out = torch.sigmoid(self.fc2(out)).view(x.size(0), x.size(1), 1)
        return x * out  # مقیاس‌بندی کانال‌ها

class MSQAResNet(nn.Module):
    def __init__(self):
        super(MSQAResNet, self).__init__()
        
        # ورودی چندمقیاسی: کانولوشن‌های کوانتیک موازی با کرنل‌های مختلف
        self.scale1 = ConvQuadraticOperation(1, 16, 32, 4, 14)  # کرنل کوچک برای فرکانس بالا
        self.scale2 = ConvQuadraticOperation(1, 16, 64, 4, 30)  # کرنل متوسط
        self.scale3 = ConvQuadraticOperation(1, 16, 128, 4, 62) # کرنل بزرگ برای فرکانس پایین
        self.bn_entry = nn.BatchNorm1d(48)  # الحاق 16*3=48
        self.maxpool_entry = nn.MaxPool1d(2, 2)
        self.relu_entry = nn.ReLU()
        
        # بلوک‌های رزنت با کوانتیک + SE
        self.res_shortcut1 = ConvQuadraticOperation(48, 64, 2, 2, 0)
        self.bn_shortcut1 = nn.BatchNorm1d(64)
        
        self.conv1 = ConvQuadraticOperation(48, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.se1 = SEBlock(64)
        self.maxpool1 = nn.MaxPool1d(2, 2)
        self.relu1 = nn.ReLU()
        
        self.res_shortcut2 = ConvQuadraticOperation(64, 128, 2, 2, 0)
        self.bn_shortcut2 = nn.BatchNorm1d(128)
        
        self.conv2 = ConvQuadraticOperation(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.se2 = SEBlock(128)
        self.maxpool2 = nn.MaxPool1d(2, 2)
        self.relu2 = nn.ReLU()
        
        # بلوک‌های بیشتر برای عمق/قدرت
        self.res_shortcut3 = ConvQuadraticOperation(128, 128, 2, 2, 0)
        self.bn_shortcut3 = nn.BatchNorm1d(128)
        
        self.conv3 = ConvQuadraticOperation(128, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.se3 = SEBlock(128)
        self.maxpool3 = nn.MaxPool1d(2, 2)
        self.relu3 = nn.ReLU()
        
        self.res_shortcut4 = ConvQuadraticOperation(128, 128, 2, 2, 0)
        self.bn_shortcut4 = nn.BatchNorm1d(128)
        
        self.conv4 = ConvQuadraticOperation(128, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.se4 = SEBlock(128)
        self.maxpool4 = nn.MaxPool1d(2, 2)
        self.relu4 = nn.ReLU()
        
        # pooling سراسری + FC
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)  # 10 کلاس

    def forward(self, x):
        # چندمقیاسی
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        out = torch.cat([s1, s2, s3], dim=1)  # (B, 48, ~512) با فرض کاهش ابعاد با گام 4
        out = self.relu_entry(self.maxpool_entry(self.bn_entry(out)))
        
        # بلوک 1
        shortcut1 = self.bn_shortcut1(self.res_shortcut1(out))
        out1 = self.maxpool1(self.se1(self.bn1(self.conv1(out))))
        out1 = self.relu1(out1 + shortcut1)
        
        # بلوک 2
        shortcut2 = self.bn_shortcut2(self.res_shortcut2(out1))
        out2 = self.maxpool2(self.se2(self.bn2(self.conv2(out1))))
        out2 = self.relu2(out2 + shortcut2)
        
        # بلوک 3
        shortcut3 = self.bn_shortcut3(self.res_shortcut3(out2))
        out3 = self.maxpool3(self.se3(self.bn3(self.conv3(out2))))
        out3 = self.relu3(out3 + shortcut3)
        
        # بلوک 4
        shortcut4 = self.bn_shortcut4(self.res_shortcut4(out3))
        out4 = self.maxpool4(self.se4(self.bn4(self.conv4(out3))))
        out4 = self.relu4(out4 + shortcut4)
        
        # طبقه‌بندی
        out = self.global_pool(out4).view(out4.size(0), -1)
        out = self.relu_fc(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# تست شکل
if __name__ == '__main__':
    model = MSQAResNet()
    x = torch.randn(2, 1, 2048)
    y = model(x)
    print(y.shape)  # باید (2, 10) باشه
    
    