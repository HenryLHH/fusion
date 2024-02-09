import torch
from torch import nn
from torch.nn.utils import spectral_norm
from PIL import Image
from torchvision import transforms

def swish(x):
    return x * x.sigmoid()


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, downsample=False):
        super().__init__()

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, mid_channels, 3,
                                             stride=2 if downsample else 1, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(mid_channels, mid_channels, 3, padding=1))
        if downsample:
            self.avg_pool = nn.AvgPool2d(2, ceil_mode=True)
        if in_channels != mid_channels:
            self.conv1x1 = spectral_norm(nn.Conv2d(in_channels, mid_channels, 1))

        self.in_out_match = (in_channels == mid_channels)
        self.downsample = downsample

    def forward(self, x):
        h = swish(self.conv1(x))
        h = swish(self.conv2(h))
        if self.downsample:
            x = self.avg_pool(x)
        if not self.in_out_match:
            x = self.conv1x1(x)

        return swish(h + x)


class MnistEnergyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 128, 3, padding=1)),
            ResBlock(128, 128, True),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 256, True),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256, True),
            ResBlock(256, 256),
            ResBlock(256, 256)
        )
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((28, 28)), transforms.ToTensor()])

    def forward(self, x):
        x_trans = torch.cat([self.transform(xi[:3, :, :]).unsqueeze(0) for xi in x], dim=0).cuda()
        h = self.resnet(x_trans).view(-1, 256 * 4 * 4)
        energy = h.sum(dim=1)
        return energy

class MnistCondEnergyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 28*28)
        self.resnet = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 128, 3, padding=1)),
            ResBlock(128, 128, True),
            ResBlock(128, 128),
            ResBlock(128, 256, True),
            ResBlock(256, 256),
            ResBlock(256, 256, True),
            ResBlock(256, 256)
        )
        self.fc = spectral_norm(nn.Linear(256 * 4 * 4, 1))

    def forward(self, x, y):
        y_embed = self.embed(y).view(-1, 1, 28, 28)
        h = torch.cat((x, y_embed), dim=1)
        h = self.resnet(h).view(-1, 256 * 4 * 4)
        energy = self.fc(h)
        return energy

if __name__ == '__main__': 
    model = MnistEnergyNN()
    data = torch.rand(1, 5, 84, 84)
    print(model(data))