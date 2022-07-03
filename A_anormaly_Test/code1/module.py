import torch.nn.functional as F
import torch.nn as nn
import torch


# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=8):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)
#

class Encoder(nn.Module):
    """
        The structure of encoder (The front part of CycleGAN generator)
    """
    def __init__(self, in_channels, nef = 64, n_downsampling = 4):
        """
            Constructor

            Arg:    in_channels     (Int)   - The input channel
                    nef             (Int)   - The number of base filter
                    downsampling    (Int)   - The number to do the down-sampling
        """
        super().__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, nef, kernel_size=7, padding=0,
                           bias=False),
                 nn.BatchNorm2d(nef),
                 nn.ReLU(True)]
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(nef * mult, nef * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(nef * mult * 2),
                      nn.ReLU(True)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
            forward process

            Arg:    input   (torch.Tensor)  - The tensor you want to deal with
            Ret:    The result tensor
        """
        output = self.model(input)
        return output


class Decoder(nn.Module):
    """
        The structure of encoder (The back part of CycleGAN generator, without residual block)
    """
    def __init__(self, out_channels, ndf=64, n_downsampling=4):
        """
            Constructor

            Arg:    out_channels    (Int)   - The output channel
                    ndf             (Int)   - The number of base filter
                    downsampling    (Int)   - The number to do the down-sampling
        """
        super().__init__()
        model = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ndf * mult, int(ndf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=False),
                      nn.BatchNorm2d(int(ndf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ndf, out_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
            forward process

            Arg:    input   (torch.Tensor)  - The tensor you want to deal with
            Ret:    The result tensor
        """
        return self.model(input)


class Discriminator(nn.Module):
    """
        The structure of encoder (70x70 PatchGAN)
    """
    def __init__(self, in_channels, out_channels = 256, nef=32, last = None):
        """
            Constructor

            Arg:    in_channels     (Int)       - The input channel
                    out_channels    (Int)       - The output channel
                    nef             (Int)       - The number of base filter
                    last            (nn.Module) - The function instance of last activation function
        """
        super().__init__()
        self.model = [
            # stride=2
            nn.Conv2d(in_channels, nef, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            # stride=2
            nn.Conv2d(nef * 1, nef * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, True),
            # stride=2
            nn.Conv2d(nef * 2, nef * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, True),
            # stride=1
            nn.Conv2d(nef * 4, nef * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nef * 8, out_channels, kernel_size=4, stride=1, padding=1)
        ]
        if last is not None:
            self.model += [last()]
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        """
            forward process

            Arg:    input   (torch.Tensor)  - The tensor you want to deal with
            Ret:    The prediction tensor
        """
        return self.model(input)