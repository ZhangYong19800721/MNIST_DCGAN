import torch
import torch.nn as nn
import tools


class ResidualBlock(nn.Module):
    def __init__(self, interChannel):
        super(ResidualBlock, self).__init__()
        self.interChannel = interChannel
        self.L01_Conv2d = nn.Conv2d(self.interChannel, self.interChannel, 3, padding=1, bias=True)
        self.L02_BatchNorm = nn.BatchNorm2d(self.interChannel)
        self.L03_PReLU = nn.PReLU(self.interChannel)
        self.L04_Conv2d = nn.Conv2d(self.interChannel, self.interChannel, 3, padding=1, bias=True)
        self.L05_BatchNorm = nn.BatchNorm2d(self.interChannel)

    def forward(self, x):
        y = self.L01_Conv2d(x)
        y = self.L02_BatchNorm(y)
        y = self.L03_PReLU(y)
        y = self.L04_Conv2d(y)
        y = self.L05_BatchNorm(y)
        y = y + x
        return y


class ResidualBlocks4(nn.Module):
    def __init__(self, interChannel):
        super(ResidualBlocks4, self).__init__()
        self.interChannel = interChannel
        self.L01_ResidualBlock = ResidualBlock(self.interChannel)
        self.L02_ResidualBlock = ResidualBlock(self.interChannel)
        self.L03_ResidualBlock = ResidualBlock(self.interChannel)
        self.L04_ResidualBlock = ResidualBlock(self.interChannel)

    def forward(self, x):
        y = self.L01_ResidualBlock(x)
        y = self.L02_ResidualBlock(y)
        y = self.L03_ResidualBlock(y)
        y = self.L04_ResidualBlock(y)
        return y


class ResidualBlocks8(nn.Module):
    def __init__(self, interChannel):
        super(ResidualBlocks8, self).__init__()
        self.interChannel = interChannel
        self.L01_ResidualBlocks4 = ResidualBlocks4(self.interChannel)
        self.L02_ResidualBlocks4 = ResidualBlocks4(self.interChannel)

    def forward(self, x):
        y = self.L01_ResidualBlocks4(x)
        y = self.L02_ResidualBlocks4(y)
        return y


class ResidualBlocks16(nn.Module):
    def __init__(self, interChannel):
        super(ResidualBlocks16, self).__init__()
        self.interChannel = interChannel
        self.L01_ResidualBlocks8 = ResidualBlocks8(self.interChannel)
        self.L02_ResidualBlocks8 = ResidualBlocks8(self.interChannel)

    def forward(self, x):
        y = self.L01_ResidualBlocks8(x)
        y = self.L02_ResidualBlocks8(y)
        return y


class PatchFeatureExtractor16(nn.Module):
    def __init__(self, inChannel=3, interChannel=64, outChannel=128, patchSize=32):
        super(PatchFeatureExtractor16, self).__init__()
        self.L01_Conv2d = nn.Conv2d(inChannel, interChannel, 3, padding=1, bias=True)
        self.L02_BatchNorm = nn.BatchNorm2d(interChannel)
        self.L03_PReLU = nn.PReLU(interChannel)
        self.L04_ResidualBlocks = ResidualBlocks16(interChannel)
        self.L05_Conv2d = nn.Conv2d(interChannel, outChannel, patchSize, stride=1, padding=0, bias=True)
        self.L06_BatchNorm = nn.BatchNorm2d(outChannel)
        self.L07_PReLU = nn.PReLU(outChannel)

    def forward(self, x):
        y = self.L01_Conv2d(x)
        y = self.L02_BatchNorm(y)
        y = self.L03_PReLU(y)
        y = self.L04_ResidualBlocks(y)
        y = self.L05_Conv2d(y)
        y = self.L06_BatchNorm(y)
        y = self.L07_PReLU(y)
        return y


class PatchFeatureExtractor8(nn.Module):
    def __init__(self, inChannel=3, interChannel=64, outChannel=128, patchSize=32):
        super(PatchFeatureExtractor8, self).__init__()
        self.L01_Conv2d = nn.Conv2d(inChannel, interChannel, 3, padding=1, bias=True)
        self.L02_BatchNorm = nn.BatchNorm2d(interChannel)
        self.L03_PReLU = nn.PReLU(interChannel)
        self.L04_ResidualBlocks = ResidualBlocks8(interChannel)
        self.L05_Conv2d = nn.Conv2d(interChannel, outChannel, patchSize, stride=1, padding=0, bias=True)
        self.L06_BatchNorm = nn.BatchNorm2d(outChannel)
        self.L07_PReLU = nn.PReLU(outChannel)

    def forward(self, x):
        y = self.L01_Conv2d(x)
        y = self.L02_BatchNorm(y)
        y = self.L03_PReLU(y)
        y = self.L04_ResidualBlocks(y)
        y = self.L05_Conv2d(y)
        y = self.L05_BatchNorm(y)
        return z


class PatchFeatureExtractor4(nn.Module):
    def __init__(self, inChannel=3, interChannel=64, outChannel=128, patchSize=32):
        super(PatchFeatureExtractor4, self).__init__()
        self.L01_Conv2d = nn.Conv2d(inChannel, interChannel, 3, padding=1, bias=True)
        self.L02_BatchNorm = nn.BatchNorm2d(interChannel)
        self.L03_PReLU = nn.PReLU(interChannel)
        self.L04_ResidualBlocks = ResidualBlocks4(interChannel)
        self.L05_Conv2d = nn.Conv2d(interChannel, outChannel, patchSize, stride=1, padding=0, bias=True)
        self.L06_BatchNorm = nn.BatchNorm2d(outChannel)
        self.L07_PReLU = nn.PReLU(outChannel)

    def forward(self, x):
        y = self.L01_Conv2d(x)
        y = self.L02_BatchNorm(y)
        y = self.L03_PReLU(y)
        y = self.L04_ResidualBlocks(y)
        y = self.L05_Conv2d(y)
        y = self.L05_BatchNorm(y)
        return z


class Generator(nn.Module):
    def __init__(self, nz=100):
        super(Generator, self).__init__()
        self.L01_Sequential = nn.Sequential(nn.ConvTranspose2d(nz, 512, kernel_size=4, stride=1, padding=0),
                                            nn.BatchNorm2d(512),
                                            nn.PReLU(512),
                                            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.PReLU(128),
                                            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
                                            nn.Tanh(),
                                            )

    # the x is low resolution images minibatch
    def forward(self, x):
        y = self.L01_Sequential(x)
        return y


class Discriminator_SP(nn.Module):
    def __init__(self, inChannel=1):
        super(Discriminator_SP, self).__init__()

        self.FeatureExtractor = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=inChannel, out_channels=32, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        self.Full_Sequential = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(2048, 1024)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(512, 1)),
        )

    def forward(self, imageHR):
        y = self.FeatureExtractor(imageHR)
        y = self.Full_Sequential(y)
        return y


class Discriminator_GP(nn.Module):
    def __init__(self, inChannel=1):
        super(Discriminator_GP, self).__init__()

        self.FeatureExtractor = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        self.Full_Sequential = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, imageHR):
        y = self.FeatureExtractor(imageHR)
        y = self.Full_Sequential(y)
        return y


class Discriminator(nn.Module):
    def __init__(self, inChannel=1):
        super(Discriminator, self).__init__()

        self.L01_Seq = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.L01_Seq(x)
        return y


if __name__ == '__main__':
    G = Generator()
    D = Discriminator()
    noise = torch.randn((2, 100, 1, 1))
    with torch.no_grad():
        z = G(noise)
        p = D(z)
    print(z.shape)
    print(p.shape)
