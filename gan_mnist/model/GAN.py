import torch
import torch.nn as nn

# Generator network (DCGAN-style): maps a noise vector to a 28x28 image via transposed convolutions
class Generator(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(input_dim,256,7,1,0,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 7 x 7
            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 14 x 14
            nn.ConvTranspose2d(128,1,4,2,1,bias=False),
            nn.Tanh(),
            # 1 x 28 x 28
        )

    # function for forward propogation
    def forward(self,x):
        x = x.view(-1,self.input_dim,1,1)
        return self.net(x)

#Discriminator network (DCGAN-style): classifies a 28x28 image as real or fake via convolutions
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,64,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout2d(0.4),
            # 64 x 14 x 14
            nn.Conv2d(64,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout2d(0.4),
            # 128 x 7 x 7
            nn.Conv2d(128,1,7,1,0,bias=False),
            nn.Sigmoid(),
            # 1 x 1 x 1
        )

    # function for forward propogation
    def forward(self,x):
        return self.net(x).view(-1)

# standard DCGAN weight init: helps training converge instead of stalling
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)
