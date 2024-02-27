import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# The original paper used MLP, but here we use CNN instead.
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1) # Output 2 categories
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)
    
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7*7*64)
        self.ct1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.ct2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2)
        self.conv = nn.Conv2d(16, 1, kernel_size=7)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = x.view(-1, 64, 7, 7)

        x = F.relu(self.ct1(x))
        x = F.relu(self.ct2(x))

        return self.conv(x)
    
class GAN(nn.Module):
    def __init__(self, latent_dim=100, lr=3e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.lr = lr

        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
