import torch
import torch.nn as nn
import torch.nn.functional as F

# Input img -> Encoder -> mean, std -> Reparametrization trick -> Decoder -> Output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()

        # encoder
        self.img2hid = nn.Linear(input_dim, h_dim)
        self.hid2mu = nn.Linear(h_dim, z_dim)
        self.hid2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z2hid = nn.Linear(z_dim, h_dim)
        self.hid2img = nn.Linear(h_dim, input_dim)

    # q_{\phi}(z|x)
    def encoder(self, x):
        h = F.relu(self.img2hid(x))
        mu, sigma = self.hid2mu(h), self.hid2sigma(h)
        return mu, sigma

    # p_{\theta}(x|z)
    def decoder(self, z):
        h = F.relu(self.z2hid(z))
        return torch.sigmoid(self.hid2img(h)) # Each MNIST pixel is normalized, so add a sigmoid to ensure values are between (0, 1).

    def forward(self, x):
        mu, sigma = self.encoder(x)
        epslion = torch.rand_like(sigma)
        z_reparametrized = mu + sigma * epslion
        x_reconstructed = self.decoder(z_reparametrized)
        return x_reconstructed, mu, sigma

if __name__ == "__main__":
    # Test the shape of output
    x = torch.randn(4, 28*28) # Assume the batchsize is 4
    vae = VariationalAutoEncoder(input_dim=784)
    print([vae(x)[i].shape for i in range(3)])