import os
import torch
import torchvision.transforms as transforms

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from model import GAN

random_seed = 42
torch.manual_seed(random_seed)

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0

class MnistDataModule:
    def __init__(self, data_dir='./data', batch_size=32, num_workers=0):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ]
        )

        self.train_dataset = MNIST(data_dir, train=True, download=True, transform=self.transform)

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

data_module = MnistDataModule(data_dir='./data', batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
train_loader = data_module.get_train_dataloader()
    
def train_gan(model, dataloader, epochs=30):
    optimizer_g = Adam(model.generator.parameters(), lr=model.lr)
    optimizer_d = Adam(model.discriminator.parameters(), lr=model.lr)

    for epoch in range(epochs):
        for batch_idx, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(DEVICE)
            z = torch.randn(real_imgs.size(0), model.latent_dim).to(DEVICE)

            # Discriminator: minimize -log(D(x))-log(1-D(G(z))) <-> maximize log(D(x))+log(1-D(G(z)))
            optimizer_d.zero_grad()
            fake_imgs = model(z)
            y_hat_real = model.discriminator(real_imgs)
            y_real = torch.ones(real_imgs.size(0), 1).to(DEVICE)
            real_loss = model.adversarial_loss(y_hat_real, y_real)

            y_hat_fake = model.discriminator(fake_imgs)
            y_fake = torch.zeros(real_imgs.size(0), 1).to(DEVICE)
            fake_loss = model.adversarial_loss(y_hat_fake, y_fake)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward(retain_graph=True)
            optimizer_d.step()

            # Generator: minimize -log(D(G(z))) <-> maximize log(D(G(z)))
            optimizer_g.zero_grad()
            y_hat = model.discriminator(fake_imgs)
            y = torch.ones(real_imgs.size(0), 1).to(DEVICE)
            g_loss = model.adversarial_loss(y_hat, y)
            g_loss.backward()
            optimizer_g.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, G_loss: {g_loss.item()}, D_loss: {d_loss.item()}")

def generate_images_with_gan(model, num_examples=1):

    os.makedirs('./pictures', exist_ok=True)

    for i in range(num_examples):
        z = torch.randn(1, model.latent_dim).to(DEVICE)  # Generate random noise as input for the generator
        with torch.no_grad():
            generated_img = model.generator(z)  # Generate an image from the random noise
            generated_img = generated_img.view(-1, 1, 28, 28)  # Reshape to the MNIST image size
            save_image(generated_img, f"./pictures/generated_img_{i}.png") 

model = GAN().to(DEVICE)
train_gan(model, train_loader)
generate_images_with_gan(model, num_examples=10)