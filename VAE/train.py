import os
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from tqdm import tqdm
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim import Adam
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 20
BATCH_SIZE = 64
NUM_WORKERS = 0
LR = 3e-4 # Best lr for Adam, Karpathy constant :)

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

model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)
bce_loss = nn.BCELoss(reduction="sum")

for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        optimizer.zero_grad()

        x = x.to(DEVICE).view(BATCH_SIZE, -1)
        x_reconstructed, mu, sigma = model(x)

        reconstructed_loss = bce_loss(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) / 2
        loss = reconstructed_loss + kl_div

        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

def prepare_encodings(model, dataset):
    """
    Prepares and returns encodings (mu, sigma) for one example of each digit (0-9).
    """
    images_per_digit = {i: None for i in range(10)}
    encodings_digit = {}

    for x, y in dataset:
        if all(value is not None for value in images_per_digit.values()):
            break  # Stop if we've found examples for all digits
        if images_per_digit[y] is None:
            images_per_digit[y] = x.view(1, 784)
    
    with torch.no_grad():
        for digit, img in images_per_digit.items():
            mu, sigma = model.encoder(img.to(DEVICE))
            encodings_digit[digit] = (mu, sigma)
    
    return encodings_digit

def inference(model, encodings_digit, digit, num_examples=1):
    """
    Generates (num_examples) images of a particular digit using precomputed encodings.
    """
    os.makedirs('./pictures', exist_ok=True)
    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decoder(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"pictures/generated_{digit}_ex{example}.png")

encodings_digit = prepare_encodings(model, data_module.train_dataset)

for idx in range(10):
    inference(model, encodings_digit, idx, num_examples=5)
        