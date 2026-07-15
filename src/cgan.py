"""
--------------------------------------------------------
Task 2: Conditional Generative Adversarial Network (CGAN)
AI-Powered Text-to-Image Generation System

Author: Manvitha Reddy
Internship Project

Objective:
Implement a Conditional Generative Adversarial Network
that generates basic geometric shapes (Circle and Square)
based on class labels.

Label 0 -> Circle
Label 1 -> Square
--------------------------------------------------------
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# Configuration
# ==========================================

IMG_SIZE = 64

NUM_CLASSES = 2

SAMPLES_PER_CLASS = 1000

LATENT_DIM = 100

BATCH_SIZE = 64

EPOCHS = 20

LEARNING_RATE = 0.0002

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# ==========================================
# Create Required Directories
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/graphs", exist_ok=True)

# ==========================================
# Dataset Generation
# ==========================================

def generate_shape(label):

    image = np.zeros(
        (IMG_SIZE, IMG_SIZE),
        dtype=np.float32
    )

    if label == 0:

        radius = random.randint(10,20)

        center_x = IMG_SIZE//2 + random.randint(-5,5)

        center_y = IMG_SIZE//2 + random.randint(-5,5)

        y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]

        mask = (
            (x-center_x)**2 +
            (y-center_y)**2
        ) <= radius**2

        image[mask] = 1.0

    else:

        side = random.randint(20,30)

        start_x = IMG_SIZE//2-side//2 + random.randint(-5,5)

        start_y = IMG_SIZE//2-side//2 + random.randint(-5,5)

        image[
            start_y:start_y+side,
            start_x:start_x+side
        ] = 1.0

    return image

def create_dataset():

    images = []

    labels = []

    for label in range(NUM_CLASSES):

        for _ in range(SAMPLES_PER_CLASS):

            img = generate_shape(label)

            images.append(img)

            labels.append(label)

    images = np.array(images)[:, None, :, :]

    labels = np.array(labels)

    return images, labels

# ==========================================
# Generator Network
# ==========================================

class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        self.label_embedding = nn.Embedding(
            NUM_CLASSES,
            10
        )

        self.model = nn.Sequential(

            nn.Linear(LATENT_DIM + 10, 256),
            nn.ReLU(True),

            nn.Linear(256, 512),
            nn.ReLU(True),

            nn.Linear(512, 1024),
            nn.ReLU(True),

            nn.Linear(
                1024,
                IMG_SIZE * IMG_SIZE
            ),

            nn.Tanh()

        )

    def forward(self, noise, labels):

        embedded_labels = self.label_embedding(labels)

        x = torch.cat(
            (noise, embedded_labels),
            dim=1
        )

        output = self.model(x)

        output = output.view(
            -1,
            1,
            IMG_SIZE,
            IMG_SIZE
        )

        return output
    


# ==========================================
# Discriminator Network
# ==========================================

class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(

            NUM_CLASSES,

            IMG_SIZE * IMG_SIZE

        )

        self.model = nn.Sequential(

            nn.Linear(
                IMG_SIZE * IMG_SIZE * 2,
                1024
            ),

            nn.LeakyReLU(0.2),

            nn.Linear(1024,512),

            nn.LeakyReLU(0.2),

            nn.Linear(512,256),

            nn.LeakyReLU(0.2),

            nn.Linear(256,1),

            nn.Sigmoid()

        )

    def forward(self,image,labels):

        image=image.view(
            image.size(0),
            -1
        )

        embedded_labels=self.label_embedding(labels)

        x=torch.cat(
            (image,embedded_labels),
            dim=1
        )

        output=self.model(x)

        return output
    



# ==========================================
# DataLoader
# ==========================================

def create_dataloader():

    images, labels = create_dataset()

    images_tensor = torch.tensor(
        images,
        dtype=torch.float32
    )

    labels_tensor = torch.tensor(
        labels,
        dtype=torch.long
    )

    dataset = TensorDataset(
        images_tensor,
        labels_tensor
    )

    dataloader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True

    )

    return dataloader

# ==========================================
# Train CGAN
# ==========================================

def train(generator, discriminator, dataloader):

    criterion = nn.BCELoss()

    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    print("\nStarting Training...\n")

    for epoch in range(EPOCHS):

        for real_images, labels in dataloader:

            batch_size = real_images.size(0)

            real_images = real_images.to(device)
            labels = labels.to(device)

            real = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

# -------------------------
# Train Discriminator
# -------------------------

            optimizer_D.zero_grad()

            real_output = discriminator(
                real_images,
                labels
            )

            d_real_loss = criterion(
                real_output,
                real
            )

            noise = torch.randn(
                batch_size,
                LATENT_DIM
            ).to(device)

            fake_images = generator(
                noise,
                labels
            )

            fake_output = discriminator(
                fake_images.detach(),
                labels
            )

            d_fake_loss = criterion(
                fake_output,
                fake
            )

            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()

            optimizer_D.step()

# -------------------------
# Train Generator
# -------------------------

            optimizer_G.zero_grad()

            noise = torch.randn(
                batch_size,
                LATENT_DIM
            ).to(device)

            generated_images = generator(
                noise,
                labels
            )

            predictions = discriminator(
                generated_images,
                labels
            )

            g_loss = criterion(
                predictions,
                real
            )

            g_loss.backward()

            optimizer_G.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"D Loss: {d_loss.item():.4f} "
            f"G Loss: {g_loss.item():.4f}"
        )

    print("\nTraining Completed!")

 # Save trained models
    torch.save(
        generator.state_dict(),
        "models/generator.pth"
    )

    torch.save(
        discriminator.state_dict(),
        "models/discriminator.pth"
    )

    print("Generator model saved successfully!")

    print("Discriminator model saved successfully!")

    return generator

# ==========================================
# Generate Images
# ==========================================

def generate_images(generator):

    generator.eval()

    noise = torch.randn(
        8,
        LATENT_DIM
    ).to(device)

    labels = torch.tensor(
        [0,0,0,0,1,1,1,1],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():

        images = generator(
            noise,
            labels
        )

    images = images.cpu()

    plt.figure(figsize=(10,5))

    for i in range(8):

        plt.subplot(2,4,i+1)

        plt.imshow(
            images[i][0],
            cmap="gray"
        )

        if labels[i] == 0:
            plt.title("Circle")
        else:
            plt.title("Square")

        plt.axis("off")

    plt.tight_layout()
    plt.savefig(
    "outputs/graphs/generated_cgan_shapes.png",
    dpi=300,
    bbox_inches="tight"
)


    plt.show()

    print("\nGenerated images saved successfully!")

# ==========================================
# Main Function
# ==========================================

def main():

    dataloader = create_dataloader()

    generator = Generator().to(device)

    discriminator = Discriminator().to(device)

    trained_generator = train(

        generator,

        discriminator,

        dataloader

    )

    generate_images(trained_generator)


if __name__ == "__main__":

    main()