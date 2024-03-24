import random

import hydra
import omegaconf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA

from dataset_transforms import get_transforms

from models.base_discriminator import Discriminator
from models.csp_generator import CSPGenerator as Generator

from models.utils import weights_init

RANDOM_SEED = 42

torch.backends.cudnn.deterministic = True
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    wandb.init(
        project=cfg.wandb.setup.project,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
    )
    print(cfg)

    transform = get_transforms(**cfg["transforms"])
    device = torch.device(cfg["device"])
    dataset = CelebA("data", download=False, transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=cfg["batch_size"])
    generator = Generator(**cfg["models"]["generator"], nc=cfg.models.nc).to(device)
    discriminator = Discriminator(
        **cfg["models"]["discriminator"], nc=cfg.models.nc
    ).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(
        cfg["transforms"]["img_size"],
        cfg["models"]["generator"]["nz"],
        1,
        1,
        device="cuda",
    )

    # Establish convention for real and fake labels during training
    real_label = 1.0
    fake_label = 0.0

    # Setup Adam optimizer_d for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), **cfg["optimizer_d"])
    optimizerG = optim.Adam(generator.parameters(), **cfg["optimizer_g"])
    # Training Loop

    # Lists to keep track of progress
    iters = 0
    device = "cuda"
    print("Starting Training Loop...")

    # For each epoch
    for epoch in range(cfg["num_epochs"]):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(
                b_size, cfg["models"]["generator"]["nz"], 1, 1, device=device
            )
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()

            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D

            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        cfg["num_epochs"],
                        i,
                        len(dataloader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )

            # Save Losses for plotting later
            wandb.log({"g_loss": errG})
            wandb.log({"d_loss": errD})

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 50 == 0) or ((i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                grid = vutils.make_grid(fake, padding=2, normalize=True)
                wandb.log({"generated": [wandb.Image(grid)]})
            iters += 1
    wandb.finish(quiet=True)


if __name__ == "__main__":
    train()
