import glob
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToTensor
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(0)


class FFHQDataset(Dataset):
    to_tensor = transforms.ToTensor()

    def __init__(self, dataset_dir: str):
        super(FFHQDataset, self).__init__()
        self.img_path_list = glob.glob(dataset_dir + "/*.png")
        print("FFHQ Dataset with {} of elements is ready".format(self.__len__()))

    def __len__(self) -> int:
        return len(self.img_path_list)

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path = self.img_path_list[index]
        img = Image.open(img_path)
        return self.to_tensor(img).float()

    def get_loader(self, batch_size: int, workers: int, shuffle: bool = True):
        return DataLoader(
            self, batch_size=batch_size, num_workers=workers, shuffle=shuffle
        )


class CIFAR100Dataset:
    def __init__(self) -> None:
        self.dataset = datasets.CIFAR100(
            root="~/Datasets/torchvision", download=True, transform=ToTensor()
        )

    def get_loader(self, batch_size: int, workers: int, shuffle: bool = True):
        return DataLoader(
            self.dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle,
        )


class Generator(nn.Module):
    color_channel_size = 3
    feature_map_size = 32
    latent_space_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        super(Generator, self).__init__()
        self.net = self.configure_network()
        self.initialize_weights()
        self.to(self.device)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor.to(self.device))

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1, 0.02)

    def configure_network(self):
        return nn.Sequential(
            # input size (batch_size, latent_space_size, 1, 1)
            nn.ConvTranspose2d(
                self.latent_space_size, self.feature_map_size * 4, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(self.feature_map_size * 4),
            nn.LeakyReLU(negative_slope=0.02),
            # state size (batch_size, feature_map_size * 16, 4, 4)
            nn.ConvTranspose2d(
                self.feature_map_size * 4,
                self.feature_map_size * 2,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.feature_map_size * 2),
            nn.LeakyReLU(negative_slope=0.02),
            # state size (batch_size, feature_map_size * 8, 8, 8)
            nn.ConvTranspose2d(
                self.feature_map_size * 2, self.feature_map_size, 4, 2, 1, bias=False,
            ),
            nn.BatchNorm2d(self.feature_map_size),
            nn.LeakyReLU(negative_slope=0.02),
            # state size (batch_size, feature_map_size * 4, 16, 16)
            nn.ConvTranspose2d(
                self.feature_map_size, self.color_channel_size, 4, 2, 1, bias=False,
            ),
            # state size (batch_size, feature_map_size * 2, 32, 32)
            nn.Sigmoid(),
        )


class Discriminator(nn.Module):
    color_channel_size = 3
    feature_map_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = self.configure_network()
        self.initialize_weights()
        self.to(self.device)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor.to(self.device))

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1, 0.02)

    def configure_network(self):
        return nn.Sequential(
            # input size (batch_size, color_channel_size, 32, 32)
            nn.Conv2d(
                self.color_channel_size, self.feature_map_size, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.feature_map_size),
            nn.LeakyReLU(),
            # state size (batch_size, feature_map_size, 16, 16)
            nn.Conv2d(
                self.feature_map_size, self.feature_map_size * 2, 4, 2, 1, bias=False,
            ),
            nn.BatchNorm2d(self.feature_map_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            # state size (batch_size, feature_map_size * 2, 8, 8)
            nn.Conv2d(
                self.feature_map_size * 2,
                self.feature_map_size * 4,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.feature_map_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            # state size (batch_size, feature_map_size * 4, 4, 4)
            nn.Conv2d(self.feature_map_size * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # output size (batch_size, 1, 1, 1)
        )


class GANTrainer:
    dataset_path = "/home/adm1n/Datasets/FFHQ/thumbnails"
    workers = 4
    batch_size = 64
    image_size = 128
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    fake_label = 0
    real_label = 1
    writer = SummaryWriter("/home/adm1n/Shop/logs")

    def __init__(self):
        self.dataset = CIFAR100Dataset()

    def train(self, gen: Generator, disc: Discriminator, num_epochs: int = 100):
        data_loader = self.dataset.get_loader(
            batch_size=self.batch_size, workers=self.workers
        )
        disc_optimizer = optim.Adam(
            disc.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        gen_optimizer = optim.Adam(
            gen.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        criterion = nn.BCELoss(reduction="none")
        epoch_progress_bar = tqdm.tqdm(range(1, num_epochs + 1))
        loss_history: List[Tuple[float, float]] = list()

        for epoch in epoch_progress_bar:
            total_disc_loss = 0.0
            total_gen_loss = 0.0
            batch_progress_bar = tqdm.tqdm(data_loader)
            for batch_iter, data in enumerate(batch_progress_bar):
                data = data[0]
                disc_optimizer.zero_grad()
                gen_optimizer.zero_grad()
                noise = torch.rand(size=(self.batch_size, gen.latent_space_size, 1, 1))
                with torch.no_grad():
                    generated: torch.Tensor = gen(noise)

                target_real = [self.real_label] * data.size(0)
                target_fake = [self.fake_label] * self.batch_size

                disc_fake = disc(generated.to(disc.device)).view(-1)
                target = torch.FloatTensor(target_fake).to(disc.device)
                disc_fake_loss = criterion(disc_fake, target)
                disc_fake_loss.backward(disc_fake_loss)

                disc_real = disc(data.to(disc.device)).view(-1)
                target = torch.FloatTensor(target_real).to(disc.device)
                disc_real_loss = criterion(disc_real, target)
                disc_real_loss.backward(disc_real_loss)
                disc_optimizer.step()

                generated: torch.Tensor = gen(noise)
                disc_fake = disc(generated.to(disc.device)).view(-1)
                target = torch.FloatTensor([self.real_label] * self.batch_size)
                gen_loss = criterion(disc_fake, target.to(disc.device))
                gen_loss.backward(gen_loss)
                gen_optimizer.step()

                total_disc_loss += (
                    disc_fake_loss.mean().item() + disc_real_loss.mean().item()
                )
                total_gen_loss += gen_loss.mean().item()
                batch_progress_bar.set_postfix(
                    {
                        "Disc Loss": "{:2.6f}".format(
                            total_disc_loss / (batch_iter + 1)
                        ),
                        "Gen Loss": "{:2.6f}".format(total_gen_loss / (batch_iter + 1)),
                    }
                )
                self.writer.add_scalar(
                    "Discriminator Batch Loss",
                    total_disc_loss / (batch_iter + 1),
                    batch_iter,
                )
                self.writer.add_scalar(
                    "Generator Batch Loss",
                    total_gen_loss / (batch_iter + 1),
                    batch_iter,
                )

            loss_history.append(
                (total_disc_loss / (batch_iter + 1), total_gen_loss / (batch_iter + 1))
            )
            epoch_progress_bar.set_postfix(
                {
                    "Disc Loss": "{:2.6f}".format(total_disc_loss / (batch_iter + 1)),
                    "Gen Loss": "{:2.6f}".format(total_gen_loss / (batch_iter + 1)),
                }
            )
            self.writer.add_scalar(
                "Discriminator Epoch Loss", total_disc_loss / (batch_iter + 1), epoch
            )
            self.writer.add_scalar(
                "Generator Epoch Loss", total_gen_loss / (batch_iter + 1), epoch
            )

        # plotting losses
        loss_hist_d = []
        loss_hist_g = []
        for elem in loss_history:
            loss_hist_d.append(elem[0])
            loss_hist_g.append(elem[1])
        plt.plot(loss_hist_d, label="disc")
        plt.plot(loss_hist_g, label="gen")
        plt.legend()
        plt.savefig("GAN_loss_graph.png")

        # checkpoint for continuing to the training
        torch.save(gen.state_dict(), "GAN_g.pth")
        torch.save(disc.state_dict(), "GAN_d.pth")
        torch.save(gen_optimizer.state_dict(), "GAN_g_opt.pth")
        torch.save(disc_optimizer.state_dict(), "GAN_d_opt.pth")


def demo_generator(gen: Generator):
    with torch.no_grad():
        while True:
            noise = torch.rand(size=(1, gen.latent_space_size, 1, 1))
            fake: torch.Tensor = gen(noise).cpu().squeeze(dim=0)
            fake = (fake * 255).transpose(0, 1).transpose(1, 2).long().numpy()
            plt.imshow(fake)
            plt.show()
            breakpoint()


def main():
    generator = Generator()
    discriminator = Discriminator()
    trainer = GANTrainer()
    trainer.train(generator, discriminator, 60)
    demo_generator(generator)


if __name__ == "__main__":
    main()
