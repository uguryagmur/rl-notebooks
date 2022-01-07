import glob
from numpy import generic
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


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
                self.latent_space_size, self.feature_map_size * 16, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(self.feature_map_size * 16),
            nn.LeakyReLU(),
            # state size (batch_size, feature_map_size * 16, 4, 4)
            nn.ConvTranspose2d(
                self.feature_map_size * 16,
                self.feature_map_size * 8,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.feature_map_size * 8),
            nn.LeakyReLU(),
            # state size (batch_size, feature_map_size * 8, 8, 8)
            nn.ConvTranspose2d(
                self.feature_map_size * 8,
                self.feature_map_size * 4,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.feature_map_size * 4),
            nn.LeakyReLU(),
            # state size (batch_size, feature_map_size * 4, 16, 16)
            nn.ConvTranspose2d(
                self.feature_map_size * 4,
                self.feature_map_size * 2,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.feature_map_size * 2),
            nn.LeakyReLU(),
            # state size (batch_size, feature_map_size * 2, 32, 32)
            nn.ConvTranspose2d(
                self.feature_map_size * 2, self.feature_map_size, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.feature_map_size),
            nn.LeakyReLU(),
            # state size (batch_size, feature_map_size, 64, 64)
            nn.ConvTranspose2d(
                self.feature_map_size, self.color_channel_size, 4, 2, 1, bias=False
            ),
            nn.Tanh(),
            # output size (batch_size, color_channel_size, 128, 128)
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
            # input size (batch_size, color_channel_size, 128, 128)
            nn.Conv2d(
                self.color_channel_size, self.feature_map_size, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(self.feature_map_size),
            nn.LeakyReLU(),
            # state size (batch_size, feature_map_size, 64, 64)
            nn.Conv2d(
                self.feature_map_size, self.feature_map_size * 2, 4, 2, 1, bias=False,
            ),
            nn.BatchNorm2d(self.feature_map_size * 2),
            nn.LeakyReLU(),
            # state size (batch_size, feature_map_size * 2, 32, 32)
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
            # state size (batch_size, feature_map_size * 4, 16, 16)
            nn.Conv2d(
                self.feature_map_size * 4,
                self.feature_map_size * 8,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.feature_map_size * 8),
            nn.LeakyReLU(),
            # state size (batch_size, feature_map_size * 8, 8, 8)
            nn.Conv2d(
                self.feature_map_size * 8,
                self.feature_map_size * 16,
                4,
                2,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.feature_map_size * 16),
            nn.LeakyReLU(),
            # state size (batch_size, feature_map_size * 16, 4, 4)
            nn.Conv2d(self.feature_map_size * 16, 1, 4, 1, 0, bias=False),
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
        self.dataset = FFHQDataset(self.dataset_path)

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
                target_real = [self.real_label] * data.size(0)
                target_fake = [self.fake_label] * self.batch_size
                target = target_real + target_fake
                target = torch.FloatTensor(target)

                disc_optimizer.zero_grad()
                noise = torch.rand(size=(self.batch_size, gen.latent_space_size, 1, 1))
                generated: torch.Tensor = gen(noise)
                disc_input = torch.cat((data.to(gen.device), generated))
                rand_perm = torch.randperm(self.batch_size + data.size(0))
                disc_output = disc(disc_input[rand_perm].detach()).view(-1)

                disc_loss: torch.Tensor = criterion(
                    disc_output, target[rand_perm].to(disc.device)
                )
                disc_loss.backward(disc_loss)
                disc_optimizer.step()

                gen_optimizer.zero_grad()
                generated: torch.Tensor = gen(noise)
                disc_output = disc(generated).view(-1)
                gen_loss: torch.Tensor = criterion(
                    disc_output,
                    torch.FloatTensor([self.real_label] * self.batch_size).to(
                        disc.device
                    ),
                )
                gen_loss.backward(gen_loss)
                gen_optimizer.step()

                total_disc_loss += disc_loss.mean().item()
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
        plt.plot(loss_hist_d)
        plt.plot(loss_hist_g)
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
            fake = fake.transpose(0, 1).transpose(1, 2).numpy()
            plt.imshow(fake * 255)
            plt.show()
            breakpoint()


def main():
    generator = Generator()
    discriminator = Discriminator()
    trainer = GANTrainer()
    trainer.train(generator, discriminator, 80)
    demo_generator(generator)


if __name__ == "__main__":
    main()
