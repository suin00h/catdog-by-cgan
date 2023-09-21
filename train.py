import torch
import torch.optim as optim
from torch import Tensor

from models.generator import Generator
from models.discriminator import Discriminator


def run():
    gen_model = Generator()
    gen_optimizer = optim.Adam()
    trainEpoch()

def trainEpoch(
    num_batch: int,
    num_channel: int
):
    input_noise = getInputNoise(num_batch, num_channel)

def getInputNoise(
    num_batch: int,
    num_channel: int
) -> Tensor:
    return torch.randn((num_batch, num_channel, 1, 1))

def main():
    #argparse and config settings
    run()

if __name__ == "__main__":
    main()
