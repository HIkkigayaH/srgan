import torch
from matplotlib import pyplot as plt

from src.utils import Dataset, show_fig

device = 'cpu'

# load the model
generator = torch.load('./round-2/srgenerator.pt',
                       map_location=torch.device(device))

# put it into test mode
generator = generator.to(device).eval()


def test(dataloader, num=5):
    for hr_real, lr_real in dataloader:
        if num == 0:
            break

        # generate fake image
        hr_fake = generator(lr_real)

        # normalize
        hr_fake = (hr_fake + 1) / 2
        hr_real = (hr_real + 1) / 2

        # detach the tensor
        hr_fake = hr_fake.detach().cpu()

        # plot the images
        show_fig(lr_real, hr_fake, hr_real)

        num -= 1


dataloader = torch.utils.data.DataLoader(
    Dataset('data', 'test', download=False, hr_size=[96, 96], lr_size=[24,
                                                                       24]),
    batch_size=1,
    pin_memory=True,
    shuffle=True,
)

test(dataloader, 1)
