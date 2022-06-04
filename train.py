import torch
from tqdm import tqdm

from src.utils import Loss, Dataset, show_tensor_images
from src.blocks.discriminator import Discriminator
from src.blocks.generator import Generator


def train_srresnet(srresnet,
                   dataloader,
                   device,
                   lr=1e-4,
                   total_steps=1e6,
                   display_step=500):
    srresnet = srresnet.to(device).train()
    optimizer = torch.optim.Adam(srresnet.parameters(), lr=lr)

    cur_step = 0
    mean_loss = 0.0
    while cur_step < total_steps:
        for hr_real, lr_real in tqdm(dataloader, position=0):
            hr_real = hr_real.to(device)
            lr_real = lr_real.to(device)

            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                hr_fake = srresnet(lr_real)
                loss = Loss.img_loss(hr_real, hr_fake)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss += loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print('Step {}: SRResNet loss: {:.5f}'.format(
                    cur_step, mean_loss))
                # show_tensor_images(lr_real * 2 - 1)
                # show_tensor_images(hr_fake.to(hr_real.dtype))
                # show_tensor_images(hr_real)
                mean_loss = 0.0

            cur_step += 1
            if cur_step == total_steps:
                break


def train_srgan(generator,
                discriminator,
                dataloader,
                device,
                lr=1e-4,
                total_steps=2e5,
                display_step=500):
    generator = generator.to(device).train()
    discriminator = discriminator.to(device).train()
    loss_fn = Loss(device=device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lambda _: 0.1)
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lambda _: 0.1)

    lr_step = total_steps // 2
    cur_step = 0

    mean_g_loss = 0.0
    mean_d_loss = 0.0

    while cur_step < total_steps:
        for hr_real, lr_real in tqdm(dataloader, position=0):
            hr_real = hr_real.to(device)
            lr_real = lr_real.to(device)

            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                g_loss, d_loss, hr_fake = loss_fn(
                    generator,
                    discriminator,
                    hr_real,
                    lr_real,
                )

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            mean_g_loss += g_loss.item() / display_step
            mean_d_loss += d_loss.item() / display_step

            if cur_step == lr_step:
                g_scheduler.step()
                d_scheduler.step()
                print('Decayed learning rate by 10x.')

            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    'Step {}: Generator loss: {:.5f}, Discriminator loss: {:.5f}'
                    .format(cur_step, mean_g_loss, mean_d_loss))
                # show_tensor_images(lr_real * 2 - 1)
                # show_tensor_images(hr_fake.to(hr_real.dtype))
                # show_tensor_images(hr_real)
                mean_g_loss = 0.0
                mean_d_loss = 0.0

            cur_step += 1
            if cur_step == total_steps:
                break


device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = Generator(n_res_blocks=16, n_ps_blocks=2)

dataloader = torch.utils.data.DataLoader(
    Dataset('data',
            'train',
            download=False,
            hr_size=[96, 96],
            lr_size=[24, 24]),
    batch_size=16,
    pin_memory=True,
    shuffle=True,
)

train_srresnet(generator,
               dataloader,
               device,
               lr=1e-4,
               total_steps=1e5,
               display_step=1000)
torch.save(generator, 'srresnet.pt')

generator = torch.load('srresnet.pt', map_location=torch.device(device))
discriminator = Discriminator(n_blocks=1, base_channels=8)

train_srgan(generator,
            discriminator,
            dataloader,
            device,
            lr=1e-4,
            total_steps=2e5,
            display_step=1000)

torch.save(generator, 'srgenerator.pt')
torch.save(discriminator, 'srdiscriminator.pt')