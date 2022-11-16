import os
from tqdm import tqdm
from nets import Generator, Discriminator
import torch
import torch.nn as nn
from dataset import dataloader
import scipy.io as io

from config import Config

opt = Config()


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_g = Generator()
    net_d = Discriminator()
    map_location = lambda storage, loc: storage
    if opt.netg_path:
        net_g.load_state_dict(torch.load(f=opt.netg_path, map_location=map_location))
    if opt.netd_path:
        net_d.load_state_dict(torch.load(f=opt.netd_path, map_location=map_location))

    net_g.to(device)
    net_d.to(device)
    g_optimizer = torch.optim.Adam(net_g.parameters(), lr=opt.g_learning_rate, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(net_d.parameters(), lr=opt.d_learning_rate, betas=(0.5, 0.999))
    criterion = nn.BCELoss().to(device)

    true_labels = torch.ones(opt.batch_size).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)

    noises = torch.randn(opt.batch_size, opt.noise_dim, 1, 1, 1).to(device)
    save_noises = torch.randn(opt.batch_size, opt.noise_dim, 1, 1, 1).to(device)

    for epoch in range(opt.max_epoch):
        for i, img in tqdm(enumerate(dataloader)):
            real_img = img.float()
            real_img = real_img.to(device)

            if i % opt.d_train == 0:
                d_optimizer.zero_grad()
                output = net_d(real_img)
                loss_d_real = criterion(output, true_labels)
                loss_d_real.backward()

                noises = noises.detach()
                fake_image = net_g(noises).detach()
                output = net_d(fake_image)
                loss_d_fake = criterion(output, fake_labels)
                loss_d_fake.backward()

                d_optimizer.step()
                loss = loss_d_real + loss_d_fake

            if epoch % opt.g_train == 0:
                g_optimizer.zero_grad()
                noises.data.copy_(torch.randn(opt.batch_size, opt.noise_dim, 1, 1, 1))
                fake_image = net_g(noises)
                output = net_d(fake_image)
                loss_g = criterion(output, true_labels)
                loss_g.backward()

                g_optimizer.step()

            if epoch % opt.save_epoch == 0:
                save_fake_image = net_g(save_noises)
                for j in range(opt.batch_size):
                    save_mat = save_fake_image.data[j, 0].cpu().numpy()
                    save_path = os.path.join(opt.save_img, '{}__{}.mat'.format(epoch, j))
                    io.savemat(save_path, {'instance': save_mat})
                torch.save(net_g.state_dict(), os.path.join(opt.save_model, 'gen_epoch_{}.pth'.format(epoch)))
                torch.save(net_d.state_dict(), os.path.join(opt.save_model, 'dis_epoch_{}.pth'.format(epoch)))


def main():
    train()


if __name__ == '__main__':
    main()



