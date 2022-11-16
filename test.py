import torch
import numpy
import scipy.io as io
from nets import Generator, Discriminator
from config import Config


opt = Config()


def generate(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_g = Generator().eval()
    net_d = Discriminator().eval()
    map_location = lambda storage, loc: storage

    net_g.load_state_dict(torch.load(r'D:\3dgan\log_1\model\gen_epoch_180.pth', map_location=map_location))
    net_d.load_state_dict(torch.load(r'D:\3dgan\log_1\model\dis_epoch_180.pth', map_location=map_location))

    net_d.to(device)
    net_g.to(device)

    noise = torch.randn(10, opt.noise_dim, 1, 1, 1).normal_(0, 0.33).to(device)
    fake_image = net_g(noise)
    score = net_d(fake_image).detach()
    index = torch.topk(score, 1)
    print(index)
    gen_img = fake_image.data[4, 0].cpu().numpy()
    io.savemat(r'D:\3dgan\log_1\generate__{}.mat'.format(4), {'instance': gen_img})


def main():
    generate()


if __name__ == '__main__':
    main()
