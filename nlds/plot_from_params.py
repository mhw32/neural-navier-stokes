import torch

from rslds import RSLDS
from rsnlds import RSSNLDS

from datasets import BernoulliLorenz
from plot_latent_space import plot_inference, plot_generator


def main(params_path, model, epoch, plot_dir):
    params = torch.load(params_path)
    model.load_state_dict(params)
    model.eval()

    test_num_timesteps = 2000
    test_dataset = BernoulliLorenz(1, test_num_timesteps, dt=0.01)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Plot inference network
    for _, test_data in enumerate(test_loader):
        test_data = test_data.to(device)
        fname = "epoch_{}_inference.png".format(epoch)
        plot_inference(model, test_data, 0.1, fname, plot_dir)

    # Plot generator network
    fname = "epoch_{}_generative.png".format(epoch)
    plot_generator(model, 2000, 0.1, fname, plot_dir)


if __name__ == "__main__":
    # How to run:
    # python plot_from_params.py --epoch 260 --model linear --plot-dir "/home/nclkong/nlsys/switching-nonlinear-systems/plots/" --cuda
    # python plot_from_params.py --epoch 212 --model nonlinear --plot-dir "/home/nclkong/nlsys/switching-nonlinear-systems/plots/" --cuda

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=260)
    parser.add_argument('--model', type=str, default="linear")
    parser.add_argument('--plot-dir', type=str, default='/home/nclkong/nlsys/switching-nonlinear-systems/plots/')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    if args.model == "linear":
        path_to_params = "/mnt/fs5/nclkong/rslds/epoch_{}_params.pth".format(args.epoch)
        model = RSLDS(2, 1, 3, 100, 10, 3, 1, 20, 20)
        model = model.to(device)
    else:
        assert args.model == "nonlinear"
        path_to_params = "/mnt/fs5/nclkong/rsnlds/epoch_{}_params.pth".format(args.epoch)
        model = RSSNLDS(2, 1, 3, 100, 10, 10, 20, 20, 64, 64)
        model = model.to(device)

    main(path_to_params, model, args.epoch, args.plot_dir)


