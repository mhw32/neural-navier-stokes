import torch
from src.spiral.ode import visualize, NeuralODE

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.checkpoint_path)
    ode = NeuralODE(4, 2, 20, 25, 1000).to(device)

    ode.load_state_dict(checkpoint['state_dict'])
    ode = ode.eval()

    visualize(ode, checkpoint['orig_trajs'], checkpoint['samp_trajs'],
              checkpoint['orig_ts'])

