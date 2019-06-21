import torch
from src.spiral.ldm import visualize
from src.spiral.ndm import NDM

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.checkpoint_path)
    rnn = NDM(3, 4, 20, 20, 25).to(device)
    rnn.load_state_dict(checkpoint['state_dict'])
    rnn = rnn.eval()

    visualize(rnn, checkpoint['orig_trajs'], checkpoint['orig_ts'],
              checkpoint['samp_trajs'], index=args.index)

