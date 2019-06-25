import torch
from src.spiral.ldm import LDM, visualize as vis_ldm
from src.spiral.ndm import NDM, visualize as vis_ndm
from src.spiral.ode import NeuralODE, visualize as vis_ode
from src.spiral.rnn import RNN, visualize as vis_rnn

name2vis = {
    'ldm': vis_ldm,
    'ndm': vis_ndm,
    'ode': vis_ode,
    'rnn': vis_rnn,
}

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
    model_name = checkpoint['model_name']

    if model_name == 'ldm':
        model = LDM(2, 4, 20, 20, 25).to(device)
    elif model_name == 'ndm':
        model = NDM(3, 4, 20, 20, 25).to(device)
    elif model_name == 'ode':
        model = NeuralODE(4, 2, 20, 25, 1000).to(device)
    elif model_name == 'rnn':
        model = RNN(3, 2, 20, 25).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.eval()

    if model_name == 'ldm':
        vis_ldm(model, checkpoint['orig_trajs'], checkpoint['orig_ts'],
                checkpoint['samp_trajs'], index=args.index)
    elif model_name == 'ndm':
        vis_ndm(model, checkpoint['orig_trajs'], checkpoint['orig_ts'],
                checkpoint['samp_trajs'], index=args.index)
    elif model_name == 'ode':
        vis_ode(model, checkpoint['orig_trajs'], checkpoint['samp_trajs'],
                checkpoint['orig_ts'], index=args.index)
    elif model_name == 'rnn':
        vis_rnn(model, checkpoint['orig_trajs'], checkpoint['orig_ts'],
                checkpoint['samp_trajs'], index=args.index)
