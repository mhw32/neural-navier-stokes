import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from src.navierstokes.generate import DATA_DIR
from src.navierstokes.models import RNNDiffEq, ODEDiffEq
from src.navierstokes.utils import (
    spatial_coarsen, AverageMeter, save_checkpoint, 
    MODEL_DIR, dynamics_prediction_error_numpy, 
    log_normal_pdf, normal_kl)
from src.navierstokes.baseline import coarsen_fine_systems


def mean_squared_error(pred, true):
    batch_size = pred.size(0)
    pred, true = pred.view(batch_size, -1), true.view(batch_size, -1)
    mse = torch.mean(torch.pow(pred - true, 2), dim=1)
    return torch.mean(mse)  # over batch size


def neural_ode_loss(u_out, v_out, p_out, u_pred, v_pred, p_pred,
                    z, qz_mu, qz_logvar, obs_std=0.3):
    """Latent variable model objective using Neural ODE."""
    device = u_out.device
    noise_std_ = torch.zeros(pred_x.size()).to(device) + obs_std  # hardcoded logvar
    noise_logvar = 2. * torch.log(noise_std_).to(device)

    logp_u = log_normal_pdf(u_out, u_pred, noise_logvar)
    logp_v = log_normal_pdf(v_out, v_pred, noise_logvar)
    logp_p = log_normal_pdf(p_out, p_pred, noise_logvar)

    logp_u = logp_u.sum(-1).sum(-1)
    logp_v = logp_v.sum(-1).sum(-1)
    logp_p = logp_p.sum(-1).sum(-1)
    logp = logp_u + logp_v + logp_p  # sum 3 components together

    pz_mu = torch.zeros_like(z)
    pz_logvar = torch.zeros_like(z)
    analytic_kl = normal_kl(qz_mu, qz_logvar, pz_mu, pz_logvar).sum(-1)
    loss = torch.mean(-logp + analytic_kl, dim=0)
    return loss


def numpy_to_torch(array, device):
    return torch.from_numpy(array).float().to(device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rnn', help='rnn|ode')
    parser.add_argument('--batch-time', type=int, default=50, 
                        help='batch of timesteps [default: 50]')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='batch size [default: 100]')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs [default: 2000]')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--test-only', action='store_true', default=False)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(1337)
    np.random.seed(1337)

    model_dir = os.path.join(MODEL_DIR, args.model)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(DATA_DIR, '1000_fine_systems.pickle'), 'rb') as fp:
        fine_systems = pickle.load(fp)

    with open(os.path.join(DATA_DIR, '1000_coarse_systems.pickle'), 'rb') as fp:
        coarse_systems = pickle.load(fp)

    # generate coarsened fine systems
    coarsened_systems = coarsen_fine_systems(fine_systems, coarse_systems)
    coarsened_config = coarsened_systems[0]['config']
    assert coarsened_config['nx'] == coarsened_config['ny']
    grid_dim = coarsened_config['nx']
    nt = coarsened_config['nt']  # num timesteps
    dt = coarsened_config['dt']  # time stepsize
    timesteps = np.arange(nt) * dt

    T = coarsened_systems[0]['u'].shape[0]
    N = len(coarsened_systems)

    N_train = int(0.8 * N)
    N_val = int(0.1 * N)

    print('Divide data into train/val/test sets.')

    # split into train/val/test sets
    train_coarsened_systems = coarsened_systems[:N_train]
    val_coarsened_systems = coarsened_systems[N_train:N_train+N_val]
    test_coarsened_systems = coarsened_systems[N_train+N_val:]

    # get all momentum and pressure sequences in a matrix
    # shape: 800 x T x grid_size x grid_size
    train_u_mat = np.stack([system['u'] for system in train_coarsened_systems])
    train_v_mat = np.stack([system['v'] for system in train_coarsened_systems])
    train_p_mat = np.stack([system['p'] for system in train_coarsened_systems])

    val_u_mat = np.stack([system['u'] for system in val_coarsened_systems])
    val_v_mat = np.stack([system['v'] for system in val_coarsened_systems])
    val_p_mat = np.stack([system['p'] for system in val_coarsened_systems])

    test_u_mat = np.stack([system['u'] for system in test_coarsened_systems])
    test_v_mat = np.stack([system['v'] for system in test_coarsened_systems])
    test_p_mat = np.stack([system['p'] for system in test_coarsened_systems])

    # divide data into input and target sequences
    train_u_in, train_v_in, train_p_in = train_u_mat[:, :T-1], train_v_mat[:, :T-1], train_p_mat[:, :T-1]
    train_u_out, train_v_out, train_p_out = train_u_mat[:, 1:], train_v_mat[:, 1:], train_p_mat[:, 1:]

    val_u_in, val_v_in, val_p_in = val_u_mat[:, :T-1], val_v_mat[:, :T-1], val_p_mat[:, :T-1]
    val_u_out, val_v_out, val_p_out = val_u_mat[:, 1:], val_v_mat[:, 1:], val_p_mat[:, 1:]

    test_u_in, test_v_in, test_p_in = test_u_mat[:, :T-1], test_v_mat[:, :T-1], test_p_mat[:, :T-1]
    test_u_out, test_v_out, test_p_out = test_u_mat[:, 1:], test_v_mat[:, 1:], test_p_mat[:, 1:]

    t_in, t_out = timesteps[:T-1], timesteps[1:]  # same for train/val/test

    print('Initialize model and optimizer.')

    if args.model == 'rnn':
        model = RNNDiffEq(grid_dim)
    elif args.model == 'ode':
        model = ODEDiffEq(grid_dim)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_loss = np.inf
    test_loss_item = np.inf

    if not args.test_only:
        pbar = tqdm(total=args.epochs)
        for iteration in range(args.epochs):
            model.train()
            # sample a batch of contiguous timesteps
            start_T = np.random.choice(np.arange(T - 1 - args.batch_time), size=args.batch_size)
            batch_I = np.random.choice(np.arange(N_train), size=args.batch_size)

            batch_u_in = numpy_to_torch(train_u_in[batch_I, start_T:start_T+args.batch_time, ...], device)
            batch_v_in = numpy_to_torch(train_v_in[batch_I, start_T:start_T+args.batch_time, ...], device)
            batch_p_in = numpy_to_torch(train_p_in[batch_I, start_T:start_T+args.batch_time, ...], device)
            batch_t_in = numpy_to_torch(t_in[start_T:start_T+args.batch_time], device)
            batch_u_out = numpy_to_torch(train_u_out[batch_I, start_T:start_T+args.batch_time, ...], device)
            batch_v_out = numpy_to_torch(train_v_out[batch_I, start_T:start_T+args.batch_time, ...], device)
            batch_p_out = numpy_to_torch(train_p_out[batch_I, start_T:start_T+args.batch_time, ...], device)

            optimizer.zero_grad()
            
            if args.model == 'rnn':
                batch_u_pred, batch_v_pred, batch_p_pred, _ = model(
                    batch_u_in, batch_v_in, batch_p_in)
                loss = (mean_squared_error(batch_u_pred, batch_u_out) + 
                        mean_squared_error(batch_v_pred, batch_v_out) + 
                        mean_squared_error(batch_p_pred, batch_p_out))
            else:
                batch_u_pred, batch_v_pred, batch_p_pred, z, qz_mu, qz_logvar, _ \
                    = model(batch_u_in, batch_v_in, batch_p_in, batch_t_in)
                loss = neural_ode_loss(batch_u_out, batch_v_out, batch_p_out, 
                                       batch_u_pred, batch_v_pred, batch_p_pred,
                                       z, qz_mu, qz_logvar, obs_std=0.3)

            loss.backward()
            optimizer.step()
            pbar.update() 
            pbar.set_postfix({'train loss': loss.item(),
                              'test_loss': test_loss_item})

            if iteration % 10 == 0:
                model.eval()
                with torch.no_grad():
                    print('Computing validation error.')
                    
                    # test on validation dataset as metric
                    val_u_in, val_u_out = numpy_to_torch(val_u_in, device), numpy_to_torch(val_u_out, device)
                    val_v_in, val_v_out = numpy_to_torch(val_v_in, device), numpy_to_torch(val_v_out, device)
                    val_p_in, val_p_out = numpy_to_torch(val_p_in, device), numpy_to_torch(val_p_out, device)
                    all_t_in = numpy_to_torch(t_in, device)

                    if args.model == 'rnn':
                        val_u_pred, val_v_pred, val_p_pred, _ = model(val_u_in, val_v_in, val_p_in)
                        val_loss = (mean_squared_error(val_u_pred, val_u_out) + 
                                    mean_squared_error(val_v_pred, val_v_out) + 
                                    mean_squared_error(val_p_pred, val_p_out))
                    else:
                        val_u_pred, val_v_pred, val_p_pred, z, qz_mu, qz_logvar, _ \
                            = model(val_u_in, val_v_in, val_p_in, all_t_in)
                        val_loss = neural_ode_loss(val_u_out, val_v_out, val_p_out, 
                                                   val_u_pred, val_v_pred, val_p_pred,
                                                   z, qz_mu, qz_logvar, obs_std=0.3)

                    val_loss_item = val_loss.item()
                    pbar.set_postfix({'train loss': loss.item(),
                                      'val_loss': val_loss_item}) 

                    if val_loss.item() < best_loss:
                        best_loss = val_loss.item()
                        is_best = True

                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'val_loss': val_loss.item(),
                    }, is_best, model_dir)    

        pbar.close()

    # load the best model
    print('Loading best weights (by validation error).')
    checkpoint = torch.load(os.path.join(model_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    model = model.eval()

    with torch.no_grad():
        # measure accuracy on unseen test set!
        print('Applying model to test set.')
        test_u_in, test_u_out = numpy_to_torch(test_u_in, device), numpy_to_torch(test_u_out, device)
        test_v_in, test_v_out = numpy_to_torch(test_v_in, device), numpy_to_torch(test_v_out, device)
        test_p_in, test_p_out = numpy_to_torch(test_p_in, device), numpy_to_torch(test_p_out, device)
        all_t_in = numpy_to_torch(t_in, device)

        if args.model == 'rnn':
            test_u_pred, test_v_pred, test_p_pred, _ = model(test_u_in, test_v_in, test_p_in)
        else:
            test_u_pred, test_v_pred, test_p_pred, z, z_mu, z_logvar, _ \
                = model(test_u_in, test_v_in, test_p_in, all_t_in)
        
        test_u_mse, test_v_mse, test_p_mse = dynamics_prediction_error_torch(
            test_u_out, test_v_out, test_p_out,
            test_u_pred, test_v_pred, test_p_pred)
        
        test_u_mse = test_u_mse.cpu().numpy()
        test_v_mse = test_u_mse.cpu().numpy()
        test_p_mse = test_u_mse.cpu().numpy()

        np.savez(os.path.join(model_dir, 'test_error.npz'),
                 u_mse=test_u_mse, v_mse=test_v_mse, p_mse=test_p_mse)
