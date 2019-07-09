import os
import sys
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from src.navierstokes.generate import DATA_DIR, DATA_SM_DIR
from src.navierstokes.models import RNNDiffEq, ODEDiffEq
from src.navierstokes.utils import (
    spatial_coarsen, AverageMeter, save_checkpoint, 
    MODEL_DIR, dynamics_prediction_error_torch, 
    log_normal_pdf, normal_kl, load_systems)
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

    print('loading fine systems')
    u_fine, v_fine, p_fine = load_systems(DATA_SM_DIR, fine=True)

    N = u_fine.shape[0]
    nx, ny = u_fine.shape[2], u_fine.shape[3]
    x_fine = np.linspace(0, 2, nx)  # slightly hardcoded
    y_fine = np.linspace(0, 2, ny)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    u_coarsened, v_coarsened, p_coarsened = coarsen_fine_systems(
        X_fine, Y_fine, u_fine, v_fine, p_fine)

    # set some hyperparameters
    grid_dim = u_coarsened.shape[2]
    T = u_coarsened.shape[1]
    dt = 0.001
    
    timesteps = np.arange(T) * dt
    timesteps = np.repeat(timesteps[np.newaxis], N, axis=0)

    N = u_fine.shape[0]
    N_train = int(0.8 * N)
    N_val = int(0.1 * N)

    print('Divide data into train/val/test sets.')

    # get all momentum and pressure sequences in a matrix
    # shape: N_train x T x grid_size x grid_size
    train_u_mat = u_coarsened[:N_train, ...]
    train_v_mat = v_coarsened[:N_train, ...]
    train_p_mat = p_coarsened[:N_train, ...]
    train_timesteps = timesteps[:N_train]

    val_u_mat = u_coarsened[N_train:(N_train+N_val), ...]
    val_v_mat = v_coarsened[N_train:(N_train+N_val), ...]
    val_p_mat = p_coarsened[N_train:(N_train+N_val), ...]
    val_timesteps = timesteps[N_train:(N_train+N_val)]

    test_u_mat = u_coarsened[N_train+N_val:, ...]
    test_v_mat = v_coarsened[N_train+N_val:, ...]
    test_p_mat = p_coarsened[N_train+N_val:, ...]
    test_timesteps = timesteps[N_train+N_val:]
    
    # divide data into input and target sequences
    train_u_in, train_v_in, train_p_in = train_u_mat[:, :T-1], train_v_mat[:, :T-1], train_p_mat[:, :T-1]
    train_u_out, train_v_out, train_p_out = train_u_mat[:, 1:], train_v_mat[:, 1:], train_p_mat[:, 1:]

    val_u_in, val_v_in, val_p_in = val_u_mat[:, :T-1], val_v_mat[:, :T-1], val_p_mat[:, :T-1]
    val_u_out, val_v_out, val_p_out = val_u_mat[:, 1:], val_v_mat[:, 1:], val_p_mat[:, 1:]

    test_u_in, test_v_in, test_p_in = test_u_mat[:, :T-1], test_v_mat[:, :T-1], test_p_mat[:, :T-1]
    test_u_out, test_v_out, test_p_out = test_u_mat[:, 1:], test_v_mat[:, 1:], test_p_mat[:, 1:]

    train_t_in, train_t_out = train_timesteps[:T-1], train_timesteps[1:]
    val_t_in, val_t_out = val_timesteps[:T-1], val_timesteps[1:]
    test_t_in, test_t_out = test_timesteps[:T-1], test_timesteps[1:]

    print('Initialize model and optimizer.')

    if args.model == 'rnn':
        model = RNNDiffEq(grid_dim)
    elif args.model == 'ode':
        model = ODEDiffEq(grid_dim)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_loss = np.inf
    val_loss_item = np.inf

    if not args.test_only:
        pbar = tqdm(total=args.epochs)
        for iteration in range(args.epochs):
            model.train()
            # sample a batch of contiguous timesteps
            start_T = np.random.choice(np.arange(T - 1 - args.batch_time), size=args.batch_size)
            batch_I = np.random.choice(np.arange(N_train), size=args.batch_size)

            def build_batch(A, batch_indices, start_time_batch, time_lapse):
                # A = batch_size, T, grid_dim, grid_dim
                batch_size = A.shape[0]
                subA = A[batch_indices]
                batchA = np.stack([
                    subA[i, start_time_batch[i]:start_time_batch[i]+time_lapse, ...]
                    for i in range(batch_size)
                ])
                return batchA

            batch_u_in = numpy_to_torch(build_batch(train_u_in, batch_I, start_T, args.batch_time), device)
            batch_v_in = numpy_to_torch(build_batch(train_v_in, batch_I, start_T, args.batch_time), device)
            batch_p_in = numpy_to_torch(build_batch(train_p_in, batch_I, start_T, args.batch_time), device)
            batch_t_in = numpy_to_torch(build_batch(train_t_in, batch_I, start_T, args.batch_time), device)
            batch_u_out = numpy_to_torch(build_batch(train_u_out, batch_I, start_T, args.batch_time), device)
            batch_v_out = numpy_to_torch(build_batch(train_v_out, batch_I, start_T, args.batch_time), device)
            batch_p_out = numpy_to_torch(build_batch(train_p_out, batch_I, start_T, args.batch_time), device)

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
                              'val_loss': val_loss_item})

            if iteration % 10 == 0 and iteration > 0:
                model.eval()
                with torch.no_grad():
                    # test on validation dataset as metric
                    _val_u_in, _val_u_out = numpy_to_torch(val_u_in, device), numpy_to_torch(val_u_out, device)
                    _val_v_in, _val_v_out = numpy_to_torch(val_v_in, device), numpy_to_torch(val_v_out, device)
                    _val_p_in, _val_p_out = numpy_to_torch(val_p_in, device), numpy_to_torch(val_p_out, device)
                    _val_t_in = numpy_to_torch(val_t_in, device)

                    if args.model == 'rnn':
                        val_u_pred, val_v_pred, val_p_pred, _ = model(_val_u_in, _val_v_in, _val_p_in)
                        val_loss = (mean_squared_error(val_u_pred, _val_u_out) + 
                                    mean_squared_error(val_v_pred, _val_v_out) + 
                                    mean_squared_error(val_p_pred, _val_p_out))
                    else:
                        val_u_pred, val_v_pred, val_p_pred, z, qz_mu, qz_logvar, _ \
                            = model(_val_u_in, _val_v_in, _val_p_in, _val_t_in)
                        val_loss = neural_ode_loss(_val_u_out, _val_v_out, _val_p_out, 
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
        print('Applying model to test set (with teacher forcing)')
        test_u_in, test_u_out = numpy_to_torch(test_u_in, device), numpy_to_torch(test_u_out, device)
        test_v_in, test_v_out = numpy_to_torch(test_v_in, device), numpy_to_torch(test_v_out, device)
        test_p_in, test_p_out = numpy_to_torch(test_p_in, device), numpy_to_torch(test_p_out, device)
        test_t_in = numpy_to_torch(test_t_in, device)

        if args.model == 'rnn':
            test_u_pred, test_v_pred, test_p_pred, _ = model(test_u_in, test_v_in, test_p_in)
        else:
            test_u_pred, test_v_pred, test_p_pred, z, z_mu, z_logvar, _ \
                = model(test_u_in, test_v_in, test_p_in, test_t_in)
        
        test_u_mse, test_v_mse, test_p_mse = dynamics_prediction_error_torch(
            test_u_out, test_v_out, test_p_out,
            test_u_pred, test_v_pred, test_p_pred)
        
        test_u_mse = test_u_mse.cpu().numpy()
        test_v_mse = test_u_mse.cpu().numpy()
        test_p_mse = test_u_mse.cpu().numpy()

        np.savez(os.path.join(model_dir, 'test_error_teacher_forcing.npz'),
                 u_mse=test_u_mse, v_mse=test_v_mse, p_mse=test_p_mse)

    print('Reload the best weights')
    checkpoint = torch.load(os.path.join(model_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    model = model.eval()

    with torch.no_grad():
        print('Applying model to test set (no teacher forcing)')
        test_u_in, test_u_out = numpy_to_torch(test_u_in, device), numpy_to_torch(test_u_out, device)
        test_v_in, test_v_out = numpy_to_torch(test_v_in, device), numpy_to_torch(test_v_out, device)
        test_p_in, test_p_out = numpy_to_torch(test_p_in, device), numpy_to_torch(test_p_out, device)
        test_t_in = numpy_to_torch(test_t_in, device)

        test_u_pred, test_v_pred, test_p_pred = [], [], []

        # take just the first step
        u, v, p = test_u_in[0], test_v_in[0], test_p_in[0]
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
        p = p.unsqueeze(0)

        rnn_h0 = None  # this will cause us to read the initial conditions

        for _ in range(T - 1):
            if args.model == 'rnn':
                u, v, p, rnn_h0 = model(u, v, p, rnn_h0=rnn_h0)
            else:
                raise NotImplementedError
            
            test_u_pred.append(copy.deepcopy(u))
            test_v_pred.append(copy.deepcopy(v))
            test_p_pred.append(copy.deepcopy(p))
        
        test_u_pred = torch.cat(test_u_pred, dim=0)
        test_v_pred = torch.cat(test_v_pred, dim=0)
        test_p_pred = torch.cat(test_p_pred, dim=0)

        test_u_mse, test_v_mse, test_p_mse = dynamics_prediction_error_torch(
            test_u_out, test_v_out, test_p_out,
            test_u_pred, test_v_pred, test_p_pred)
        
        test_u_mse = test_u_mse.cpu().numpy()
        test_v_mse = test_u_mse.cpu().numpy()
        test_p_mse = test_u_mse.cpu().numpy()

        np.savez(os.path.join(model_dir, 'test_error_no_teacher_forcing.npz'),
                 u_mse=test_u_mse, v_mse=test_v_mse, p_mse=test_p_mse)
