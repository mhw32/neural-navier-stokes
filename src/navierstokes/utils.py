import os
import torch
import shutil
import numpy as np
from tqdm import tqdm

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_DIR = os.path.realpath(MODEL_DIR)


def numpy_to_torch(array, device):
    return torch.from_numpy(array).float().to(device)


def spatial_coarsen(X, Y, u_seq, v_seq, p_seq, agg_x=4, agg_y=4):
    """Given dynamics of a certain coarseness, we want to 
    aggregate by averaging over regions in the spatial grid.

    Args
    ----
    X := np.array (size: nx by ny)
         meshgrid for x 
    Y := np.array (size: nx by ny)
         meshgrid for y
    u_seq := np.array (size: T x nx by ny)
             u-momentum components
    v_seq := np.array (size: T x nx by ny)
             v-momentum components
    p_seq := np.array (size: T x nx by ny)
             pressure components
    agg_x := integer (default: 4)
             coarsen factor for x-coordinates
    agg_y := integer (default: 4)
             coarsen factor for y-coordinates

    We return each element but coarsened.
    """
    nx, ny = X.shape[0], X.shape[1]
    T = u_seq.shape[0]

    assert nx % agg_x == 0
    assert ny % agg_y == 0

    new_u_seq = np.zeros((T, nx // agg_x, ny // agg_y))
    new_v_seq = np.zeros((T, nx // agg_x, ny // agg_y))
    new_p_seq = np.zeros((T, nx // agg_x, ny // agg_y))

    new_x = np.linspace(0, 2, nx // agg_x)
    new_y = np.linspace(0, 2, ny // agg_y)
    new_X, new_Y = np.meshgrid(new_x, new_y)

    for i in range(nx // agg_x):
        for j in range(ny // agg_x):
            u_sub = u_seq[:, i*agg_x:(i+1)*agg_x, j*agg_y:(j+1)*agg_y].reshape(T, -1)
            v_sub = v_seq[:, i*agg_x:(i+1)*agg_x, j*agg_y:(j+1)*agg_y].reshape(T, -1)
            p_sub = p_seq[:, i*agg_x:(i+1)*agg_x, j*agg_y:(j+1)*agg_y].reshape(T, -1)

            new_u_seq[:, i, j] = np.mean(u_sub, axis=1)
            new_v_seq[:, i, j] = np.mean(v_sub, axis=1)
            new_p_seq[:, i, j] = np.mean(p_sub, axis=1)

    return new_X, new_Y, new_u_seq, new_v_seq, new_p_seq


def dynamics_prediction_error_numpy(
    u_seq, v_seq, p_seq, u_hat_seq, v_hat_seq, p_hat_seq, dim=1):
    """
    Mean squared error between predicted momentum and 
    pressure and ground truth dynamics.

    Each object has shape (batch_size, T, grid_dim, grid_dim)
    """
    u_seq_mse = np.sum(np.sum(np.power(u_hat_seq - u_seq, 2), dim), dim)
    v_seq_mse = np.sum(np.sum(np.power(v_hat_seq - v_seq, 2), dim), dim)
    p_seq_mse = np.sum(np.sum(np.power(p_hat_seq - p_seq, 2), dim), dim)

    return u_seq_mse, v_seq_mse, p_seq_mse


def dynamics_prediction_error_torch(
    u_seq, v_seq, p_seq, u_hat_seq, v_hat_seq, p_hat_seq, dim=1):
    """
    Mean squared error between predicted momentum and 
    pressure and ground truth dynamics.

    Each object has shape (batch_size, T, grid_dim, grid_dim)
    """
    u_seq_mse = torch.sum(torch.sum(torch.pow(u_hat_seq - u_seq, 2), dim), dim)
    v_seq_mse = torch.sum(torch.sum(torch.pow(v_hat_seq - v_seq, 2), dim), dim)
    p_seq_mse = torch.sum(torch.sum(torch.pow(p_hat_seq - p_seq, 2), dim), dim)

    return u_seq_mse, v_seq_mse, p_seq_mse


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def mean_squared_error(pred, true):
    batch_size = pred.size(0)
    pred, true = pred.view(batch_size, -1), true.view(batch_size, -1)
    mse = torch.mean(torch.pow(pred - true, 2), dim=1)
    return torch.mean(mse)  # over batch size


def log_normal_pdf(x, mean, logvar):
    # sigma = 0.5 * torch.exp(logvar)
    # return dist.Normal(mean, sigma).log_prob(x)
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def load_systems(data_dir, fine=True):
    subdir = 'fine' if fine else 'coarse'
    basename = os.path.realpath(os.path.join(data_dir, 'numpy', subdir))
    filenames = os.listdir(basename)
    n = len(filenames)

    print('Loading data from {} ({} files).'.format(basename, n))   
    
    u_mat, v_mat, p_mat = [], [], []
    for i in tqdm(range(n)):
        name = 'system_{}.npz'.format(i)
        assert name in filenames
        data_i = np.load(os.path.join(basename, name))
        u_mat.append(data_i['u'])
        v_mat.append(data_i['v'])
        p_mat.append(data_i['p'])

    u_mat = np.stack(u_mat)
    v_mat = np.stack(v_mat)
    p_mat = np.stack(p_mat)

    return u_mat, v_mat, p_mat


def linear_systems(n=1000, t=200, dt=0.001, grid_dim=50):
    """
    Randomly pick a intercept and slope.
    Generate according to 
        du/dt = slope
    with initial conditions being the intercept.
    """
    # unique intercept for each spatial index and system
    u_intercept = np.repeat(np.random.uniform(0, 1, size=n*grid_dim**2)[:, np.newaxis], t, axis=1)
    v_intercept = np.repeat(np.random.uniform(0, 1, size=n*grid_dim**2)[:, np.newaxis], t, axis=1)
    p_intercept = np.repeat(np.random.uniform(0, 1, size=n*grid_dim**2)[:, np.newaxis], t, axis=1)
    
    u_intercept = np.swapaxes(u_intercept.reshape(n, grid_dim, grid_dim, t), -1, 1)
    v_intercept = np.swapaxes(v_intercept.reshape(n, grid_dim, grid_dim, t), -1, 1)
    p_intercept = np.swapaxes(p_intercept.reshape(n, grid_dim, grid_dim, t), -1, 1)

    # one slope for each spatial index (same for all systems)
    u_slope = np.random.uniform(0, 1, size=grid_dim**2).reshape(grid_dim, grid_dim)
    v_slope = np.random.uniform(0, 1, size=grid_dim**2).reshape(grid_dim, grid_dim)
    p_slope = np.random.uniform(0, 1, size=grid_dim**2).reshape(grid_dim, grid_dim)

    u_slope = u_slope[np.newaxis, np.newaxis, :, :]
    v_slope = v_slope[np.newaxis, np.newaxis, :, :]
    p_slope = p_slope[np.newaxis, np.newaxis, :, :]

    u_slope = np.repeat(np.repeat(u_slope, n, axis=0), t, axis=1)
    v_slope = np.repeat(np.repeat(v_slope, n, axis=0), t, axis=1)
    p_slope = np.repeat(np.repeat(p_slope, n, axis=0), t, axis=1)

    timesteps = np.repeat(np.arange(t)[np.newaxis, :], n, axis=0) * dt
    timesteps = timesteps[:, :, np.newaxis, np.newaxis]

    u_mat = timesteps * u_slope + u_intercept
    v_mat = timesteps * v_slope + v_intercept
    p_mat = timesteps * p_slope + p_intercept

    return u_mat, v_mat, p_mat


def neural_ode_loss(u_out, v_out, p_out, u_pred, v_pred, p_pred,
                    z, qz_mu, qz_logvar, obs_std=0.3):
    """Latent variable model objective using latent Neural ODE."""
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
