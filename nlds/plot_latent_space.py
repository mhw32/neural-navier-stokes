import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_generator(model, T, temp, iteration_num):
    z_sample, p_z_logit, x_sample, p_x_mu, p_x_logvar = model.generative_model(1, T, temp)

    fname = "generative_iteration_{}.png".format(iteration_num)
    x_sample = np.squeeze(x_sample.detach().numpy())
    z_sample = np.squeeze(z_sample.detach().numpy())
    plot_latent_space(x_sample, z_sample, fname=fname)

def plot_inference(model, data, temp, iteration_num):
    z_sample, z_logit, x_sample, x_mu, x_logvar = model.inference_network(data, temp)

    fname = "inference_iteration_{}.png".format(iteration_num)
    x_sample = np.squeeze(x_sample.detach().numpy())
    z_sample = np.squeeze(z_sample.detach().numpy())
    plot_latent_space(x_sample, z_sample, fname=fname)

def plot_latent_space(x, z, fname=None):
    # x is np array of shape (TIMESTEPS, 3)
    # z is np array of shape (TIMESTEPS, 2)
    assert x.shape[0] == z.shape[0]
    timesteps = x.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(timesteps):
        # RGB
        c = (1.*z[i,0],0.,1.*z[i,1])
        ax.scatter(x[i,0], x[i,1], x[i,2], color=c)

    if fname is not None:
        plt.savefig(fname)


if __name__ == "__main__":
    x = np.arange(30)
    x = np.tile(x,[3,1])
    z = np.random.uniform(0,1,30)

    plot_latent_space(x,z)


