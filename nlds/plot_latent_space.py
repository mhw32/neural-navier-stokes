import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_generator(model, T, temp, fname, save_dir, lines=False):
    z_sample, p_z_logit, x_sample, p_x_mu, p_x_logvar = model.generative_model(1, T, temp)

    if fname is None or save_dir is None:
        fname = None
    else:
        fname = save_dir + '/' + fname
    x_sample = np.squeeze(x_sample.detach().cpu().numpy())
    z_sample = np.squeeze(z_sample.detach().cpu().numpy())
    plot_latent_space(x_sample, z_sample, fname=fname, lines=lines)

def plot_inference(model, data, temp, fname, save_dir, lines=False):
    z_sample, z_logit, x_sample, x_mu, x_logvar = model.inference_network(data, temp)

    if fname is None or save_dir is None:
        fname = None
    else:
        fname = save_dir + '/' + fname
    x_sample = np.squeeze(x_sample.detach().cpu().numpy())
    z_sample = np.squeeze(z_sample.detach().cpu().numpy())
    plot_latent_space(x_sample, z_sample, fname=fname, lines=lines)

def plot_latent_space(x, z, fname=None, lines=False):
    # x is np array of shape (TIMESTEPS, 3)
    # z is np array of shape (TIMESTEPS, 2)
    assert x.shape[0] == z.shape[0]
    timesteps = x.shape[0]

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    colours = list()
    for i in range(timesteps):
        # RGB
        #c = (1.*z[i,0],0.,1.*z[i,1])
        c=plt.cm.jet(z[i,0])
        colours.append(c)

    colours = np.array(colours)
    if not lines:
        ax.scatter(x[:,0], x[:,1], x[:,2], color=colours)
    else:
        ax.plot(x[:,0], x[:,1], x[:,2])

    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()


if __name__ == "__main__":
    x = np.arange(30)
    x = np.tile(x,[3,1])
    z = np.random.uniform(0,1,30)

    plot_latent_space(x,z)


