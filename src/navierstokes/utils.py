import numpy as np


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

    new_X = np.zeros((nx // agg_x, ny // agg_y))
    new_Y = np.zeros((nx // agg_x, ny // agg_y))
    new_u_seq = np.zeros((T, nx // agg_x, ny // agg_y))
    new_v_seq = np.zeros((T, nx // agg_x, ny // agg_y))
    new_p_seq = np.zeros((T, nx // agg_x, ny // agg_y))

    for i in range(nx // agg_x):
        for j in range(ny // agg_x):
            new_X[i, j] = np.sum(X[i*agg_x:(i+1)*agg_x, j*agg_y:(j+1)*agg_y])
            new_Y[i, j] = np.sum(Y[i*agg_x:(i+1)*agg_x, j*agg_y:(j+1)*agg_y])
            new_u_seq[:, i, j] = np.sum(u_seq[:, i*agg_x:(i+1)*agg_x, j*agg_y:(j+1)*agg_y])
            new_v_seq[:, i, j] = np.sum(v_seq[:, i*agg_x:(i+1)*agg_x, j*agg_y:(j+1)*agg_y])
            new_p_seq[:, i, j] = np.sum(p_seq[:, i*agg_x:(i+1)*agg_x, j*agg_y:(j+1)*agg_y])

    return new_X, new_Y, new_u_seq, new_v_seq, new_p_seq
