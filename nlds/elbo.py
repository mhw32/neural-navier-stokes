from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn.functional as F

from utils import (
    unit_gaussian_log_pdf,
    gaussian_log_pdf,
    bernoulli_log_pdf,
    categorical_log_pdf,
    gumbel_softmax,
    log_mean_exp,
)


def evidence_lower_bound(data, output):
    r"""Lower bound on log p(y).

    @param data: this is the input data
    @param output: this is the output of RSSNLDS.forward(.)
    """
    elbo = 0
    batch_size, T, _ = data.size()

    for t in xrange(T):
        log_p_yt_given_xt = bernoulli_log_pdf(data[:, t, :], output['y_emission_probs'][:, t, :])
        # even though we use ST-gumbel softmax, it seems the original paper just uses Categorical KL
        log_p_xt_given_zt = gaussian_log_pdf(output['q_x'][:, t, :], output['x_emission_mu'][:, t, :],
                                             output['x_emission_logvar'][:, t, :])

        if t == 0:
            log_p_z1 = ... 
            log_q_x1_given_y = ... 
            log_q_z1_given_x_K = ... 

            elbo_i = log_p_yt_given_xt + log_p_xt_given_zt + log_p_z1 - \
                     log_q_x1_given_y - log_q_z1_given_x_K

        else:
            log_p_zt_given_zt1_xt1 = ...
            log_q_xt_given_xt1_y = ...
            log_q_zt_given_zt1_x_K = ...

            elbo_i = log_p_yt_given_xt + log_p_xt_given_zt + log_p_zt_given_zt1_xt1 - \
                     log_q_xt_given_xt1_y - log_q_zt_given_zt1_x_K

        elbo += elbo_i
    
    elbo = torch.mean(elbo)
    elbo = -elbo  # turn into a minimization problem

    return elbo
