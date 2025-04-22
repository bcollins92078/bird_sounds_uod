"""
An implementation of VaDE(https://arxiv.org/pdf/1611.05148.pdf).

03-22-2023: modified by replacing the MLP DNN with a CNN
"""                                                                
import math
import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

### set default floating point precision to double
#torch.set_default_dtype(torch.float64)
    


def _reparameterize(mu, logvar):
    """Reparameterization trick.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z


class VaDE_CNN(torch.nn.Module):
    """Variational Deep Embedding(VaDE).

    Args:
        n_classes (int): Number of clusters.
        data_dim (int): Dimension of observed data.
        latent_dim (int): Dimension of latent space.
    """
    def __init__(self, n_classes, img_dim1, img_dim2, latent_dim):
        super(VaDE_CNN, self).__init__()

        self._pi = Parameter(torch.zeros(n_classes))
        self.mu = Parameter(torch.randn(n_classes, latent_dim))
        self.logvar = Parameter(torch.randn(n_classes, latent_dim))

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(int(img_dim1/4)*int(img_dim2/4)*32, 128),
            #torch.nn.Linear(8*10*32, 128),
            torch.nn.ReLU(),
        )
        self.encoder_mu = torch.nn.Linear(128, latent_dim)
        self.encoder_logvar = torch.nn.Linear(128, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, int(img_dim1/4)*int(img_dim2/4)*32),
            #torch.nn.Linear(128, 8*10*32),
            torch.nn.Unflatten(dim=1, unflattened_size=(32, int(img_dim1/4), int(img_dim2/4))),
            #torch.nn.Unflatten(dim=1, unflattened_size=(32, 8, 10)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1, output_padding=0),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid(),
        )

    @property
    def weights(self):
        return torch.softmax(self._pi, dim=0)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = _reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def classify(self, x, n_samples=8):
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = torch.stack(
                [_reparameterize(mu, logvar) for _ in range(n_samples)], dim=1)
            z = z.unsqueeze(2)
            h = z - self.mu
            h = torch.exp(-0.5 * torch.sum(h * h / self.logvar.exp(), dim=3))
            # Same as `torch.sqrt(torch.prod(self.logvar.exp(), dim=1))`
            h = h / torch.sum(0.5 * self.logvar, dim=1).exp()
            p_z_given_c = h / (2 * math.pi)
            p_z_c = p_z_given_c * self.weights
            y = p_z_c / torch.sum(p_z_c, dim=2, keepdim=True)
            y = torch.sum(y, dim=1)
            pred = torch.argmax(y, dim=1)
        return pred


def lossfun0(model, x, recon_x, mu, logvar):
    batch_size = x.size(0)

    # Compute gamma ( q(c|x) )
    z = _reparameterize(mu, logvar).unsqueeze(1)
    h = z - model.mu
    h = torch.exp(-0.5 * torch.sum((h * h / model.logvar.exp()), dim=2))
    # Same as `torch.sqrt(torch.prod(model.logvar.exp(), dim=1))`
    h = h / torch.sum(0.5 * model.logvar, dim=1).exp()
    p_z_given_c = h / (2 * math.pi)
    p_z_c = p_z_given_c * model.weights
    gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)

    h = logvar.exp().unsqueeze(1) + (mu.unsqueeze(1) - model.mu).pow(2)
    h = torch.sum(model.logvar + h / model.logvar.exp(), dim=2)
    loss = F.binary_cross_entropy(recon_x, x, reduction='sum') \
        + 0.5 * torch.sum(gamma * h) \
        - torch.sum(gamma * torch.log(model.weights + 1e-9)) \
        + torch.sum(gamma * torch.log(gamma + 1e-9)) \
        - 0.5 * torch.sum(1 + logvar)
    loss = loss / batch_size
    return loss

def cluster_probs(z, mu_priors, logvar_priors):
    """
    Calculate log probability of data point belonging to each 
    gaussian distribution ( p(z|c) ). Formula follows the log of the
    gaussian PDF, simplified.
    Parameters
    ----------
    latent : Tensor
        Sampled latent vector (batch, z_dim)
    mu_priors  : Tensor
        GMM means (means for each cluster - (clusters, z_dim))
    logvar_priors: Tensor
        GMM logvars (logvars for each cluster - (clusters, z_dim))
        
    ----------
    Returns
    ----------
    Tensor
        Stacked tensors containing probabilities for each data point
        in the batch belonging in a cluster, for each cluster (batch, clusters)
    ----------
    """
    log_probs = []

    # iterate over each cluster
    for (mean, log_var) in zip(mu_priors, logvar_priors):
        log_prob = torch.pow(z - mean, 2)
        log_prob += log_var
        log_prob += np.log(2 * np.pi)
        log_prob /= torch.exp(log_var)
        log_prob = -0.5 * torch.sum(log_prob, dim=1)

        log_probs.append(log_prob.view(-1, 1))

    cat_shape = torch.cat(log_probs, 1)
    # print(cat_shape.shape)  # [batch, clusters]
    return cat_shape
    
"""
function: lossfun

This is a modified version of the loss function in the original download.
See lossfun0 above for the original version. Changes in this version are
* compute ln p(z|c) instead of p(z:c) directly to avoid some of the NaN loss issues
* returned the reconstruction loss and regularizer (KL) terms separately

"""
def lossfun(model, x, recon_x, mu, logvar):
    batch_size = x.size(0)
    mu_priors = model.mu
    logvar_priors = model.logvar
    pi_priors = model.weights
    
    # Compute gamma ( q(c|x) )
    z = _reparameterize(mu, logvar)
    
    log_p_z_given_c = cluster_probs(z, mu_priors, logvar_priors)
    
    gamma = torch.exp(torch.log(pi_priors.unsqueeze(0)) + log_p_z_given_c) + 1e-10
    gamma = gamma / torch.sum(gamma, dim=1, keepdim=True)

    h = logvar.exp().unsqueeze(1) + (mu.unsqueeze(1) - mu_priors).pow(2)
    h = torch.sum(logvar_priors + h / logvar_priors.exp(), dim=2)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_term = 0.5 * torch.sum(gamma * h) \
        - torch.sum(gamma * torch.log(model.weights + 1e-9)) \
        + torch.sum(gamma * torch.log(gamma + 1e-9)) \
        - 0.5 * torch.sum(1 + logvar)
    loss = (recon_loss + kl_term) / batch_size
    return loss, recon_loss / batch_size, kl_term / batch_size
    

class AutoEncoderForPretrain(torch.nn.Module):
    """Auto-Encoder for pretraining VaDE.

    Args:
        data_dim (int): Dimension of observed data.
        latent_dim (int): Dimension of latent space.
    """
    def __init__(self, img_dim1, img_dim2, latent_dim):
        super(AutoEncoderForPretrain, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(int(img_dim1/4)*int(img_dim2/4)*32, 128),
            torch.nn.ReLU(),
        )
        self.encoder_mu = torch.nn.Linear(128, latent_dim)
        self.encoder_logvar = torch.nn.Linear(128, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, int(img_dim1/4)*int(img_dim2/4)*32),
            torch.nn.Unflatten(dim=1, unflattened_size=(32, int(img_dim1/4), int(img_dim2/4))),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1, output_padding=0),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder_mu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x
