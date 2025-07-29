from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

__all__= ['GMM']
 
class GMM(nn.Module):
    def __init__(self, M, hidden_size, output_size, device = 'cpu', **kwargs) -> None:
        torch.set_default_device(device)
        super(GMM, self).__init__()
        self.activation1 = nn.Tanh()
        self.mu = nn.Linear(hidden_size, output_size * M, device=device)
        
        # Sequential of Linear(hidden, M, device) and Exp on last shape
        self.sigma = nn.Linear(hidden_size, M, device=device)
        
        # Sequential of Linear(hidden, M, device) and Softmax on last shape
        self.pi = nn.Sequential(
            nn.Linear(hidden_size, M, device=device),
            nn.Softmax(dim=-1)
        )
        
        self.M = M
        self.device = device
    
    def __call__(self, X):
        mu = self.mu(X)
        sigma = torch.exp(self.sigma(X))
        pi = self.pi(X)
        return mu, sigma, pi
    
    @staticmethod
    def sample(mu, sigmas, pis):
        mu = mu.reshape(mu.shape[0], mu.shape[1], sigmas.shape[-1], mu.shape[-1] // sigmas.shape[-1])
        sigmas = sigmas.unsqueeze(-1)
        pis = pis.unsqueeze(-1)
        
        dist = torch.distributions.Normal(mu, torch.sqrt(sigmas))
        
        out = pis * dist.sample()
        out = torch.sum(out, axis=-2)
        return out
    
    @staticmethod
    def get_loss():
        return loss_fn_vectorized

def gaussian_pdf_vectorized(x, mu, sigma, alpha):
    """Compute the Gaussian PDF using torch.distributions.Normal (vectorized)."""
    sigma = torch.clamp(sigma, min=1e-8)  # Ensure numerical stability for sigma
    coeff = alpha / (sigma * torch.sqrt(torch.tensor(2. * np.pi)))
    
    exponent = -torch.sum((mu - x) ** 2, dim=-1, keepdim=True) / (2 * sigma ** 2)
    likelihood = coeff * torch.exp(exponent)
    
    return likelihood

def loss_fn_vectorized(target, mu, sigma, pi):
    """Vectorized mixture density loss function with torch.distributions."""
    # Reshape the mixture components
    batch_size = target.size(0)
    step_size = target.size(1)
    n_f = target.size(2)
    M = sigma.shape[-1]
    
    # Shape of mu: (batch_size, step_size, n_f, k)
    mu = mu.view(batch_size, step_size, M, n_f)
    
    # Shape of sigma: (batch_size, k)
    sigma = torch.sqrt(sigma)  # convert variance to standard deviation
    
    # Target shape: (batch_size, step_size, n_f, 1) to broadcast across k components
    target = target.unsqueeze(-2)
    
    # Compute Gaussian likelihood for each component (batch_size, step_size, n_f, k)
    likelihood_z_x = gaussian_pdf_vectorized(target, mu, sigma.unsqueeze(-1), pi.unsqueeze(-1))
    
    loss = torch.mean(-torch.log(likelihood_z_x + 1e-8))  # add a small constant for numerical stability
    
    return loss