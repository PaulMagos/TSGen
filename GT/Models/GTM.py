from torch import nn
from .GMM import GMM
import torch.optim as optim
from .EarlyStopping import EarlyStopping
import numpy as np
import torch
from tqdm import tqdm
import random
torch.autograd.set_detect_anomaly(True)

__all__ = ['GTM']

class GTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, mixture_dim, dropout, num_layers, bidirectional, lr, weight_decay, callbacks, device, exo_size=0) -> None:
        super(GTM, self).__init__()
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_size+exo_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional)
        self.gmm = GMM(mixture_dim, hidden_size*(2 if bidirectional else 1), output_size, device = device)
        self.input_size = input_size
        self.loss = GMM.get_loss()
        self.device = device
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.callbacks = {}
        if 'EarlyStopping' in callbacks:
            self.callbacks['EarlyStopping'] = EarlyStopping()
            
        self.horizon = 1
        self.window = 1
    
    def forward(self, x):
        out = self.lstm(x)[0]
        gmm = self.gmm(out)
        return gmm

    def train_step(self, train_data, exo_var=None, window=1, horizon=1, epochs = 1):
        train_data = train_data.to(self.device)
        val_data = train_data
        if exo_var is not None:
            train_data = torch.cat([exo_var, train_data], dim=-1)
            
        self.train()
        print("Starting training...")
        history = {'loss': []}
        windows = (train_data.shape[1]-horizon) // window
        batches = len(train_data)
        
        self.horizon = horizon
        self.window = window
        
        for epoch in range(1, epochs + 1):
            losses_epoch = []
            
            with tqdm(total=batches) as pbar:
                for batch in range(batches):
                    for i in range(windows):
                        batch_idx = i * window
                        
                        inputs = train_data[batch:batch+1, batch_idx:batch_idx+window, :]
                        check = val_data[batch:batch+1, batch_idx+horizon:batch_idx+window+horizon, :]
                        
                        mu, sigma, pi = self(inputs)
                        loss = self.loss(check, mu, sigma, pi)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        losses_epoch.append(loss.item())
                        mean_loss = np.mean(losses_epoch)
                    pbar.set_description(f"Loss {mean_loss}")
                    pbar.update(1)

            mean_loss = np.mean(losses_epoch)
            history['loss'].append(mean_loss)
            print(f'Epoch {epoch} - loss:', mean_loss)
            if self.callbacks['EarlyStopping'](self, mean_loss):
                print(f'Early Stopped at epoch {epoch} with loss {mean_loss}')
                break
        return self, history

    def predict_step(self, data, exo_var=None, start = 0, steps = 7):
        M = self.gmm.M
        D = data.shape[-1]
        self.eval()
        output = torch.Tensor().to(self.device)
        data = data.to(self.device)
        
        with tqdm(total=steps) as pbar:
            for i in range(start, start+steps):
                inputs = data[:, :i+1]
                
                if exo_var is not None:
                    inputs = torch.cat([exo_var[:, :i+1], inputs], dim=-1) 
                
                mu, sigma, pi = self(inputs)
                
                means = mu.reshape(data.shape[0], i+1, M, D)
                stds = sigma.unsqueeze(-1)
                gmm_weights = pi.unsqueeze(-1)
                
                pred = GMM.sample(means, stds, gmm_weights)
                pred = torch.sum(pred, axis=-2).to(self.device)
                
                output = torch.concat([output, pred[:, i:i+1]], axis=1)
                pbar.update(1)
        
        return np.array(output.cpu().detach())

    def generate_step(self, shape: tuple, exo_var=None, window=None, horizon=None):
        M = self.gmm.M
        
        num_timeseries = shape[0]
        steps = shape[1]
        D = shape[2]
        
        shape = shape
        
        if window is None:
            window = self.window
        if horizon is None:
            horizon = self.horizon
        
        input_shape = (num_timeseries, window, D)
       
        self.eval()
            
        input = torch.ones(input_shape)
        output = None
        
        with tqdm(total=steps//horizon) as pbar:
            for i in range(steps//horizon):
                if exo_var is not None:
                    input = torch.cat([exo_var[:, i*horizon:window + i*horizon], input], dim=-1)
                
                mu, sigma, pi = self(input)
                
                means = mu.reshape(num_timeseries, mu.shape[1], M, D)
                stds = sigma.unsqueeze(-1)
                gmm_weights = pi.unsqueeze(-1)

                pred = GMM.sample(means, stds, gmm_weights)
                pred = torch.sum(pred, axis=-2).to(self.device)
                
                output = torch.concat([output, pred[:, -horizon:]], axis=1) if output is not None else pred[:, -horizon:]
                input = pred[:, -window:]
                pbar.update(1)
        return np.array(output.cpu().detach())
