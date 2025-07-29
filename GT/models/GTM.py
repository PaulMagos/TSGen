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
    def __init__(self, input_size, output_size, hidden_size, mixture_dim, window, horizon, dropout, num_layers, bidirectional, lr, weight_decay, callbacks, device, exo_size=0) -> None:
        super(GTM, self).__init__()
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_size+exo_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional, batch_first=True)
        self.gmm = GMM(mixture_dim, hidden_size*(2 if bidirectional else 1), output_size, device = device)
        self.input_size = input_size
        self.loss = GMM.get_loss()
        self.device = device
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.callbacks = {}
        if 'EarlyStopping' in callbacks:
            self.callbacks['EarlyStopping'] = EarlyStopping()
        self.input_size = input_size
            
        self.horizon = horizon
        self.window = window
    
    def forward(self, x):
        out = self.lstm(x)[0]
        mu, sigma, pi = self.gmm(out)
        return mu, sigma, pi

    def train_step(self, train_data, exo_var=None, batch_size=1, epochs = 1):
        train_data, val_data, exo_var = self.parse_data(train_data, exo_var)
            
        self.train()
        print("Starting training...")
        history = {'loss': []}
        batches = len(train_data)
        
        for epoch in range(1, epochs + 1):
            losses_epoch = []
            
            with tqdm(total=batches) as pbar:
                for batch in range(0, batches, batch_size):
                    inputs = train_data[batch:batch+batch_size]
                    check = val_data[batch:batch+batch_size]
                    
                    mu, sigma, pi = self(inputs)
                    loss = self.loss(check, mu, sigma, pi)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses_epoch.append(loss.item())
                    mean_loss = np.mean(losses_epoch)
                    pbar.set_description(f"Loss {mean_loss}")
                    pbar.update(batch_size)

            mean_loss = np.mean(losses_epoch)
            history['loss'].append(mean_loss)
            print(f'Epoch {epoch} - loss:', mean_loss)
            stop, min_loss = self.callbacks['EarlyStopping'](self, mean_loss)
            if stop:
                print(f'Early Stopped at epoch {epoch} with loss {min_loss}')
                break
        return self, history

    def predict_step(self, data, mask=None, exo_var=None):
        M = self.gmm.M
        D = data.shape[-1]
        self.eval()
        data = data.to(self.device)
        exo_var = exo_var.to(self.device) if exo_var is not None else None
        mask = mask.to(self.device) if mask is not None else None
        
        window = self.window
        horizon = self.horizon
        output = data[:, :horizon]
        with tqdm(total=data.shape[1]-horizon) as pbar:
            for i in range(horizon, data.shape[1], horizon):
                inputs = data[:, :i]
                
                if exo_var is not None:
                    inputs = torch.cat([exo_var[:, :i], inputs], dim=-1) 
                
                mu, sigma, pi = self(inputs)
                
                pred = GMM.sample(mu, sigma, pi).to(self.device)
                if mask is not None:
                    pred[:, -horizon:] = torch.where(mask[:, i:i+horizon]==0., data[:, i:i+horizon], pred[:, -horizon:])
                
                output = torch.concat([output, pred[:, -horizon:]], axis=1)
                pbar.update(horizon)
        
        return np.array(output.cpu().detach())
    
    def generate_step(self, shape: tuple, exo_var=None):
        self.eval()
        
        num_timeseries = shape[0]
        steps = shape[1]
        
        shape = shape
        
        window = self.window
        horizon = self.horizon
        
        exo_var = exo_var.to(self.device) if exo_var is not None else None
        input_shape = (num_timeseries, window, self.input_size + exo_var.shape[-1] if exo_var is not None else self.input_size)
        mu, sigma, pi = self(torch.rand(input_shape).to(self.device))
        inputs = GMM.sample(mu, sigma, pi)
       
        output = None
        with tqdm(total=steps//horizon) as pbar:
            for i in range(steps//horizon):
                if exo_var is not None:
                    inputs = torch.cat([exo_var[:, i:window + i], inputs], dim=-1)
                
                mu, sigma, pi = self(inputs)
                
                pred = GMM.sample(mu, sigma, pi).to(self.device)
                
                output = torch.concat([output, pred[:, -horizon:]], axis=1) if output is not None else pred[:, -horizon:]
                inputs = pred[:, -window:]
                pbar.update(1)
        return np.array(output.cpu().detach())

    def parse_data(self, train_data, exo_var=None):
        
        window = self.window
        horizon = self.horizon
        device = self.device
        train_data = torch.Tensor(train_data)
        train_data = train_data.reshape(train_data.shape[0]*train_data.shape[1], train_data.shape[2])
        
        val_data = torch.Tensor(train_data)
        
        
        max_size = (train_data.shape[0] - horizon) // window
        val_data = train_data[horizon:(window*max_size)+horizon]
        train_data = train_data[:len(val_data)]
        
        train_data = train_data.reshape(train_data.shape[0]//window, window, train_data.shape[1]).to(device)
        val_data  = val_data.reshape(val_data.shape[0]//window, window, val_data.shape[1]).to(device)
    
        if exo_var is not None:
            exo_var = torch.Tensor(exo_var)
            exo_var = exo_var.reshape(exo_var.shape[0]*exo_var.shape[1], exo_var.shape[2])
            exo_var = exo_var[:window*max_size]
            exo_var = exo_var.reshape(exo_var.shape[0]//window, window, exo_var.shape[1]).to(device)
            train_data = torch.cat([exo_var, train_data], dim=-1)
        
        return train_data, val_data, exo_var