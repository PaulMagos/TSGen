from torch import nn
import torch.optim as optim
from .EarlyStopping import EarlyStopping
import numpy as np
import torch
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)

__all__ = ['GTLSTM']

class GTLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, weight_decay, window, horizon, dropout, num_layers, bidirectional, loss, lr, callbacks, device, exo_size=0) -> None:
        super(GTLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size+exo_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional, batch_first=True)
        self.activation1 = nn.Tanh()
        self.out_activation = nn.Tanh()
        self.dense = nn.Linear(in_features=hidden_size*(2 if bidirectional else 1), out_features=output_size, device=device)
        self.optimizer = optim.SGD(self.parameters(), weight_decay=weight_decay, lr=lr, momentum=0.3)
        self.callbacks = {}
        if 'EarlyStopping' in callbacks:
            self.callbacks['EarlyStopping'] = EarlyStopping()
        self.device = device
        match(loss):
            case 'L1Loss':
                self.loss = nn.L1Loss(reduction='mean')
                self.metrics = nn.MSELoss()
            case 'mse':
                self.loss = nn.MSELoss()
                self.metrics = nn.L1Loss(reduction='mean')
                
        self.window = window
        self.horizon = horizon

    def forward(self, x, exo_var=None):
        if exo_var is not None:
            x_in = torch.cat([exo_var, x], dim=-1)
        else:                                                                                                                                                                                                                                     
            x_in = x
        out = self.lstm(x_in)[0]
        activation1 = self.activation1(out)
        dense = self.dense(activation1)
        out_activation = self.out_activation(dense)
        return out_activation

    def train_step(self, train_data, exo_var=None, batch_size=1, epochs = 1):
        self.train()
        train_data, val_data, exo_var = self.parse_data(train_data, exo_var)

        print("Starting training...")
        history = {'loss': [], 'MAE': []}
        for epoch in range(1, epochs + 1):
            losses_epoch = []
            metrics_epoch = []
            with tqdm(total=len(train_data)//batch_size) as pbar:
                for i in range(0, len(train_data), batch_size):
                    self.optimizer.zero_grad()
                    outputs = self(train_data[i:i+batch_size, :, :].to(self.device), exo_var[i:i+batch_size] if exo_var is not None else None)
                    loss = self.loss(val_data[i:i+batch_size, :, :].to(self.device), outputs)
                    metric = self.metrics(val_data[i:i+batch_size, :, :].to(self.device), outputs)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=20)
                    self.optimizer.step()
                    losses_epoch.append(loss.item())
                    metrics_epoch.append(metric.item())
                    mean_loss = np.mean(losses_epoch)
                    mean_metric = np.mean(metrics_epoch)
                    pbar.set_description(f"Loss {mean_loss}, MAE : {mean_metric}")
                    pbar.update(batch_size)
            mean_loss = np.mean(losses_epoch)
            mean_metric = np.mean(metrics_epoch)
            history['loss'].append(mean_loss)
            history['MAE'].append(mean_metric)
            print(f'Epoch {epoch} - MSE: {mean_loss} - MAE: {mean_metric}')
            stop, min_loss = self.callbacks['EarlyStopping'](self, mean_loss)
            if stop:
                print(f'Early Stopped at epoch {epoch} with loss {min_loss}')
                break

        return self, history
    
    def predict_step(self, data, mask=None, exo_var=None):
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
                pred = self(inputs, exo_var[:, :i] if exo_var is not None else None)
                if mask is not None:
                    pred[:, -horizon:] = torch.where(mask[:, i:i+horizon]==0., data[:, i:i+horizon], pred[:, -horizon:])
                output = torch.concat([output, pred[:, -horizon:]], axis=1)
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
        
        return train_data, val_data, exo_var