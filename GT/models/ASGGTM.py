from torch import nn
from .GMM import GMM
import torch.optim as optim
from .EarlyStopping import EarlyStopping
import numpy as np
import torch
from tsl.nn.layers.graph_convs import DiffConv
from tqdm import tqdm
import torch.nn.functional as F
from tsl.ops.connectivity import adj_to_edge_index
from vector_vis_graph import WeightMethod, horizontal_vvg, natural_vvg

__all__ = ['ASGGTM']

class ASGGTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, mixture_dim, window, horizon, dropout, num_layers, bidirectional, lr, weight_decay, callbacks, device, emb_size, exo_size=0) -> None:
        super(ASGGTM, self).__init__()
        self.tempo_diff_conv = DiffConv(input_size+exo_size, hidden_size, 2, root_weight=False).to(device)
        self.spatio_diff_conv = DiffConv(window, window, 2, root_weight=False).to(device)
        self.N1 = nn.Parameter(torch.randn(input_size, emb_size).to(device), requires_grad = True).to(device)
        self.N2 = nn.Parameter(torch.randn(emb_size, input_size).to(device), requires_grad = True).to(device)
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=(input_size+exo_size)*2+hidden_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional, batch_first=True)
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
    
    def get_adj(self, ts):
        nvvg_adj = torch.tensor(natural_vvg(torch.tensor(ts).cpu().numpy(), weight_method=WeightMethod.TIME_DIFF_EUCLIDEAN_DISTANCE, directed=True))
        edge_index, edge_weights = adj_to_edge_index(nvvg_adj)
        return edge_index.to(self.device), edge_weights.float().to(self.device)
                
    def forward(self, x, exo_var=None, temporal_edge_i=None, temporal_edge_w=None):
        if exo_var is not None:
            x_in = torch.cat([exo_var, x], dim=-1).to(self.device)
        else:                                                                                                                                                                                                                                     
            x_in = x.to(self.device)
        
        diff_tempo = torch.Tensor().to(self.device)
        
        for i in range(len(x)):
            edge_i, edge_w = self.get_adj(x[i]) if temporal_edge_i is None else (temporal_edge_i[i], temporal_edge_w[i])
            res = self.tempo_diff_conv(x_in[i], edge_i, edge_w).unsqueeze(0).to(self.device)
            diff_tempo = torch.cat([diff_tempo, res], dim=0)
            
        adp = F.softmax(F.relu(torch.mm(self.N1, self.N2)), dim=1).to(self.device)
        spatio_edge_index, spatio_edge_weight = adj_to_edge_index(adp)
        diff_spatio = self.spatio_diff_conv(x_in.permute(0, 2, 1), spatio_edge_index, spatio_edge_weight).permute(0, 2, 1).to(self.device)
        
        x_in = torch.cat([diff_tempo, diff_spatio, x_in], dim=-1)
        
        out = self.lstm(x_in)[0]

        gmm = self.gmm(out)
        return gmm

    def train_step(self, train_data, exo_var=None, batch_size=32, epochs = 1):
        train_data, val_data, exo_var = self.parse_data(train_data, exo_var)
            
        self.train()
        print("Starting training...")
        history = {'loss': []}
        batches = len(train_data)
        
        edge_index, edge_weights = [], []
        for i in range(batches):
            edge_i, edge_w = self.get_adj(train_data[i])
            edge_index.append(edge_i)
            edge_weights.append(edge_w)
        
        for epoch in range(1, epochs + 1):
            losses_epoch = []
            
            with tqdm(total=batches) as pbar:
                for batch in range(0, batches, batch_size):
                    inputs = train_data[batch:batch+batch_size]
                    check = val_data[batch:batch+batch_size]
                    exo = exo_var[batch:batch+batch_size] if exo_var is not None else None
                    
                    mu, sigma, pi = self(inputs, exo, edge_index[batch:batch+batch_size], edge_weights[batch:batch+batch_size])
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
        output = data[:, :window]
        
        with tqdm(total=data.shape[1]-window) as pbar:
            for i in range(window, data.shape[1], horizon):
                inputs = data[:, i-window if i > window else 0:i]
                
                mu, sigma, pi = self(inputs, exo_var[:, i-window if i > window else 0:i] if exo_var is not None else None)
                
                pred = GMM.sample(mu, sigma, pi).to(self.device)
                if mask is not None:
                    pred[:, -horizon:] = torch.where(mask[:, i:i+horizon]==0., data[:, i:i+horizon], pred[:, -horizon:])
                output = torch.concat([output, pred[:, -horizon:]], axis=1)
                pbar.update(horizon)
        
        return np.array(output.cpu().detach())

    def generate_step(self, shape: tuple, exo_var=None):
        self.eval()
        
        num_timeseries = shape[0]
        
        shape = shape
        
        window = self.window
        horizon = self.horizon
        exo_var = exo_var.to(self.device) if exo_var is not None else None
        
        steps = shape[1]
        
        input_shape = (num_timeseries, window, self.input_size)
        exo_shape = (num_timeseries, window, exo_var.shape[-1]) if exo_var is not None else self.input_size
        
        exo = torch.rand(exo_shape) if exo_var is not None else None
        
        mu, sigma, pi = self(torch.rand(input_shape), exo)
        inputs = GMM.sample(mu, sigma, pi)
        
        output = None
        with tqdm(total=steps//horizon) as pbar:
            for i in range(steps//horizon):
                mu, sigma, pi = self(inputs, exo_var[:, i:window + i] if exo_var is not None else None)
                
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
    
        idx = torch.randperm(train_data.size(0)).to(train_data.device).to(device)
        train_data = train_data[idx].to(device)
        val_data  = val_data[idx].to(device)
    
        if exo_var is not None:
            exo_var = torch.Tensor(exo_var)
            exo_var = exo_var.reshape(exo_var.shape[0]*exo_var.shape[1], exo_var.shape[2])
            exo_var = exo_var[:window*max_size]
            exo_var = exo_var.reshape(exo_var.shape[0]//window, window, exo_var.shape[1]).to(device)
            exo_var = exo_var[idx]

        
        return train_data, val_data, exo_var