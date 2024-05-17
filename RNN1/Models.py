from GMM import GMM
from torch import nn
import numpy as np
import torch
from tqdm import tqdm
from EarlyStopping import EarlyStopping
import torch.optim as optim
# import wandb

torch.autograd.set_detect_anomaly(True)
__all__ = ['GTM', 'GTLSTM', 'GTR']

class GTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, mixture_dim, dropout, num_layers, bidirectional, loss, lr, weight_decay, callbacks, device, debug) -> None:
        super(GTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional)
        self.dense = nn.Linear(in_features=hidden_size*(2 if bidirectional else 1), out_features=(output_size+2)*mixture_dim, device=device)
        self.gmm = GMM(M = mixture_dim, device = device, debug=debug)
        self.loss = loss
        self.device = device
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.callbacks = {}
        if 'EarlyStopping' in callbacks:
            self.callbacks['EarlyStopping'] = EarlyStopping()

    def forward(self, x):
        return self.gmm(self.dense(self.lstm(x)[0]))

    def train_step(self, train_data, train_label=None, batch_size=1, epochs = 1, step_log=500):
        # wandb.init(project='MCS_Thesis')
        if train_label==None:
            train_label = train_data[1:]
            train_data = train_data[:-1]
        train_label = torch.Tensor(train_label).type(torch.float32)
        self.train()
        print("Starting training...")
        for epoch in range(1, epochs + 1):
            losses_epoch = []
            with tqdm(total=len(train_data) - 1) as pbar:
                for i in range(len(train_data) - 1):
                    self.optimizer.zero_grad()
                    outputs = self(train_data[i:i+batch_size, :, :].to(self.device))
                    loss = self.loss(train_label[i:i+batch_size, :, :].to(self.device), outputs)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                    self.optimizer.step()
                    losses_epoch.append(loss.item())
                    if i%step_log == 0:
                        mean_loss = np.mean(losses_epoch)
                        # wandb.log({"loss": mean_loss})
                        pbar.set_description(f"Loss {mean_loss}")
                    pbar.update(1)

            mean_loss = np.mean(losses_epoch)
            print(f'Epoch {epoch} - loss:', mean_loss)
            # wandb.log({f'Epoch - loss:': mean_loss})
            if self.callbacks['EarlyStopping'](self, mean_loss):
                print(f'Early Stopped at epoch {epoch} with loss {mean_loss}')
                break

        return self, losses_epoch

    def predict_step(self, data, start = 0, steps = 7):
        M = self.gmm.M
        D = data.shape[-1]
        self.eval()
        output = []
        with tqdm(total=steps) as pbar:
            for i in range(start, start+steps):
                data_ = self(data[i:i+1, :, :].to(self.device))
                means = data_[:M*D].cpu().detach().numpy()
                stds = data_[:, M*D : M*(D+1)].cpu().detach().numpy()
                gmm_weights = data_[:, M*(D+1):].cpu().detach().numpy()
                
                means = means.reshape(M, D)
                stds = stds[:, np.newaxis]
                gmm_weights = gmm_weights[:, np.newaxis]
                
                normal = np.random.normal(means, stds)
                pred = gmm_weights * normal
                pred = np.sum(pred, axis=0)#[:, np.newaxis]
                output.append(pred)
                pbar.update(1)
        return np.array(output)


class GTLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, num_layers, bidirectional, loss, lr, callbacks, device) -> None:
        super(GTLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional)
        self.activation1 = nn.Tanh()
        self.activation2 = nn.Tanh()
        self.dense = nn.Linear(in_features=hidden_size*(2 if bidirectional else 1), out_features=output_size, device=device)
        # self.Relu = nn.Linear(output_size, output_size)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.3)
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

    def forward(self, x):
        return self.activation2(self.dense(self.activation1(self.lstm(x)[0])))
        # return self.dense(self.activation1(self.lstm(x)[0]))

    def train_step(self, train_data, train_label=None, batch_size=1, epochs = 1, step_log=500):
        self.train()
        if train_label==None:
            train_label = train_data[1:]
            train_data = train_data[:-1]
        train_label = torch.Tensor(train_label).type(torch.float32)
        print("Starting training...")
        history = {'loss': [], 'MAE': []}
        for epoch in range(1, epochs + 1):
            losses_epoch = []
            metrics_epoch = []
            with tqdm(total=len(train_data)//batch_size+1) as pbar:
                for i in range(0, len(train_data), batch_size):
                    self.optimizer.zero_grad()
                    outputs = self(train_data[i:i+batch_size, :, :].to(self.device))
                    loss = self.loss(train_label[i:i+batch_size, :, :].to(self.device), outputs)
                    metric = self.metrics(train_label[i:i+batch_size, :, :].to(self.device), outputs)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=20)
                    self.optimizer.step()
                    losses_epoch.append(loss.item())
                    metrics_epoch.append(metric.item())
                    if i%step_log == 0:
                        mean_loss = np.mean(losses_epoch)
                        mean_metric = np.mean(metrics_epoch)
                        pbar.set_description(f"Loss {mean_loss}, MAE : {mean_metric}")
                    pbar.update(1)

            mean_loss = np.mean(losses_epoch)
            mean_metric = np.mean(metrics_epoch)
            history['loss'].append(mean_loss)
            history['MAE'].append(mean_metric)
            print(f'Epoch {epoch} - MSE: {mean_loss} - MAE: {mean_metric}')
            # wandb.log({'Epoch - loss:': mean_loss, 'Epoch - MSE:': mean_metric})
            if self.callbacks['EarlyStopping'](self, mean_loss):
                print(f'Early Stopped at epoch {epoch} with loss {mean_loss}')
                break

        return self, history
    
    def predict_step(self, data, start = 0, steps = 7):
        self.eval()
        data_ = data[start:start+steps-1, :].to(self.device)
        tmp_out = self(data_).cpu().detach().numpy()
        return tmp_out
    
    def generate_step(self, data, start = 0, steps = 7):
        self.eval()
        output = data[start, :].to(self.device)
        for i in range(steps):
            tmp_out = self(output)[-1, :].reshape(1, -1)
            output = torch.concatenate([output, tmp_out])
        return output.cpu().detach().numpy()

class GTR(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, num_layers, bidirectional, lr, callbacks, device) -> None:
        super(GTR, self).__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional)
        self.dense = nn.Linear(in_features=hidden_size*(2 if bidirectional else 1), out_features=output_size, device=device)
        self.activation1 = nn.Tanh()
        self.activation2 = nn.Tanh()
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.3)
        self.callbacks = {}
        if 'EarlyStopping' in callbacks:
            self.callbacks['EarlyStopping'] = EarlyStopping()
        self.device = device
        self.loss = nn.MSELoss()
        self.metrics = nn.L1Loss()

    def forward(self, x):
        out, hidden = self.rnn(x)
        return self.activation2(self.dense(self.activation1(out)))

    def train_step(self, train_data, train_label=None, batch_size=1, epochs = 1, step_log=500):
        hidden = None
        # hidden = torch.Tensor(self.num_layers*(2 if self.bidirectional else 1), 1, self.hidden_size).to(self.device)
        self.train()
        if train_label==None:
            train_label = train_data[1:]
            train_data = train_data[:-1]
        train_label = torch.Tensor(train_label).type(torch.float32)
        history = {'loss': [], 'MEE': []}
        print("Starting training...")
        for epoch in range(1, epochs + 1):
            losses_epoch = []
            metrics_epoch = []
            with tqdm(total=len(train_data)//batch_size+1) as pbar:
                for i in range(0, len(train_data), batch_size):
                    window =  i+batch_size-32 if (i+batch_size - 32 >0) else 0
                    outputs = self(train_data[window:i+batch_size, :, :].to(self.device))
                    loss = self.loss(train_label[window:i+batch_size, :, :].to(self.device), outputs)
                    metric = self.metrics(train_label[window:i+batch_size, :, :].to(self.device), outputs)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    losses_epoch.append(loss.item())
                    metrics_epoch.append(metric.item())
                    if i%step_log == 0:
                        mean_loss = np.mean(losses_epoch)
                        mean_metric = np.mean(metrics_epoch)
                        # wandb.log({"loss": mean_loss, 'MSE': mean_metric})
                        pbar.set_description(f"Loss {mean_loss}, MEE : {mean_metric}")
                    pbar.update(1)
                    
            mean_loss = np.mean(losses_epoch)
            mean_metric = np.mean(metrics_epoch)
            history['loss'].append(mean_loss)
            history['MEE'].append(mean_metric)
            print(f'Epoch {epoch} - MSE: {mean_loss} - MEE: {mean_metric}')
            if self.callbacks['EarlyStopping'](self, mean_loss):
                print(f'Early Stopped at epoch {epoch} with loss {mean_loss}')
                break

        return self, history

    # def predict_step(self, data, start = 0, steps = 7):
    #     self.eval()
    #     data_ = data[start:start+1, :].to(self.device)
    #     output = []
    #     for i in range(steps):
    #         data_, _ = self(data_)
    #         out_tmp = data_[0].detach().unsqueeze(0)
    #         output.append(out_tmp[0].cpu().detach().numpy())
    #     return np.arroutput

    def predict_step(self, data, start = 0, steps = 7):
        self.eval()
        data_ = data[start:start+steps-1, :].to(self.device)
        tmp_out = self(data_).cpu().detach().numpy()
        return tmp_out
    
    def generate_step(self, data, start = 0, steps = 7):
        self.eval()
        output = data[start, :].to(self.device)
        for i in range(steps):
            tmp_out = self(output)[-1, :].reshape(1, -1)
            output = torch.concatenate([output, tmp_out])
        return output.cpu().detach().numpy()