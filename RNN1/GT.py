from GMM import GMM
from torch import nn
import numpy as np
import torch
from tqdm import tqdm
from EarlyStopping import EarlyStopping
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)
__all__ = ['GT', 'GTM']

class GTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, mixture_dim, dropout, num_layers, bidirectional, loss, lr, callbacks, device, debug) -> None:
        super(GTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional)
        self.dense = nn.Linear(in_features=hidden_size*(2 if bidirectional else 1), out_features=(output_size+2)*mixture_dim, device=device)
        self.gmm = GMM(M = mixture_dim, device = device, debug=debug)
        self.loss = loss
        self.device = device
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.callbacks = {}
        if 'EarlyStopping' in callbacks:
            self.callbacks['EarlyStopping'] = EarlyStopping()

    def forward(self, x):
        return self.gmm(self.dense(self.lstm(x)[0]))

    def train_step(self, train_data, epochs = 1):
        self.train()
        print("Starting training...")
        for epoch in range(1, epochs + 1):
            losses_epoch = []
            with tqdm(total=len(train_data) - 1) as pbar:
                for i in range(len(train_data) - 1):
                    self.optimizer.zero_grad()
                    if torch.rand(1).item() < 0 and  i > 100:
                        print(outputs2, outputs2.detach().unsqueeze(0), train_data[i:i+1, :].to(self.device))
                        outputs = self(outputs2.detach().unsqueeze(0))
                    else:
                        outputs = self(train_data[i:i+1, :].to(self.device))
                        outputs2 = outputs
                    loss = self.loss(train_data[i+1, :].to(self.device), outputs)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=20)
                    self.optimizer.step()
                    losses_epoch.append(loss.item())
                    if i%100 == 0:
                        mean_loss = np.mean(losses_epoch)
                        pbar.set_description("Loss %s" % mean_loss)
                        # wandb.log({"loss": mean_loss})
                    pbar.update(1)

            mean_loss = np.mean(losses_epoch)
            print(f'Epoch {epoch} - loss:', mean_loss)
            if self.callbacks['EarlyStopping'](mean_loss):
                print(f'Early Stopped at epoch {epoch} with loss {mean_loss}')
                break

        return self

    def predict_step(self, data, start = 0, steps = 7):
        self.eval()
        data_ = data[start:start+1, :].to(self.device)
        output = []
        for i in range(steps):
            out = self(data_)
            out_tmp = out[0].detach().unsqueeze(0)
            data_ = out
            output.append(out_tmp[0].cpu().detach().numpy())
        return output


class GTLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, num_layers, bidirectional, loss, lr, callbacks, device) -> None:
        super(GTLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional)
        self.dense = nn.Linear(in_features=hidden_size*(2 if bidirectional else 1), out_features=output_size, device=device)
        self.Relu = nn.Sigmoid()
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.3)
        self.callbacks = {}
        if 'EarlyStopping' in callbacks:
            self.callbacks['EarlyStopping'] = EarlyStopping()
        self.device = device
        match(loss):
            case 'L1Loss':
                self.loss = nn.L1Loss(reduction='mean')

    def forward(self, x):
        return self.Relu(self.dense(self.lstm(x)[0]))

    def train_step(self, train_data, epochs = 1):
        self.train()
        print("Starting training...")
        for epoch in range(1, epochs + 1):
            losses_epoch = []
            with tqdm(total=len(train_data) - 1) as pbar:
                for i in range(len(train_data) - 1):
                    self.optimizer.zero_grad()
                    if torch.rand(1).item() < 0 and  i > 1000:
                        outputs = self(outputs2.detach().unsqueeze(0))
                    else:
                        outputs = self(train_data[i:i+1, :].to(self.device))
                        outputs2 = outputs[0]
                    loss = self.loss(train_data[i+1, :].to(self.device), outputs)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=20)
                    self.optimizer.step()
                    losses_epoch.append(loss.item())
                    if i%100 == 0:
                        mean_loss = np.mean(losses_epoch)
                        pbar.set_description("Loss %s" % mean_loss)
                        # wandb.log({"loss": mean_loss})
                    pbar.update(1)

            mean_loss = np.mean(losses_epoch)
            print(f'Epoch {epoch} - loss:', mean_loss)
            if self.callbacks['EarlyStopping'](mean_loss):
                print(f'Early Stopped at epoch {epoch} with loss {mean_loss}')
                break

        return self

    def predict_step(self, data, start = 0, steps = 7):
        self.eval()
        data_ = data[start:start+1, :].to(self.device)
        output = []
        for i in range(steps):
            data_ = self(data_)
            out_tmp = data_[0].detach().unsqueeze(0)
            output.append(out_tmp[0].cpu().detach().numpy())
        return output

class GTR(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout, num_layers, bidirectional, lr, callbacks, device) -> None:
        super(GTR, self).__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, device=device, bidirectional=bidirectional)
        self.dense = nn.Linear(in_features=hidden_size*(2 if bidirectional else 1), out_features=output_size, device=device)
        # self.Relu = nn.Tanh()
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.3)
        self.callbacks = {}
        if 'EarlyStopping' in callbacks:
            self.callbacks['EarlyStopping'] = EarlyStopping()
        self.device = device
        self.loss = nn.L1Loss()
        self.metrics = nn.MSELoss()

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        return self.dense(out), hidden.detach()

    def train_step(self, train_data, epochs = 1):
        hidden = torch.Tensor(self.num_layers*(2 if self.bidirectional else 1), self.hidden_size)
        self.train()
        print("Starting training...")
        for epoch in range(1, epochs + 1):
            losses_epoch = []
            metrics_epoch = []
            with tqdm(total=len(train_data) - 1) as pbar:
                for i in range(len(train_data) - 1):
                    outputs, hidden = self(train_data[i:i+1, :].to(self.device), hidden)
                    loss = self.loss(train_data[i+1, :].to(self.device), outputs)
                    metric = self.metrics(train_data[i+1, :].to(self.device), outputs)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=20)
                    self.optimizer.step()
                    losses_epoch.append(loss.item())
                    metrics_epoch.append(metric.item())
                    if i%50 == 0:
                        mean_loss = np.mean(losses_epoch)
                        mean_metric = np.mean(metrics_epoch)
                        pbar.set_description(f"L1 Loss {mean_loss}, MSE : {mean_metric}")
                        # wandb.log({"loss": mean_loss})
                    pbar.update(1)

            mean_loss = np.mean(losses_epoch)
            mean_metric = np.mean(metrics_epoch)
            print(f'Epoch {epoch} - L1 loss: {mean_loss} - MSE: {mean_metric}')
            if self.callbacks['EarlyStopping'](mean_loss):
                print(f'Early Stopped at epoch {epoch} with loss {mean_loss}')
                break

        return self

    def predict_step(self, data, start = 0, steps = 7):
        self.eval()
        data_ = data[start:start+1, :].to(self.device)
        output = []
        for i in range(steps):
            data_ = self(data_)
            out_tmp = data_[0].detach().unsqueeze(0)
            output.append(out_tmp[0].cpu().detach().numpy())
        return output
