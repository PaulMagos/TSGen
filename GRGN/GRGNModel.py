import torch
from torch import Tensor
from typing import Optional
from tsl.nn.models import BaseModel 
from tsl.nn.layers import NodeEmbedding
from torch_geometric.typing import Adj, OptTensor
from .Cells import GRGNCell
from GRGN.Utils.reshapes import reshape_to_original
from tqdm import tqdm
from torch.nn import Linear, Sequential, ReLU, Dropout

class GRGNModel(BaseModel):
    def __init__(self, 
                input_size: int,
                hidden_size: int = 64,
                mixture_size: int = 32,
                embedding_size: Optional[int] = None,
                n_layers: int = 1,
                n_nodes: Optional[int] = None,
                kernel_size: int = 2,
                decoder_order: int = 1,
                layer_norm: bool = True,
                dropout: float =.0,
                ):
        super(GRGNModel, self).__init__()
        
        # Model Name
        self._name = 'GRGN'
        # Number of gaussians (means and variances)
        self.M = mixture_size
        # Input size of the data
        self.input_size = input_size
        
        # Forward Step
        self.fwd_grgl = GRGNCell(input_size = input_size, 
                                 hidden_size = hidden_size, 
                                 mixture_size = mixture_size, 
                                 kernel_size = kernel_size, 
                                 dropout=dropout,
                                 n_nodes=n_nodes,
                                 n_layers=n_layers,
                                 layer_norm = layer_norm, 
                                 decoder_order = decoder_order)
        # Backward Step
        self.bwd_grgl = GRGNCell(input_size = input_size, 
                                 hidden_size = hidden_size, 
                                 mixture_size = mixture_size, 
                                 kernel_size = kernel_size, 
                                 dropout=dropout,
                                 n_nodes=n_nodes,
                                 n_layers=n_layers,
                                 layer_norm = layer_norm,
                                 decoder_order = decoder_order)
        
        
        if embedding_size is not None:
            assert n_nodes is not None
            self._embedding = NodeEmbedding(n_nodes, embedding_size)
        else:
            self.register_parameter('embedding', None)
            
        self.out = getattr(torch, 'mean')
        
    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                u: OptTensor = None) -> list:
        """"""
        # x: [batch, steps, nodes, channels]
        fwd_out, fwd_pred, fwd_repr, _ = self.fwd_grgl(x,
                                                       edge_index,
                                                       edge_weight,
                                                       u=u)
        # Backward
        rev_x = x.flip(1)
        rev_u = u.flip(1) if u is not None else None
        *bwd, _ = self.bwd_grgl(rev_x,
                                edge_index,
                                edge_weight,
                                u=rev_u)
        bwd_out, bwd_pred, bwd_repr = [res.flip(1) for res in bwd]

        generation = torch.stack([fwd_out, bwd_out], dim=-1)
        generation = self.out(generation, dim=-1)
        
        prediction = torch.stack([fwd_pred, bwd_pred], dim=-1)
        prediction = self.out(prediction, dim=-1)

        return torch.cat([prediction, generation], dim=-1)
    
    def predict(self,
                X: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                method: str = 'mean',
                u: OptTensor = None, 
                encoder_only=False,
                both_mean=False
                ) -> Tensor:
        """"""
        out = self.forward(x=X,
                            u=u,
                            edge_index=edge_index,
                            edge_weight=edge_weight)
        
        input_features = X.shape[-1]
        D = X.shape[-1] * X.shape[-2]

        if both_mean and not both_mean:
            out1 = (out[..., :(self.input_size * 3) * self.M] + out[..., (self.input_size * 3) * self.M:])/2
        else:
            if encoder_only:
                out1 = out[..., :(self.input_size * 3) * self.M]
            else:
                out1 = out[..., (self.input_size * 3) * self.M:]
        
        M = out1.shape[-1] // (input_features*3)
        stds_index = self.M*D
        weights_index = self.M*(D*2)

        out1 = reshape_to_original(out1, X.shape[-2], input_features, M)
        
        means = out1[..., :stds_index].reshape(-1, M, D)
        stds  = out1[..., stds_index:weights_index].reshape(-1, M, D)
        weights = out1[..., weights_index:].reshape(-1, M, D)
        
        means = torch.where(torch.isnan(means) | torch.isinf(means), torch.zeros_like(means), means).to(means.device)
        stds = torch.where(torch.isnan(stds) | torch.isinf(stds), torch.zeros_like(stds), stds).to(stds.device)
        weights = torch.where(torch.isnan(weights) | torch.isinf(weights), torch.zeros_like(weights), weights).to(weights.device)
            
        pred = weights * torch.normal(means, stds)
        match(method):
            case 'mean':
                pred = torch.mean(pred, axis=1)
            case 'sum':
                pred = torch.sum(pred, axis=1)
        
        return pred, out
    
    def generate(self,
                   X: Tensor,
                   edge_index: Adj,
                   edge_weight: OptTensor = None,
                   u: OptTensor = None,
                   method: str = 'mean',
                   steps: int = 32, 
                   encoder_only = False,
                   both_mean = False
                   ) -> Tensor:
    
        nextval = X
        output = []
        output_not_computed = []
        input_features = X.shape[-1]
        D = X.shape[-1] * X.shape[-2]
        
        with tqdm(len(range(0, steps)), total=steps, desc='Generating', unit='step') as t:
            for i in range(steps):
                out = self.forward(x=nextval,
                            u=u,
                            edge_index=edge_index,
                            edge_weight=edge_weight
                            )
                if both_mean and not both_mean:
                    out1 = (out[..., :(self.input_size * 3) * self.M] + out[..., (self.input_size * 3) * self.M:])/2
                else:
                    if encoder_only:
                        out1 = out[..., :(self.input_size * 3) * self.M]
                    else:
                        out1 = out[..., (self.input_size * 3) * self.M:(self.input_size * 3) * self.M*2]
                    
                if i==0:
                    M = out1.shape[-1] // (input_features*3)
                    stds_index = self.M*D
                    weights_index = self.M*(D*2)

                out1 = reshape_to_original(out1, X.shape[-2], input_features, M)
                
                means = out1[..., :stds_index].reshape(-1, M, D)
                stds  = out1[..., stds_index:weights_index].reshape(-1, M, D)
                weights = out1[..., weights_index:].reshape(-1, M, D)
                
                means = torch.where(torch.isnan(means) | torch.isinf(means), torch.zeros_like(means), means).to(means.device)
                stds = torch.where(torch.isnan(stds) | torch.isinf(stds), torch.zeros_like(stds), stds).to(stds.device)
                weights = torch.where(torch.isnan(weights) | torch.isinf(weights), torch.zeros_like(weights), weights).to(weights.device)
                
                gen = weights * torch.normal(means, stds)
                
                match(method):
                    case 'mean':
                        gen = torch.mean(gen, axis=1)
                    case 'sum':
                        gen = torch.sum(gen, axis=1)
                        
                nextval = gen.reshape(1, 1, gen.shape[-1], 1)
                output.append(nextval)
                output_not_computed.append(out)
                t.update(1)
        
        return output, output_not_computed

    def predict_iter(self,
                   X: Tensor,
                   edge_index: Adj,
                   edge_weight: OptTensor = None,
                   u: OptTensor = None,
                   method: str = 'mean',
                   encoder_only = False,
                   both_mean=False
                   ) -> Tensor:
    
        nextval = X
        output = []
        output_not_computed = []
        steps = X.shape[0]
        
        input_features = X.shape[-1]
        D = X.shape[-1] * X.shape[-2]
        
        with tqdm(len(range(0, steps)), total=steps, desc='Predicting', unit='step') as t:
            for i in range(steps):
                nextval = X[i].reshape(1, 1, X.shape[-2], 1)
                out = self.forward(x=nextval,
                            u=u,
                            edge_index=edge_index,
                            edge_weight=edge_weight
                            )

                if both_mean and not encoder_only:
                    out1 = (out[..., :(self.input_size * 3) * self.M] + out[..., (self.input_size * 3) * self.M:])/2
                else:
                    if encoder_only:
                        out1 = out[..., :(self.input_size * 3) * self.M]
                    else:
                        out1 = out[..., (self.input_size * 3) * self.M:]
                
                if i==0:
                    M = out1.shape[-1] // (input_features*3)
                    stds_index = self.M*D
                    weights_index = self.M*(D*2)

                out1 = reshape_to_original(out1, X.shape[-2], input_features, M)
                
                means = out1[..., :stds_index].reshape(-1, M, D)
                stds  = out1[..., stds_index:weights_index].reshape(-1, M, D)
                weights = out1[..., weights_index:].reshape(-1, M, D)
                
                means = torch.where(torch.isnan(means) | torch.isinf(means), torch.zeros_like(means), means).to(means.device)
                stds = torch.where(torch.isnan(stds) | torch.isinf(stds), torch.zeros_like(stds), stds).to(stds.device)
                weights = torch.where(torch.isnan(weights) | torch.isinf(weights), torch.zeros_like(weights), weights).to(weights.device)
                
                gen = weights * torch.normal(means, stds)
                
                match(method):
                    case 'mean':
                        gen = torch.mean(gen, axis=1)
                    case 'sum':
                        gen = torch.sum(gen, axis=1)
                        
                output.append(gen.reshape(1, 1, gen.shape[-1], 1))
                output_not_computed.append(out)
                t.update(1)
        
        return output, output_not_computed