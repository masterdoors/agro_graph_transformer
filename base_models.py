import torch
from torch import nn
import torch.nn.functional as F
from timekan.models.tkan_lstm import tKANLSTM
import numpy as np
import copy
from torch.nn import BCELoss
from torch.nn import L1Loss
from pytorch_forecasting.metrics.distributions import BetaDistributionLoss
import uuid
import os
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self,rnn_layer, hidden_size,output_size, drop = 0., scaled = True, beta = False,dtype=torch.float64):
        super(RNNModel, self).__init__()

        self.num_layers = 1
        self.rnn = rnn_layer 
        self.fc1 = nn.Linear(hidden_size, output_size,dtype=dtype)
        self.beta = beta
        self.scaled = scaled
        if beta:
            self.fc2 = nn.Linear(hidden_size, output_size,dtype=dtype)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x, hidden = None):
        if hidden is not None:
            out,_ = self.rnn(x,  hidden[:,0])
            hidden[:] = out
        else:
            out = self.rnn(x, None)    
        if isinstance(out,tuple):
            out = out[0]    
        out = self.drop(out) 
        out = torch.squeeze(out)
        out_ = out.reshape(out.shape[0]*out.shape[1], -1)
        out = self.fc1(out_)
        if self.scaled:
            out = F.sigmoid(out)
        if self.beta:
            out2 = F.relu(self.fc2(out_)) + 0.0001
            return torch.hstack([out, out2])
        else:    
            return out

class LSTMModel(RNNModel):
    def __init__(self,input_size, hidden_size, output_size, drop = 0., scaled = True, beta = False):
        lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1,batch_first=True,dtype=torch.float64)
        super(LSTMModel, self).__init__(lstm,  hidden_size, output_size, drop = drop, scaled = scaled, beta = beta)

class GRUModel(RNNModel):
    def __init__(self,input_size, hidden_size, output_size, drop = 0., scaled = True, beta = False):
        lstm = torch.nn.GRU(input_size, hidden_size, num_layers=1,batch_first=True,dtype=torch.float64)
        super(GRUModel, self).__init__(lstm,  hidden_size, output_size, drop = drop, scaled = scaled, beta = beta)

class TKANModel(RNNModel):
    def __init__(self,input_size, hidden_size, output_size, drop = 0., scaled = True, beta = False):
        tkan = tKANLSTM(
            input_dim=input_size,
            hidden_dim=hidden_size,
            return_sequences=True,
            bidirectional=False,
            kan_type='fourier',
            sub_kan_configs={'gridsize': 50, 'addbias': True})
        super(TKANModel, self).__init__(tkan,  hidden_size, output_size, drop = drop, scaled = scaled, beta = beta,dtype=torch.float32)

def make_modelLSTM(input_shape, hidden_size, output_size, dropout, scaled = True, beta = False):
    return LSTMModel(input_shape, hidden_size,output_size, dropout,scaled,beta)
 
def make_GRU(input_shape, hidden_size, output_size, dropout, scaled = True, beta = False):
    return GRUModel(input_shape, hidden_size,output_size, dropout,scaled,beta)

def make_modelTKAN(input_shape, hidden_size, output_size, dropout, scaled = True, beta = False):
    return TKANModel(input_shape, hidden_size,output_size, dropout,scaled,beta)

class RNNFitter:
    def __init__(self,model,batch_size,ep,loss_type, lr,device):
        self.model = model
        self.batch_size = batch_size
        self.ep = ep
        self.loss_type = loss_type
        self.root_ckpt_dir = "rnn_base"
        self.id_ = str(uuid.uuid4())
        ckpt_dir = os.path.join(self.root_ckpt_dir, "RUN_")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.device = device
        self.lr = lr
        torch.save(self.model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + self.id_))        

    def predict(self,X):
        if isinstance(self.model,TKANModel):
            X = X.astype(np.float32)
        return self.model(torch.from_numpy(X).to(self.device)).reshape(X.shape[0],X.shape[1],-1)[:,:,0].detach().numpy()

    def fit(self,X,y):   
        ckpt_dir = os.path.join(self.root_ckpt_dir, "RUN_")
        self.model.load_state_dict(torch.load('{}.pkl'.format(ckpt_dir + "/epoch_" + self.id_), weights_only=True))        
        if self.loss_type == 'beta':
            criterion = BetaDistributionLoss()
        elif self.loss_type == 'mae':         
            criterion = L1Loss()
        else:    
            criterion = BCELoss()

        if isinstance(self.model,TKANModel):
            X = X.astype(np.float32)
            y = y.astype(np.float32)            
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.1,
                                                         patience=5,
                                                         verbose=True)
        distr = None
        best_loss = 1e10
        best_model = copy.deepcopy(self.model)
        for epoch in range(self.ep):
            eloss = 0.
            eacc = 0.
            for i in range(int(X.shape[0] / self.batch_size)):
                batch_idxs = np.random.randint(0,X.shape[0],self.batch_size)
                X_batch =  torch.from_numpy(X[batch_idxs]).to(self.device)
                y_batch  = torch.from_numpy(y[batch_idxs]).to(self.device)
                
                out  = self.model(X_batch)
        
                loss = criterion(out, y_batch.reshape(-1,1))
                eloss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
        
            with torch.no_grad():
                out = self.model(torch.from_numpy(X).to(self.device))
                lss = criterion(out, torch.from_numpy(y).to(self.device).reshape(-1,1))
                if lss < best_loss:
                    best_loss = lss
                    best_model = copy.deepcopy(self.model)
                scheduler.step(lss) 

        self.model = best_model