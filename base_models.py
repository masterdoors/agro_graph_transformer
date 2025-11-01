import torch
from torch import nn
import torch.nn.functional as F
from timekan.models.tkan_lstm import tKANLSTM
import numpy as np
import copy
from torch.nn import BCELoss
from torch.nn import L1Loss
from torch.nn import MSELoss
from sklearn.metrics import r2_score
from pytorch_forecasting.metrics.distributions import BetaDistributionLoss
import uuid
import os
import torch.nn.functional as F

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class RNNModel(nn.Module):
    def __init__(self,rnn_layer, hidden_size,output_size, drop = 0., scaled = True, beta = False,mse = False,dtype=torch.float64):
        super(RNNModel, self).__init__()

        self.num_layers = 1
        self.rnn = rnn_layer 
        self.fc1 = nn.Linear(hidden_size, output_size,dtype=dtype)
        self.beta = beta
        self.scaled = scaled
        self.mse = mse
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
        if self.scaled and not self.mse:
            out = F.sigmoid(out)
        if self.beta:
            out2 = F.relu(self.fc2(out_))
            out2 = torch.clip(out2, min=0.0000001)
            return torch.hstack([out, out2])
        else:    
            return out

class LSTMModel(RNNModel):
    def __init__(self,input_size, hidden_size, output_size, drop = 0., scaled = True, beta = False, mse = False):
        lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1,batch_first=True,dtype=torch.float64)
        super(LSTMModel, self).__init__(lstm,  hidden_size, output_size, drop = drop, scaled = scaled, beta = beta, mse = mse)

class GRUModel(RNNModel):
    def __init__(self,input_size, hidden_size, output_size, drop = 0., scaled = True, beta = False, mse = False):
        lstm = torch.nn.GRU(input_size, hidden_size, num_layers=1,batch_first=True,dtype=torch.float64)
        super(GRUModel, self).__init__(lstm,  hidden_size, output_size, drop = drop, scaled = scaled, beta = beta, mse = mse)

class TKANModel(RNNModel):
    def __init__(self,input_size, hidden_size, output_size, drop = 0., scaled = True, beta = False, mse = False):
        tkan = tKANLSTM(
            input_dim=input_size,
            hidden_dim=hidden_size,
            return_sequences=True,
            bidirectional=False,
            kan_type='chebyshev',
            sub_kan_configs={'degree':5}
            )
        super(TKANModel, self).__init__(tkan,  hidden_size, output_size, drop = drop, scaled = scaled, beta = beta, mse = mse,dtype=torch.float32)

def make_modelLSTM(input_shape, hidden_size, output_size, dropout, scaled = True, beta = False, mse = False):
    return LSTMModel(input_shape, hidden_size,output_size, dropout,scaled,beta, mse)

def make_GRU(input_shape, hidden_size, output_size, dropout, scaled = True, beta = False, mse = False):
    return GRUModel(input_shape, hidden_size,output_size, dropout,scaled,beta, mse)

def make_modelTKAN(input_shape, hidden_size, output_size, dropout, scaled = True, beta = False, mse = False):
    return TKANModel(input_shape, hidden_size,output_size, dropout,scaled,beta, mse)

class RNNFitter:
    def __init__(self,model,batch_size,ep,loss_type, lr,device,early=True):
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
        self.early = early
        torch.save(self.model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + self.id_))        

    def predict(self,X):
        if isinstance(self.model,TKANModel):
            X = X.astype(np.float32)
        res = self.model(torch.from_numpy(X).to(self.device))    
        if self.loss_type == 'beta':
            res = res[:,0]
        return res.reshape(X.shape[0],X.shape[1]).detach().cpu().numpy()

    def fit(self,X,y):   
        ckpt_dir = os.path.join(self.root_ckpt_dir, "RUN_")
        self.model.load_state_dict(torch.load('{}.pkl'.format(ckpt_dir + "/epoch_" + self.id_), weights_only=True))    
        self.model.to(self.device)
        if self.loss_type == 'beta':
            criterion = BetaDistributionLoss()
        elif self.loss_type == 'mae':         
            criterion = L1Loss()
        elif self.loss_type == 'mse':    
            criterion = MSELoss()
        else:    
            criterion = BCELoss()

        val_criterion = L1Loss()
        if isinstance(self.model,TKANModel):
            X = X.astype(np.float32)
            y = y.astype(np.float32)            
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) #,weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.1,
                                                         patience=5
                                                         )
        distr = None
        best_loss = 1e10
        best_model = copy.deepcopy(self.model)
        early_stopper = EarlyStopper(patience=15,min_delta=0.0001)
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
                if not self.loss_type == 'beta':
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
        
            with torch.no_grad():
                out = self.model(torch.from_numpy(X).to(self.device))
                lss_ = criterion(out, torch.from_numpy(y).to(self.device).reshape(-1,1))                
                
                if self.loss_type == 'beta':
                    out = out[:,0]
                
                lss = val_criterion(out.reshape(-1,1),torch.from_numpy(y).to(self.device).reshape(-1,1))

                if early_stopper.early_stop(lss) and self.early:             
                    break                
                if lss_ < best_loss:
                    best_loss = lss_
                    best_model = copy.deepcopy(self.model)
                scheduler.step(lss_) 

        self.model = best_model
