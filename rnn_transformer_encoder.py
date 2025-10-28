from layers.graph_transformer_edge_layer import GraphTransformerLayer
import torch
from torch import nn

import hashlib
from sktime.forecasting.arima import AutoARIMA
import pandas as pd
from tqdm import tqdm
import scipy

import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import scipy.sparse as sp

from torch.nn import BCELoss
from torch.nn import L1Loss
from pytorch_forecasting.metrics.distributions import BetaDistributionLoss
import time
import uuid
from torch import optim
from torch.utils.data import DataLoader
import os
import numpy as np

class TransformerFitter:
    def __init__(self,model,trainer,batch_size,ep,device,root_ckpt_dir):
        self.model = model
        self.batch_size = batch_size
        self.ep = ep
        self.device = device
        self.trainer = trainer
        self.id_ = str(uuid.uuid4())
        self.root_ckpt_dir = root_ckpt_dir
        ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
        torch.save(self.model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + self.id_))

    def predict(self,testset, collate_fn):
        test_loader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        
        return self.trainer.test_network(self.model, self.device, test_loader)
        
    def fit(self,trainset,valset, collate_fn):
        #reset model before training
        ckpt_dir = os.path.join(self.root_ckpt_dir, "RUN_")
        self.model.load_state_dict(torch.load('{}.pkl'.format(ckpt_dir + "/epoch_" + self.id_), weights_only=True))
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.5,
                                                         patience=30,
                                                         verbose=True)
        
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(valset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        min_loss = 100
        min_val = 0
        min_name = ""    
        try:
            with tqdm(range(self.ep)) as t:
                for epoch in t:
                    t.set_description('Epoch %d' % epoch)
                    start = time.time()
                    epoch_train_loss, optimizer = self.trainer.train_epoch(self.model, optimizer, self.device, train_loader, epoch)
                    epoch_val_loss = self.trainer.evaluate_network(self.model, self.device, val_loader, epoch)
        
                    # Saving checkpoint
                    ckpt_dir = os.path.join(self.root_ckpt_dir, "RUN_")
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    id_ = str(uuid.uuid4())
                    if epoch_train_loss < min_loss:
                        min_loss = epoch_train_loss
                        min_val = epoch_val_loss
                        min_name = '{}.pkl'.format(ckpt_dir + "/epoch_" + id_)
                        
                    t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_loss, val_loss=epoch_val_loss)
         
        
                    torch.save(self.model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + id_))
        
                    scheduler.step(epoch_val_loss)
                    
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early because of KeyboardInterrupt')
        
        print("Convergence Time (Epochs): {:.4f}".format(epoch))
        
        print("Min loss:", min_loss,min_val)
        
        #model = GraphTransformerNet(net_params)
        self.model.load_state_dict(torch.load(min_name, weights_only=True))



"""
    MLP Layer used after graph vector representation
"""

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2, fft = False, scaled = True, beta = False): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim , input_dim , bias=True, dtype=torch.double) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim , output_dim , bias=True,dtype=torch.double ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        if scaled:
            self.tn = nn.Sigmoid()
        else:
            self.tn = nn.ReLU()
        self.fft = fft
        self.beta = beta
        if self.beta:
            self.rel = nn.ReLU()
            self.blayer = nn.Linear( input_dim , output_dim , bias=True,dtype=torch.double)
        self.drop = nn.Dropout(p=0.05)
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
            y = self.drop(y) 
        y = self.FC_layers[self.L](y)

        if self.beta:
            y2 = self.blayer(y)
            y2 = self.rel(y2) + 0.0001
            if self.fft:
                return torch.hstack([y.reshape(-1,1),y2.reshape(-1,1)])
            else:    
                return torch.hstack([self.tn(y).reshape(-1,1),y2.reshape(-1,1)])            
        else:
            if self.fft:
                return y
            else:    
                return self.tn(y)


class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.fft = net_params['fft']
        num_states = net_params['num_states']
        num_trade_indicators = net_params['num_trade_indicators']
        self.hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        self.num_heads = num_heads
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.is_recurrent = net_params['is_recurrent']
        self.scaled = net_params['scaled']
        
        max_wl_role_index = net_params['max_cluster_size'] # this is maximum graph size in the dataset
        self.num_states = num_states
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, self.hidden_dim,dtype=torch.double)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, self.hidden_dim)

        
        self.embedding_h = nn.Linear(1, self.hidden_dim,dtype=torch.double)
        if self.is_recurrent:
            self.embedding_h = nn.GRUCell(1, self.hidden_dim,dtype=torch.double)
        else:    
            self.embedding_h = nn.Linear(1, self.hidden_dim,dtype=torch.double)
            
        if self.edge_feat and self.is_recurrent:
            #self.embedding_e = nn.Embedding(num_trade_indicators, hidden_dim)
            self.embedding_e = nn.GRUCell(2, self.hidden_dim,dtype=torch.double)
        else:
            self.embedding_e = nn.Linear(2, self.hidden_dim,dtype=torch.double)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.beta = False
        if net_params['loss'] == 'bce':
            self._loss = nn.BCELoss()
        elif net_params['loss'] == 'mae':
            self._loss = nn.L1Loss()
        else:
            self._loss = BetaDistributionLoss()
            self.beta = True
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(self.hidden_dim, self.hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(self.hidden_dim, self.hidden_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(self.hidden_dim, 1,1,self.fft,scaled = self.scaled,beta = self.beta)
        self.eps = 1e-12
        
    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):
        # input embedding
        h = self.embedding_h(h[:,self.hidden_dim:],h[:,:self.hidden_dim])
        #h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.double()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc

        #if not self.fft:
        e = self.embedding_e(e[:,self.hidden_dim:],e[:,:self.hidden_dim])  
        #else:    
        #    e = self.embedding_e(torch.fft.fft(e[:,self.hidden_dim:]),e[:,:self.hidden_dim])  
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        g.edata['h'] = e
        
        e = e.reshape((-1,e.shape[1]))

        mlp = self.MLP_layer(e)

        #if self.fft:
        #    mlp = F.sigmoid(torch.fft.ifft(mlp))

        if self.beta:
            mlp = mlp.reshape(2,-1)
        else:
            mlp = mlp.reshape(1,-1)
            
        return mlp, e, h
        
        
    def loss(self, scores, targets):
        loss = self._loss(scores,targets)
        #loss = nn.L1Loss()(scores, targets)
        return loss
        
class EncoderTrainer:
    def __init__(self,features_max):
        self.features_max = features_max

    def train_epoch(self, model, optimizer, device, data_loader, epoch):
        model.train()
        epoch_loss = 0
        epoch_train_mae = 0
        nb_data = 0
        gpu_mem = 0
    
        y_pred = []   
        y_true = []    
        for iter, (batch_graphs_, batch_targets_) in enumerate(data_loader):
            batch_scores = None
            batch_vert_scores = None
            batch_res = []
            bt = []
            bx = []
            be = []
    
            optimizer.zero_grad()        
            for t in range(len(batch_targets_)):
                batch_graphs = batch_graphs_[t]
    
                batch_targets = batch_targets_[t]
                batch_graphs = batch_graphs.to(device)
                batch_x_ = batch_graphs.ndata['feat'].reshape(-1,1).to(device)  # num x feat
                batch_e = batch_graphs.edata['feat'].reshape(-1,2).to(device)
                if t > 0:
                    batch_e = torch.hstack([batch_scores, batch_e])
                    batch_x = torch.hstack([batch_vert_scores, batch_x_])                
                else:
                    batch_e = torch.hstack([torch.randn((batch_e.shape[0],model.hidden_dim)), batch_e])
                    batch_x = torch.hstack([torch.randn((batch_x_.shape[0],model.hidden_dim)), batch_x_])                
                    
                batch_targets = batch_targets.edata['feat'].reshape(1,-1)
                #ez = torch.zeros((1,model.num_states*model.num_states - batch_targets.shape[1]))
                #batch_targets = torch.hstack([batch_targets,ez])
                batch_targets = batch_targets.to(device)
                bt.append(batch_targets)
                bx = batch_x_[batch_graphs.edges("all")[0]]
                #be.append(torch.hstack([batch_e, bx]))
                be = torch.hstack([batch_e, bx])
                
                try:
                    batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
                    sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
                    sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                    batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
                except:
                    batch_lap_pos_enc = None
                    
                try:
                    batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
                except:
                    batch_wl_pos_enc = None
        
                batch_res_, batch_scores, batch_vert_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
    
                batch_res.append(batch_res_)
    
    
            ten = torch.nan_to_num(torch.hstack(batch_res).real,nan=0., posinf=1 - model.eps,neginf=model.eps)
    
            if model.fft:
                if model.beta:
                    ten = torch.vstack(model.MLP_layer.tn(ten[0]),ten[1])
                else:    
                    ten = model.MLP_layer.tn(ten)
            
            if model.beta:
                loss = model.loss(torch.t(ten), torch.hstack(bt).reshape(-1,1))
            else:    
                loss = model.loss(ten.flatten(), torch.hstack(bt).flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.detach().item()
            nb_data += batch_targets.size(0)
        epoch_loss /= (iter + 1)
        #r2 = r2_score(np.asarray(y_pred).flatten(),np.asarray(y_true).flatten())    
        #print("r2: ",r2)
        return epoch_loss, optimizer
    
    def evaluate_network(self, model, device, data_loader, epoch):
        model.eval()
        epoch_test_loss = 0
        epoch_test_mae = 0
        nb_data = 0
        with torch.no_grad():
            for iter, (batch_graphs_, batch_targets_) in enumerate(data_loader):
                batch_scores = None
                batch_vert_scores = None
                batch_res = []  
                bt = []
                for t in range(len(batch_targets_)):
                    batch_graphs = batch_graphs_[t]
                    batch_targets = batch_targets_[t]            
                    batch_graphs = batch_graphs.to(device)
                    batch_x_ = batch_graphs.ndata['feat'].reshape(-1,1).to(device)
                    batch_e = batch_graphs.edata['feat'].reshape(-1,2).to(device)
                    
                    if t > 0:
                        batch_e = torch.hstack([batch_scores, batch_e])
                        batch_x = torch.hstack([batch_vert_scores, batch_x_])                    
                    else:
                        batch_e = torch.hstack([torch.zeros((batch_e.shape[0],model.hidden_dim)), batch_e])
                        batch_x = torch.hstack([torch.zeros((batch_x_.shape[0],model.hidden_dim)), batch_x_])
                    
                    batch_targets = batch_targets.edata['feat'].reshape(1,-1)
                    #ez = torch.zeros((1,model.num_states*model.num_states - batch_targets.shape[1]))
                    #batch_targets = torch.hstack([batch_targets,ez])
                    batch_targets = batch_targets.to(device)
    
                    bt.append(batch_targets)
                    try:
                        batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
                    except:
                        batch_lap_pos_enc = None
                    
                    try:
                        batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
                    except:
                        batch_wl_pos_enc = None
    
                    bx = batch_x_[batch_graphs.edges("all")[0]]
                    #be.append(torch.hstack([batch_e, bx]))
                    be = torch.hstack([batch_e, bx])                
                        
                    batch_res_, batch_scores, batch_vert_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
                    batch_res.append(batch_res_)
    
                ten = torch.nan_to_num(torch.hstack(batch_res).real,nan=0., posinf=1 - model.eps,neginf=model.eps)
                if model.fft:
                    if model.beta:
                        ten = torch.vstack(model.MLP_layer.tn(ten[0]),ten[1])
                    else:    
                        ten = model.MLP_layer.tn(ten)
                
                if model.beta:
                    loss = model.loss(torch.t(ten), torch.hstack(bt).reshape(-1,1))
                else:    
                    loss = model.loss(ten.flatten(), torch.hstack(bt).flatten())
                epoch_test_loss += loss.detach().item()
                nb_data += batch_targets.size(0)
            epoch_test_loss /= (iter + 1)
            
        return epoch_test_loss 
    
    def test_network(self, model, device, data_loader):
        model.eval()
        epoch_test_loss = 0
        epoch_test_mae = 0
        nb_data = 0
        batch_res = []
        batch_tar = []
        with torch.no_grad():
            for iter, (batch_graphs_, batch_targets_) in enumerate(data_loader):
                batch_scores = None
                batch_vert_scores = None
                bt = []
                for t in range(len(batch_targets_)):
                    batch_graphs = batch_graphs_[t]
                    batch_targets = batch_targets_[t]            
                    batch_graphs = batch_graphs.to(device)
                    batch_x_ = batch_graphs.ndata['feat'].reshape(-1,1).to(device)
                    batch_e = batch_graphs.edata['feat'].reshape(-1,2).to(device)
                    
                    if t > 0:
                        batch_e = torch.hstack([batch_scores, batch_e])
                        batch_x = torch.hstack([batch_vert_scores, batch_x_])
                    else:
                        batch_e = torch.hstack([torch.zeros((batch_e.shape[0],model.hidden_dim)), batch_e])
                        batch_x = torch.hstack([torch.zeros((batch_x_.shape[0],model.hidden_dim)), batch_x_])
                    
                    batch_targets = batch_targets.edata['feat'].reshape(1,-1)
                    #ez = torch.zeros((1,model.num_states*model.num_states - batch_targets.shape[1]))
                    #batch_targets = torch.hstack([batch_targets,ez])
                    batch_targets = batch_targets.to(device)
    
                    bt.append(batch_targets)
                    try:
                        batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
                    except:
                        batch_lap_pos_enc = None
                    
                    try:
                        batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
                    except:
                        batch_wl_pos_enc = None
    
                    bx = batch_x_[batch_graphs.edges("all")[0]]
                    #be.append(torch.hstack([batch_e, bx]))
                    be = torch.hstack([batch_e, bx])                
                        
                    batch_res_, batch_scores, batch_vert_scores = model.forward(batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc)
                    batch_res.append(batch_res_)
                    batch_tar.append(batch_targets.flatten())    
                    
        ynn_test = torch.hstack(batch_tar).detach().cpu().numpy()
        y_pred = torch.hstack(batch_res).detach().cpu().real
        
        if model.fft:
            if model.beta:
                y_pred = model.MLP_layer.tn(y_pred[0]).flatten()
            else:    
                y_pred = model.MLP_layer.tn(y_pred).flatten()
    
        y_pred = y_pred.numpy()
    
        #print(ynn_test[:30], y_pred[:30])
        mse_score = mean_squared_error(ynn_test.flatten() * self.features_max[0],y_pred.flatten() * self.features_max[0])
        mae_score = mean_absolute_error(ynn_test.flatten() * self.features_max[0],y_pred.flatten() * self.features_max[0])
        r2_ = r2_score(ynn_test.flatten(),y_pred.flatten())
        return mse_score,mae_score,r2_,ynn_test.flatten() - y_pred.flatten() 

def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adj_external(scipy_fmt='csr')
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order

    pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).double() 
    if pos_enc.shape[1] < pos_enc_dim:
        df = pos_enc_dim - pos_enc.shape[1]
        dfz = torch.zeros((pos_enc.shape[0],df))
        pos_enc = torch.hstack([pos_enc,dfz])
        #print("Ext on ", df)
    g.ndata['lap_pos_enc'] = pos_enc
    
    return g

def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1


    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g


class TradeDGL(torch.utils.data.Dataset):
    def __init__(self, data,device,fft = False, random_ratio = None):
        self.data = data
        self.device = device
        self.num_graphs = len(data)
                
        """
        data is a list of TradeUnion dict objects with following attributes
        
          un = data[idx]
        ; un['num_states'] : nb of states, an integer (N)
        ; un['production'] : tensor of size N, each element is a production value, a float number > 0
        ; un['export'] : tensor of size N x N, each element is a value of export, a float number > 0
        ; un['pred_export'] : tensor of size N x N, each element is a value of export, a float number > 0
        """
        
        self.graph_lists = []
        self.graph_labels = []
        self.fft = fft
        self.n_samples = len(self.data)
        self._prepare()

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        batched_graphs = []
        labels_ = []
        
        for i in range(len(graphs[0])):
            blabels = dgl.batch([l[i] for l in labels]) 
            batched_graph = dgl.batch([g[i] for g in graphs])       
            batched_graphs.append(batched_graph)          
            labels_.append(blabels)
        return batched_graphs, labels_
    
    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        self.graph_lists = [[laplacian_positional_encoding(g, pos_enc_dim) for g in gt] for gt in self.graph_lists]

    def _add_wl_positional_encodings(self):
        # WL positional encoding from Graph-Bert, Zhang et al 2020.
        self.graph_lists = [[wl_positional_encoding(g) for g in gt] for gt in self.graph_lists]
    
    def _prepare(self):
        all_countries = []
        for c,y in self.data:
            all_countries += list(self.data[(c,y)][0].keys())

        self.all_countries = list(set(all_countries))
        self.le = LabelEncoder().fit(all_countries)
        
        for c,y in self.data:
            dat_label = []
            dat_cluster = []
            for t in range(5):
                prod = torch.zeros((len(all_countries),),dtype=torch.double)
                expt = torch.zeros((len(all_countries),len(all_countries),2),dtype=torch.double)
                tar = torch.zeros((len(all_countries),len(all_countries)),dtype=torch.double)
                inidxs = []
                outidxs = []
                for j,k in self.data[(c,y)][1]:
                    j_ = self.le.transform([j])[0]
                    k_ = self.le.transform([k])[0]                    
                    inidxs.append(j_)
                    outidxs.append(k_)
                    expt[j_,k_,0] = self.data[(c,y)][1][j,k][0][t,0]
                    expt[j_,k_,1] = self.data[(c,y)][1][j,k][0][t,1]
                    prod[j_] = self.data[(c,y)][1][j,k][0][t,2]
                    prod[k_] = self.data[(c,y)][1][j,k][0][t,3]
                    tar[j_,k_] = self.data[(c,y)][1][j,k][1][t]    
                node_features = prod.to(device=self.device)

                adj = expt.to(device=self.device)
    
                edge_features = adj[inidxs,outidxs].reshape(-1,2)
                
                # Create the DGL Graph
                g = dgl.DGLGraph()
                g.my_id = (y,c,t)
                g.add_nodes(len(all_countries))
                g.ndata['feat'] = node_features
                
                for src, dst in zip(inidxs,outidxs):
                    g.add_edges(src.item(), dst.item())
                g.edata['feat'] = edge_features
                dat_cluster.append(g)
                
                # Create the target DGL Graph
                g = dgl.DGLGraph()
                g.my_id = (y,c,t,"tar")
                g.add_nodes(len(all_countries))
                #g.ndata['feat'] = torch.zeros(prod.shape).to(device=self.device)
                adj = tar.to(device=self.device)

                edge_features = adj[inidxs,outidxs].reshape(-1)
                
                for src, dst in zip(inidxs,outidxs):
                    g.add_edges(src.item(), dst.item())
      
                g.edata['feat'] = edge_features   
                dat_label.append(g)
            self.graph_labels.append(dat_label)
            self.graph_lists.append(dat_cluster)            
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]

