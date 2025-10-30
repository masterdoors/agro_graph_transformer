from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from rnn_transformer_encoder import *

import numpy as np

from layers.graph_transformer_edge_layer import EncoderDecoderLayer

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()
        self.l1 = GraphTransformerLayer(in_dim, out_dim, num_heads, dropout,
                                                    layer_norm, batch_norm, residual,use_FFN=False)
        self.l2 = EncoderDecoderLayer(out_dim, out_dim, out_dim, num_heads, dropout,
                                                    layer_norm, batch_norm, residual)    
    def forward(self, g, h, e, h2, e2, mask):
        hidden_h, hidden_e = self.l1(g,h,e,mask)
        return self.l2(g,h2,e2,hidden_h, hidden_e)

class GraphTransformerNetDec(nn.Module):
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
        n_enc_layers = net_params['encL']
        n_dec_layers = net_params['decL']

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
        
        #self.embedding_h = nn.Linear(1, self.hidden_dim,dtype=torch.double)
        self.embedding_h = nn.Linear(1, self.hidden_dim,dtype=torch.double)
        if self.is_recurrent:
            self.embedding_h = nn.GRUCell(1, self.hidden_dim,dtype=torch.double)
        else:    
            self.embedding_h = nn.Linear(1, self.hidden_dim,dtype=torch.double)        
        self.embedding_h_dec = nn.Linear(1, self.hidden_dim,dtype=torch.double)        

        if self.edge_feat and self.is_recurrent:
            #self.embedding_e = nn.Embedding(num_trade_indicators, hidden_dim)
            self.embedding_e = nn.GRUCell(2, self.hidden_dim,dtype=torch.double)
            self.embedding_e_dec = nn.Linear(1, self.hidden_dim,dtype=torch.double)
        else:
            self.embedding_e = nn.Linear(2, self.hidden_dim,dtype=torch.double)
            self.embedding_e_dec = nn.Linear(1, self.hidden_dim,dtype=torch.double)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.beta = False
        if net_params['loss'] == 'bce':
            self._loss = nn.BCELoss()
        elif net_params['loss'] == 'mae':
            self._loss = nn.L1Loss()
        else:
            self._loss = BetaDistributionLoss()
            self.beta = True

        self.enc_layers = nn.ModuleList([GraphTransformerLayer(self.hidden_dim, self.hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_enc_layers) ]) 
        self.dec_layers = nn.ModuleList([DecoderBlock(self.hidden_dim, self.hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_dec_layers) ])
        self.MLP_layer = MLPReadout(self.hidden_dim, 1,1,self.fft,scaled = self.scaled,beta = self.beta)
        self.eps = 1e-12
        
    def forward(self, g1, h1, e1, g2, h2, e2, mask, h_lap_pos_enc=None, h_wl_pos_enc=None):
        # input embedding
        if self.is_recurrent: 
            h1 = self.embedding_h(h1[:,self.hidden_dim:],h1[:,:self.hidden_dim])
        else:
            h1 = self.embedding_h(h1)    

        h2 = self.embedding_h_dec(h2)
        #h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.double()) 
            h1 = h1 + h_lap_pos_enc
            h2 = h2 + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h1 = h1 + h_wl_pos_enc
            h2 = h2 + h_wl_pos_enc

        #print("e2:",e2)
            
        if self.is_recurrent: 
            e1 = self.embedding_e(e1[:,self.hidden_dim:],e1[:,:self.hidden_dim])  
        else:
            e1 = self.embedding_e(e1)  

        e2 = self.embedding_e_dec(e2)              

        #print("h: ", h.shape)
        #print("e22: ", e2)
        
        #ENCODER
        # convnets
        for conv in self.enc_layers:
            h1, e1 = conv(g1, h1, e1)
        g1.ndata['h'] = h1
        g1.edata['h'] = e1
        
        #DECODER
        
        for i,conv in enumerate(self.dec_layers):
            #print("dec layer ", i)
            h2, e2 = conv(g2,h2,e2,h1,e1, mask)
        
        e2 = e2.reshape((-1,e2.shape[1]))

        g2.ndata['h'] = h2
        g2.edata['h'] = e2        

        #OUTPUT
        mlp = self.MLP_layer(e2)

        #if self.fft:
        #    mlp = self.MLP_layer.tn(torch.fft.ifft(mlp))
        if self.beta:
            mlp = mlp.reshape(2,-1)
        else:
            mlp = mlp.reshape(1,-1)
        return mlp, e1, h1, e2, h2
        
        
    def loss(self, scores, targets):
        loss = self._loss(scores,targets)
        #loss = nn.L1Loss()(scores, targets)
        return loss

class EncoderDecoderTrainer:
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
        losses = []
        for iter, (batch_graphs_, batch_targets_, batch_targets_n_) in enumerate(data_loader):
            batch_scores = None
            batch_vert_scores = None
            batch_dec_scores = None
            batch_dec_vert_scores = None        
            batch_res = []
            bt = []
            bx = []
            be = []
            loss_mask = []
    
            optimizer.zero_grad()        
            for t in range(len(batch_targets_)):
                batch_graphs = batch_graphs_[t]
    
                batch_targets = batch_targets_[t]
                batch_graphs = batch_graphs.to(device)
                batch_x_ = batch_graphs.ndata['feat'].reshape(-1,1).to(device)  # num x feat
                batch_e = batch_graphs.edata['feat'].reshape(-1,2).to(device)

                if model.is_recurrent:
                    if t > 0:
                        batch_e = torch.hstack([batch_scores, batch_e])
                        batch_x = torch.hstack([batch_vert_scores, batch_x_])                
                    else:
                        batch_e = torch.hstack([torch.randn((batch_e.shape[0],model.hidden_dim)), batch_e])
                        batch_x = torch.hstack([torch.randn((batch_x_.shape[0],model.hidden_dim)), batch_x_])         
                else:
                    batch_x = batch_x_                
                    
                batch_targets_n = batch_targets_n_[t]
                batch_targets_n = batch_targets_n.to(device)
                batch_tx = batch_targets_n.ndata['feat'].reshape(-1,1).to(device)  # num x feat
                batch_te = batch_targets_n.edata['feat'].reshape(-1,1).to(device)  
                masked_edges = batch_targets_n.edata['mask'].flatten().to(device)  
                
                batch_targets = batch_targets.edata['feat'].reshape(1,-1)

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
        
                mask = torch.zeros((batch_e.shape[0],model.num_heads,1))
                loss_mask.append(masked_edges)
    
                batch_res_, batch_scores, batch_vert_scores, batch_dec_scores, batch_dec_vert_scores = model.forward(batch_graphs, batch_x, batch_e, batch_targets_n, batch_tx, batch_te, mask, batch_lap_pos_enc, batch_wl_pos_enc)
                batch_res.append(batch_res_)
    
            ten = torch.nan_to_num(torch.hstack(batch_res).real,nan=0., posinf=1 - model.eps,neginf=model.eps)
            if model.fft:
                if model.beta:
                    ten = torch.vstack([model.MLP_layer.tn(ten[0]),ten[1]])
                else:    
                    ten = model.MLP_layer.tn(ten)
            
            loss_mask = torch.hstack(loss_mask)     
            doubled_mask = None
            if model.beta:
                doubled_mask = loss_mask.repeat(2).reshape(2,-1)
                #masked_ten = ten.masked_fill(~doubled_mask,0.)
                #masked_bt = torch.hstack(bt).flatten().masked_fill(~loss_mask,0.).reshape(-1,1)
            #else:    
            #    masked_ten = ten.masked_fill(~loss_mask,0.)
            #    masked_bt = torch.hstack(bt).flatten().masked_fill(~loss_mask,0.)

            #print(loss_mask.sum())
            #loss = model.loss(masked_ten, masked_bt)
            #loss = model.loss(ten, torch.hstack(bt).flatten())
    
            if model.beta:
                loss = model.loss(ten[doubled_mask], torch.hstack(bt).flatten()[loss_mask].reshape(-1,1))
            else:    
                loss = model.loss(ten.flatten()[loss_mask], torch.hstack(bt).flatten()[loss_mask])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            epoch_loss += loss.detach().item()
    
            nb_data += batch_targets.size(0)
        epoch_loss /= (iter + 1)
    
        return epoch_loss, optimizer
    
    def evaluate_network(self,model, device, data_loader, epoch):
        model.eval()
        epoch_test_loss = 0
        epoch_test_mae = 0
        nb_data = 0
        losses = []
        with torch.no_grad():
            for iter, (batch_graphs_, batch_targets_, batch_targets_n_) in enumerate(data_loader):
                batch_scores = None
                batch_vert_scores = None
                batch_dec_scores = None
                batch_dec_vert_scores = None               
                batch_res = []  
                bt = []
                loss_mask = []
                for t in range(len(batch_targets_)):
                    batch_graphs = batch_graphs_[t]
                    batch_targets = batch_targets_[t]            
                    batch_graphs = batch_graphs.to(device)
                    batch_x_ = batch_graphs.ndata['feat'].reshape(-1,1).to(device)
                    batch_e = batch_graphs.edata['feat'].reshape(-1,2).to(device)
                    
                    if model.is_recurrent:
                        if t > 0:
                            batch_e = torch.hstack([batch_scores, batch_e])
                            batch_x = torch.hstack([batch_vert_scores, batch_x_])                    
                        else:
                            batch_e = torch.hstack([torch.zeros((batch_e.shape[0],model.hidden_dim)), batch_e])
                            batch_x = torch.hstack([torch.zeros((batch_x_.shape[0],model.hidden_dim)), batch_x_])
                    else:
                        batch_x = batch_x_   
                    
                    batch_targets_n = batch_targets_n_[t]
                    batch_targets_n = batch_targets_n.to(device)
                    batch_tx = batch_targets_n.ndata['feat'].reshape(-1,1).to(device)  # num x feat
                    batch_te = batch_targets_n.edata['feat'].reshape(-1,1).to(device)    
                    masked_edges = batch_targets_n.edata['mask'].flatten().to(device) 
                     
                    
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
                    
                    mask = torch.zeros((batch_e.shape[0],model.num_heads,1))
                    loss_mask.append(masked_edges)
                   
                    batch_res_, batch_scores, batch_vert_scores, batch_dec_scores, batch_dec_vert_scores = model.forward(batch_graphs, batch_x, batch_e, batch_targets_n, batch_tx, batch_te,mask,batch_lap_pos_enc, batch_wl_pos_enc)
                    batch_res.append(batch_res_)
    
                ten = torch.nan_to_num(torch.hstack(batch_res).real,nan=0., posinf=1 - model.eps,neginf=model.eps)
                if model.fft:
                    if model.beta:
                        ten = torch.vstack([model.MLP_layer.tn(ten[0]),ten[1]])
                    else:    
                        ten = model.MLP_layer.tn(ten)             
                #print(ten[ten<0],ten[ten == 0],ten[ten>1],ten[ten==1],torch.isnan(ten).any(),torch.isinf(ten).any())
                loss_mask = torch.hstack(loss_mask)
                
                doubled_mask = None
                if model.beta:
                    doubled_mask = loss_mask.repeat(2).reshape(2,-1)
                    masked_ten = ten.masked_fill(~doubled_mask,0.)
                    masked_bt = torch.hstack(bt).flatten().masked_fill(~loss_mask,0.).reshape(-1,1)
                else:    
                    masked_ten = ten.masked_fill(~loss_mask,0.)
                    masked_bt = torch.hstack(bt).flatten().masked_fill(~loss_mask,0.)
    
                #print(loss_mask.sum())
                #loss = model.loss(masked_ten, masked_bt)
                #loss = model.loss(ten, torch.hstack(bt).flatten())
                if model.beta:
                    loss = model.loss(ten[doubled_mask], torch.hstack(bt).flatten()[loss_mask].reshape(-1,1))
                else:    
                    loss = model.loss(ten.flatten()[loss_mask], torch.hstack(bt).flatten()[loss_mask])          
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
        labeled_pred = []
        loss_mask = []
        with torch.no_grad():
            for iter, (batch_graphs_, batch_targets_, batch_targets_n_) in enumerate(data_loader):
                batch_scores = None
                batch_vert_scores = None
                batch_dec_scores = None
                batch_dec_vert_scores = None               
                bt = []
                for t in range(len(batch_targets_)):
                    batch_graphs = batch_graphs_[t]
                    batch_targets = batch_targets_[t]            
                    batch_graphs = batch_graphs.to(device)
                    batch_x_ = batch_graphs.ndata['feat'].reshape(-1,1).to(device)
                    batch_e = batch_graphs.edata['feat'].reshape(-1,2).to(device)
                    
                    if model.is_recurrent:
                        if t > 0:
                            batch_e = torch.hstack([batch_scores, batch_e])
                            batch_x = torch.hstack([batch_vert_scores, batch_x_])                    
                        else:
                            batch_e = torch.hstack([torch.zeros((batch_e.shape[0],model.hidden_dim)), batch_e])
                            batch_x = torch.hstack([torch.zeros((batch_x_.shape[0],model.hidden_dim)), batch_x_])
                    else:
                        batch_x = batch_x_   
    
                    batch_targets_n = batch_targets_n_[t]
                    batch_targets_n = batch_targets_n.to(device)
                    batch_tx = batch_targets_n.ndata['feat'].reshape(-1,1).to(device)  # num x feat
                    batch_te = batch_targets_n.edata['feat'].reshape(-1,1).to(device)    
                    masked_edges = batch_targets_n.edata['mask'].flatten().to(device)   
                 
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
    
                    mask = torch.zeros((batch_e.shape[0],model.num_heads,1))
                    loss_mask.append(masked_edges)
    
                    invals =  batch_graphs.edges("all")[0]
                    outvals = batch_graphs.edges("all")[1]
                    batch_res_, batch_scores, batch_vert_scores, batch_dec_scores, batch_dec_vert_scores = model.forward(batch_graphs, batch_x, batch_e, batch_targets_n, batch_tx, batch_te,mask,batch_lap_pos_enc, batch_wl_pos_enc)
                    #labeled_pred [t,[(i,j,value),..,]]
                    
                    if model.fft:
                        if model.beta:
                            batch_res_ = torch.vstack([model.MLP_layer.tn(batch_res_[0]),batch_res_[1]])
                        else:    
                            batch_res_ = model.MLP_layer.tn(batch_res_)

                    labeled_pred.append([t,[(i,j,batch_res_[0,k]) for k,(i,j) in enumerate(zip(invals,outvals))]])
                    batch_res.append(batch_res_[0].flatten())
                    batch_tar.append(batch_targets.flatten())    
                    
        loss_mask = torch.hstack(loss_mask)
        if loss_mask.sum() > 0:
            ynn_test = torch.hstack(batch_tar).detach()[loss_mask].cpu().numpy()
            y_pred =  torch.nan_to_num(torch.hstack(batch_res).detach()[loss_mask].cpu().real,nan=0.,posinf=0.,neginf=0.).numpy()
        else:
            ynn_test = torch.hstack(batch_tar).detach().cpu().numpy()
            y_pred =  torch.nan_to_num(torch.hstack(batch_res).detach().cpu().real,nan=0.,posinf=0.,neginf=0.).numpy()
    
        #print(ynn_test[:30], y_pred[:30])
        if model.scaled:
            mse_score = mean_squared_error(ynn_test.flatten() * self.features_max[0],y_pred.flatten() * self.features_max[0])
            mae_score = mean_absolute_error(ynn_test.flatten() * self.features_max[0],y_pred.flatten() * self.features_max[0])
        else:
            mse_score = mean_squared_error(ynn_test.flatten() * 100000,y_pred.flatten() * 100000)
            mae_score = mean_absolute_error(ynn_test.flatten()* 100000,y_pred.flatten() * 100000)

        r2_ = r2_score(ynn_test.flatten(),y_pred.flatten())
        return mse_score,mae_score,r2_,ynn_test.flatten() - y_pred.flatten(),labeled_pred 

class TradeDGLDecoder(TradeDGL):
    def __init__(self, data,device, fft = False, random_ratio = 0.8):
        self.data = data
        self.device = device
        self.num_graphs = len(data)
        self.fft = fft
                
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
        self.graph_noisy_labels = []
        self.n_samples = len(self.data)
        self.random_ratio = random_ratio        
        self._prepare()

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels, noised_labels = map(list, zip(*samples))
        batched_graphs = []
        labels_ = []
        noised_labels_ = []
        for i in range(len(graphs[0])):
            blabels = dgl.batch([l[i] for l in labels]) 
            bnlabels = dgl.batch([l[i] for l in noised_labels]) 
            batched_graph = dgl.batch([g[i] for g in graphs])       
            batched_graphs.append(batched_graph)          
            labels_.append(blabels)
            noised_labels_.append(bnlabels)
        return batched_graphs, labels_, noised_labels_
    
    def _prepare(self):
        all_countries = []
        for c,y in self.data:
            all_countries += list(self.data[(c,y)][0].keys())

        self.all_countries = list(set(all_countries))
        self.le = LabelEncoder().fit(all_countries)
        tars = [] 
        prods = []
        for c,y in self.data:
            for t in range(5):
                tar = np.zeros((len(all_countries),len(all_countries)))
                prod = np.zeros((len(all_countries),))
                for j,k in self.data[(c,y)][1]:
                    j_ = self.le.transform([j])[0]
                    k_ = self.le.transform([k])[0]  
                    tar[j_,k_] = self.data[(c,y)][1][j,k][1][t]          
                    prod[j_] = self.data[(c,y)][1][j,k][0][t,2]
                    prod[k_] = self.data[(c,y)][1][j,k][0][t,3]
                
                tars.append(tar)
                prods.append(prod)

        tars = np.hstack(tars).flatten()
        prods = np.hstack(prods).flatten()

        #self.tar_loc, self.tar_scale = laplace.fit(tars)
        #self.prod_loc, self.prod_scale = laplace.fit(prods)

        #random_tars = laplace.rvs(self.tar_loc,self.tar_scale,size=len(tars))
        #random_prods = laplace.rvs(self.prod_loc, self.prod_scale,size=len(prods))

        for c,y in self.data:
            dat_label = []
            dat_cluster = []
            dat_random_label = []
            for t in range(5):
                prod = np.zeros((len(all_countries),))
                expt = np.zeros((len(all_countries),len(all_countries),2))
                tar = np.zeros((len(all_countries),len(all_countries)))
                random_tar = np.zeros((len(all_countries),len(all_countries)))
                random_prod = np.zeros((len(all_countries),))
                random_mask = np.zeros((len(all_countries),len(all_countries)),dtype=bool)

                inidxs = []
                outidxs = []
                for j,k in self.data[(c,y)][1]:
                    j_ = self.le.transform([j])[0]
                    k_ = self.le.transform([k])[0]                    
                    inidxs.append(j_)
                    outidxs.append(k_)
                    expt[j_,k_] = self.data[(c,y)][1][j,k][0][t,:2]
                    prod[j_] = self.data[(c,y)][1][j,k][0][t,2]
                    prod[k_] = self.data[(c,y)][1][j,k][0][t,3]
                    tar[j_,k_] = self.data[(c,y)][1][j,k][1][t]    
                    if len(self.data[(c,y)]) > 2:
                        #use pre-defined values:
                        preprod_j = self.data[(c,y)][2][j,k][0][t,2]
                        preprod_k = self.data[(c,y)][2][j,k][0][t,3]
                        pretar = self.data[(c,y)][2][j,k][1][t]                           
                        random_tar[j_,k_] = pretar
                        random_prod[j_] = preprod_j
                        random_prod[k_] = preprod_k
                    else:    
                        if bool(np.random.choice([0,1], 1, p=[self.random_ratio, 1. - self.random_ratio])[0]):
                            random_tar[j_,k_] = tar[j_,k_]
                            random_prod[j_] = prod[j_] 
                            random_prod[k_] = prod[k_]
                        else:    
                            random_tar[j_,k_] = self.data[(c,y)][1][j,k][0][t,:1]
                            random_mask[j_,k_] = True
                            random_prod[j_] = prod[j_]
                            random_prod[k_] = prod[k_]
                        
                node_features = torch.from_numpy(prod).to(device=self.device)

                adj = torch.from_numpy(expt).to(device=self.device)
    
                edge_features = adj[inidxs,outidxs].reshape(-1,2)
                
                # Create the DGL Graph
                g = dgl.DGLGraph()
                g = g.to(device=self.device)
                g.my_id = (y,c,t)
                g.add_nodes(len(all_countries))
                g.ndata['feat'] = node_features
                
                for src, dst in zip(inidxs,outidxs):
                    g.add_edges(src.item(), dst.item())
                g.edata['feat'] = edge_features
                dat_cluster.append(g)
                
                # Create the target DGL Graph
                g = dgl.DGLGraph()
                g = g.to(device=self.device)
                g.my_id = (y,c,t,"tar")
                g.add_nodes(len(all_countries))
                #g.ndata['feat'] = torch.zeros(prod.shape).to(device=self.device)
                adj = torch.from_numpy(tar).to(device=self.device)

                edge_features = adj[inidxs,outidxs].reshape(-1)
                
                for src, dst in zip(inidxs,outidxs):
                    g.add_edges(src.item(), dst.item())
      
                                
                g.edata['feat'] = edge_features   
                dat_label.append(g)

                #generate randomized output graph
                g = dgl.DGLGraph()
                g = g.to(device=self.device)
                g.my_id = (y,c,t,"random_tar")
                g.add_nodes(len(all_countries))
                node_features = torch.from_numpy(random_prod).to(device=self.device)
                g.ndata['feat'] = node_features
                adj = torch.from_numpy(random_tar).to(device=self.device)

                edge_features = adj[inidxs,outidxs].reshape(-1)
                
                for src, dst in zip(inidxs,outidxs):
                    g.add_edges(src.item(), dst.item())
      
                random_mask = torch.from_numpy(random_mask).to(device=self.device)[inidxs,outidxs].reshape(-1)
                g.edata['feat'] = edge_features   
                g.edata['mask'] = random_mask
                dat_random_label.append(g)
            
            self.graph_labels.append(dat_label)
            self.graph_lists.append(dat_cluster)            
            self.graph_noisy_labels.append(dat_random_label)            

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
        return self.graph_lists[idx], self.graph_labels[idx], self.graph_noisy_labels[idx]
