#!/usr/bin/env python
# coding: utf-8
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import optuna
import os

from sklearn.model_selection import KFold
import time

from tqdm import tqdm
import warnings
import copy
import argparse
from sklearn.utils import resample
from torch.utils.data import Subset
from sktime.transformations.series.impute import Imputer
from rnn_transformer_encoder_decoder import *
from base_models import *
import json

def warn(*args, **kwargs):
    pass

def printf(*args, fname="log.txt"):
    with open(os.path.join("test_outputs",fname),"a+") as f:
        for a in args:
            f.write(str(a) + " ")
        f.write("\n") 
    print(args) 


# LSTM/GRU
def model_based_imputation_ds(ds_train, ds_test, fitter,Dataset,device,imp_test = False,random_ratio = None):
    imp = Imputer(method="ffill", missing_values=0.)   
    for c,y in ds_train:
        ds_train[(c,y)][1][j,k][2] = ds_train[(c,y)][1][j,k][0] == 0
        ds_train[(c,y)][1][j,k][3] = ds_train[(c,y)][1][j,k][1] == 0
        
        ds_train[(c,y)][1][j,k][0] = imp.fit_transform(ds_train[(c,y)][1][j,k][0])    
        ds_train[(c,y)][1][j,k][1] = imp.fit_transform(ds_train[(c,y)][1][j,k][1])

    for c,y in ds_test:
        ds_test[(c,y)][1][j,k][0] = imp.fit_transform(ds_train[(c,y)][1][j,k][0])   

    if fft:
        ds_train_ = copy.deepcopy(ds_train)
        ds_test_ = copy.deepcopy(ds_test)
        genFFTfeatures(ds_train_,ds_test_)                                
    else:
        ds_train_ = ds_train
        ds_test_ = ds_test
        
    dataset_train_ = Dataset(ds_train_,device,random_ratio) 
    dataset_test_ = Dataset(ds_test_,device,random_ratio = 1.0)

    train_len = int(dataset_train.n_samples * 0.8)
    val_len = dataset_train.n_samples - train_len
    test_len = dataset_test.n_samples
    
    if fitter.model.lap_pos_enc:
        dataset_train_._add_laplacian_positional_encodings(net_params['pos_enc_dim'])
        dataset_test_._add_laplacian_positional_encodings(net_params['pos_enc_dim'])        
    if fitter.model.wl_pos_enc:
        dataset_train_._add_wl_positional_encodings()
        dataset_test_._add_wl_positional_encodings()
        
    trainset, valset = torch.utils.data.random_split(dataset_train, [train_len, val_len])
    
    fitter.fit(trainset, valset,dataset_train.collate)
    _,_,_,_,output = fitter.predict(dataset_train_,dataset_train_.collate)

    for t, vals in output:
        for i,j,v in vals:
            i_ = dataset_train_.le.inverse_transform([[int(i)]])[0]
            j_ = dataset_train_.le.inverse_transform([[int(j)]])[0]
            if ds_train_[(c,y)][1][i_,j_][3][t]:
                ds_train_[(c,y)][1][i_,j_][1][t] = float(v)
    if imp_test:
        _,_,_,_,output = fitter.predict(dataset_test_,dataset_train_.collate)
    
        for t, vals in output:
            for i,j,v in vals:
                i_ = dataset_test_.le.inverse_transform([[int(i)]])[0]
                j_ = dataset_test_.le.inverse_transform([[int(j)]])[0]
                if ds_test[(c,y)][1][i_,j_][3][t]:
                    ds_test[(c,y)][1][i_,j_][1][t] = float(v)
        
def model_based_imputation(X_train, X_test, ynn_train, ynn_test, fitter, imp_test = True):
    imp = Imputer(method="ffill", missing_values=0.)   
    zero_idxs_train =  ynn_train == 0
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp = imp.fit_transform(X_test)
    ynn_train_imp = imp.fit_transform(ynn_train)
    fitter.fit(X_train_imp,ynn_train_imp)

    ynn_train_imp[zero_idxs_train] = fitter.predict(X_train_imp)[zero_idxs_train]
    return X_train_imp, X_test_imp, ynn_train_imp
    if imp_test:
        zero_idxs_test = ynn_test == 0        
        ynn_test_imp =  imp.fit_transform(ynn_test)
        ynn_test_imp[zero_idxs_test] = fitter.predict(X_test_imp)[zero_idxs_test]
        return X_train_imp, X_test_imp,ynn_train_imp, ynn_test_imp

def genFFTfeatures(cluster_year_train_, cluster_year_test_):
    for c,y in cluster_year_train_:
        for j,k in cluster_year_train_[(c,y)][1]: 
            f1 = torch.fft.rfft2(torch.from_numpy(cluster_year_train_[(c,y)][1][j,k][0][:,:2]),norm="ortho")
            f2 = torch.fft.rfft2(torch.from_numpy(cluster_year_train_[(c,y)][1][j,k][0][:,2:4]),norm="ortho")
            #print(f1.shape,f2.shape,cluster_year_[(c,y)][1][j,k][0].shape)
            cluster_year_train_[(c,y)][1][j,k][0] =  torch.hstack([f1,f2])

    for c,y in cluster_year_test_:
        for j,k in cluster_year_test_[(c,y)][1]: 
            f1 = torch.fft.rfft2(torch.from_numpy(cluster_year_test_[(c,y)][1][j,k][0][:,:2]),norm="ortho")
            f2 = torch.fft.rfft2(torch.from_numpy(cluster_year_test_[(c,y)][1][j,k][0][:,2:4]),norm="ortho")
            #print(f1.shape,f2.shape,cluster_year_[(c,y)][1][j,k][0].shape)
            cluster_year_test_[(c,y)][1][j,k][0] =  torch.hstack([f1,f2])          
    
if __name__ == "__main__":
    warnings.warn = warn
    parser = argparse.ArgumentParser(description="Run experiments on the particularly imputed dataset.")
    parser.add_argument("--imputation", type=str, help="ARIMA, FF, INTERPOLATION, MODEL")    

    args = parser.parse_args()

    if "imputation" in args:

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")        
        features_min, features_max = np.load("norm_constants" + args.imputation + ".npy",allow_pickle=True).tolist()

        for is_normed in [True, False]:
            with open('cluster_year' + args.imputation + str(is_normed) +  '_train.pkl', 'rb') as f:
                cluster_year_train = pickle.load(f)

            with open('cluster_year' + args.imputation + str(is_normed) + '_test.pkl', 'rb') as f:
                cluster_year_test = pickle.load(f)
            
            with open('name_encoder' + args.imputation + '.pkl', 'rb') as f:
                le = pickle.load(f)
        
            targs = []
            datas = []
            
            for c, y in cluster_year_train:
                for j,k in cluster_year_train[c,y][1]:
                    targs += [cluster_year_train[c,y][1][j,k][1]]
                    datas += [cluster_year_train[c,y][1][j,k][0]]
            
            Xtr = np.vstack(datas).reshape(-1,5,4)
            ytr = np.vstack(targs)

            for c, y in cluster_year_test:
                for j,k in cluster_year_test[c,y][1]:
                    targs += [cluster_year_test[c,y][1][j,k][1]]
                    datas += [cluster_year_test[c,y][1][j,k][0]]
            
            Xtest = np.vstack(datas).reshape(-1,5,4)
            ytest = np.vstack(targs)
            
            for _ in range(5):
                X_train, ynn_train = resample(Xtr, ytr, n_samples=int(Xtr.shape[0]*0.7), replace=False)
                X_test, ynn_test = resample(Xtest, ytest, n_samples=int(Xtest.shape[0]*0.7), replace=False)    

                models = {"LSTM":make_modelLSTM, "GRU": make_GRU, "TKAN": make_modelTKAN}
                #models = {"TKAN": make_modelTKAN}
                
                for model_name in models:
                    if is_normed:
                        loss_types = ['beta','bce']
                    else:
                        loss_types = ['mae']
                        
                    for loss_type in loss_types:
                        make_model = models[model_name]
                        batch_size = 64       
                    
                        def objective(trial):
                            lr = trial.suggest_float('lr', 0.00001, 0.01)
                            hidden_size = trial.suggest_int('hs', 2, 32)
                    
                            if model_name == "TKAN":
                                do = trial.suggest_float('dropout', 1e-5, 1e-2)
                                #ep = 50
                                ep = 5 
                            else:    
                                do = trial.suggest_float('dropout', 0.05, 0.2)
                                #ep = 1000
                                ep = 5
                                
                            kf = KFold(n_splits=3)
                            scores = []
                            for _, (train_index, test_index) in enumerate(kf.split(X_train)):
                                if loss_type == "beta":
                                    outp_size = 2
                                else:
                                    outp_size = 1
                                    
                                model = make_model(input_shape=X_train.shape[2],hidden_size=hidden_size, output_size = outp_size, dropout = do)
                                model.to(device)
                                fitter = RNNFitter(model,batch_size,ep,loss_type,device=device) 
                                #impute data if necessary
                                X_train_imp = X_train[train_index]
                                X_test_imp = X_train[test_index]
                                ynn_train_imp = ynn_train[train_index]
                                ynn_test_imp =  ynn_train[test_index]
                                
                                if args.imputation == "NO_IMP":
                                        X_train_imp, X_test_imp, ynn_train_imp,ynn_test_imp = model_based_imputation(X_train_imp, X_test_imp, ynn_train_imp, ynn_test_imp, fitter, imp_test = True) 
    
                                fitter.fit(X_train_imp,ynn_train_imp)
    
                                try:
                                    y_pred = fitter.predict(X_test_imp) #, batch_size=batch_size)
                                    scores.append(mean_squared_error(ynn_test_imp.flatten(),y_pred.flatten()))
                                except Exception as e:
                                    print(e)
                                    scores.append(1e26)
                                del history
                                del model    
                            return np.asarray(scores).mean() 
                            
                        study = optuna.create_study(direction='minimize')
                        #study.optimize(objective, n_trials=50)    
                        study.optimize(objective, n_trials=5)
                        
                        lr = study.best_trial.params["lr"]     
                        hs = study.best_trial.params["hs"]     
                        do = study.best_trial.params["dropout"]   

                        ep = 5
                        if model_name == "TKAN":
                            #ep = 50
                            ep = 5 
                        else:    
                            #ep = 1000
                            ep = 5         

                        if loss_type == "beta":
                            outp_size = 2
                        else:
                            outp_size = 1                                           
                    
                        model = make_model(input_shape=X_train.shape[2],hidden_size=hs, output_size = outp_size, dropout = do)
                        fitter = RNNFitter(model,batch_size,ep,loss_type,device=device)  
                       
                        X_train_imp = X_train
                        X_test_imp = X_test
                        ynn_train_imp = ynn_train
                        if args.imputation == "NO_IMP":
                            X_train_imp, X_test_imp, ynn_train_imp = model_based_imputation(X_train_imp, X_test_imp, ynn_train_imp, None, fitter, imp_test = False) 
                        
                        fitter.fit(X_train_imp,ynn_train_imp) 
                        try:
                            y_pred = fitter.predict(X_test_imp) #, batch_size=batch_size)
                            mse_score = mean_squared_error(ynn_test.flatten() * features_max[0],y_pred.flatten() * features_max[0])
                            mae_score = mean_absolute_error(ynn_test.flatten() * features_max[0],y_pred.flatten() * features_max[0])
                            r2_ = r2_score(ynn_test.flatten(),y_pred.flatten())
                            printf(model_name,mse_score, mae_score, r2_,args.imputation,is_normed,loss_type,fname="baseline_output.txt")     
                            #nn_data.append([model_name,mse_score, mae_score])
                        except Exception as e:
                            print(e)
                        del model
            
            
            # # Transformer regressor
            warnings.filterwarnings("ignore")

            for fft in [True, False]:
                for L in [1,2,3,4]:
                    for attn_heads in [1,2,3,4]:
                        for is_recurrent in [True, False]:
                            if is_normed:
                                loss_types = ['beta','bce']
                            else:
                                loss_types = ['mae']
                                
                            for loss_type in loss_types:
                            
                                max_cluster_size = 0
                                
                                lap_pos_enc = True
                                wl_pos_enc = True
                                batch_size = 8
                                #epochs = 1000
                                epochs = 5
                                
                                root_log_dir = ""
                                root_ckpt_dir = "checkpoints_encoder"
                                write_file_name = ""
                                write_config_file = ""
                                min_lr = 0.00000001
    
                                net_params = {}
                                with open("encoder.json","r") as f:
                                    net_params = json.load(f)   

                                cluster_year_train_ = copy.deepcopy(cluster_year_train)
                                cluster_year_test_ = copy.deepcopy(cluster_year_test)                                    

                                dataset_train_ = TradeDGL(cluster_year_train_,device)
                                dataset_test_ = TradeDGL(cluster_year_test_,device)                                     
                                
                                net_params['fft'] = fft
                                net_params['n_heads'] = attn_heads
                                net_params['L'] = L
                                net_params['device'] = device
                                net_params['lap_pos_enc'] = lap_pos_enc
                                net_params['wl_pos_enc'] = wl_pos_enc
                                net_params['max_cluster_size'] = len(dataset_train_.all_countries)
                                net_params['is_recurrent'] = is_recurrent
                                net_params['num_states'] = len(dataset_train_.all_countries)
                                net_params['loss'] = loss_type

                                if is_normed:
                                    net_params['scaled'] = True
                                else:
                                    net_params['scaled'] = False
                                    

    
                                model = GraphTransformerNet(net_params)
                                model = model.to(device)
                                trainer = EncoderTrainer(features_max)
    
                                fitter = TransformerFitter(model, trainer, batch_size,epochs,device,root_ckpt_dir)                                
                                if args.imputation == "NO_IMP":
                                    model_based_imputation_ds(cluster_year_train_, cluster_year_test_, fitter,TradeDGL,device=device,imp_test = False,random_ratio = None)
    
                                if fft:
                                    genFFTfeatures(cluster_year_train_,cluster_year_test_)                                
    

                                    
                                for _ in range(5):
                                    tst_idxs = np.random.choice(list(range(len(dataset_test_))), size=int(len(dataset_test_)*0.7), replace=False)
                                    tr_idxs = np.random.choice(list(range(len(dataset_train_))), size=int(len(dataset_train_)*0.7), replace=False)
                                    dataset_train = Subset(dataset_train_,tr_idxs)
                                    dataset_test = Subset(dataset_test_, tst_idxs)
    
    
                                    train_len = int(dataset_train.n_samples * 0.8)
                                    val_len = dataset_train.n_samples - train_len
                                    test_len = dataset_test.n_samples
                                    
                                    if lap_pos_enc:
                                        st = time.time()
                                        
                                        print("[!] Adding Laplacian positional encoding.")
                                        dataset_train._add_laplacian_positional_encodings(net_params['pos_enc_dim'])
                                        print('Time LapPE:',time.time()-st)
                                        
                                    if wl_pos_enc:
                                        st = time.time()
                                        print("[!] Adding WL positional encoding.")
                                        dataset_test._add_wl_positional_encodings()
                                        print('Time WL PE:',time.time()-st)
                                    testset = dataset_test
                                    trainset, valset = torch.utils.data.random_split(dataset_train, [train_len, val_len])
                                    
                                    fitter.fit(trainset, valset,dataset_train.collate)
                                    sq,a,r,residuals = fitter.predict(testset,dataset_test.collate)
                                    printf('Transformer: encoder',fft,sq,a,r, L, attn_heads,is_recurrent,args.imputation,is_normed,loss_type)

            # # Decoder
            for fft in [True, False]:
                for encL in [1,2]:
                    for decL in [1,2]:
                        for attn_heads in [1,2,3,4]:
                            for mask_ratio in [0.1,0.3,0.5,0.7]:
                                for is_recurrent in [True, False]:
                                    if is_normed:
                                        loss_types = ['beta','bce']
                                    else:
                                        loss_types = ['mae']
                                        
                                    for loss_type in loss_types:
                                    
                                        warnings.filterwarnings("ignore")
                                        
                                        max_cluster_size = 0
                                        
                                        lap_pos_enc = True
                                        wl_pos_enc = True
                                        batch_size = 8
                                        #epochs = 3000
                                        epochs = 5
                                        
                                        write_file_name = ""
                                        write_config_file = ""
                                        min_lr = 0.00000001
                                        cluster_year_train_ = copy.deepcopy(cluster_year_train)
                                        cluster_year_test_ = copy.deepcopy(cluster_year_test)
    
                                        model = GraphTransformerNetDec(net_params)
                                        model = model.to(device)
                                        trainer = EncoderDecoderTrainer(features_max)
                                        
                                        fitter = TransformerFitter(model, trainer, batch_size,epochs,device,root_ckpt_dir)       
                                        if args.imputation == "NO_IMP":
                                            model_based_imputation_ds(cluster_year_train_, cluster_year_test_, fitter,TradeDGLDecoder,device=device,imp_test = False,random_ratio = mask_ratio)                                    
    
                                        if fft:
                                            genFFTfeatures(cluster_year_train_,cluster_year_test_)                                
                                    
                                        train_dataset_ = TradeDGLDecoder(cluster_year_train_,device,random_ratio = mask_ratio)
                                        test_dataset_ = TradeDGLDecoder(cluster_year_test_,device,random_ratio = 1.0)                                
                                        for _ in range(5):
                                            tst_idxs = np.random.choice(list(range(len(dataset_test_))), size=int(len(dataset_test_)*0.7), replace=False)
                                            tr_idxs = np.random.choice(list(range(len(dataset_train_))), size=int(len(dataset_train_)*0.7), replace=False)
                                            dataset_train = Subset(dataset_train_,tr_idxs)
                                            dataset_test = Subset(dataset_test_, tst_idxs)
                                            net_params = {}
                                            with open("encoder.json","r") as f:
                                                net_params = json.load(f)  
                                            
                                            net_params['fft'] = fft
                                            net_params['encL'] = encL
                                            net_params['decL'] = decL
                                            net_params['n_heads'] = attn_heads                                    
                                            net_params['device'] = device
                                            net_params['lap_pos_enc'] = lap_pos_enc
                                            net_params['wl_pos_enc'] = wl_pos_enc
                                            net_params['max_cluster_size'] = max(len(dataset_train.all_countries),len(dataset_train.all_countries))
                                            net_params['is_recurrent'] = is_recurrent
                                            net_params['num_states'] = max(len(dataset_train.all_countries),len(dataset_train.all_countries))
                                            net_params['loss'] = loss_type

                                            if is_normed:
                                                net_params['scaled'] = True
                                            else:
                                                net_params['scaled'] = False                                            
                                            
                                            if lap_pos_enc:
                                                st = time.time()
                                                print("[!] Adding Laplacian positional encoding.")
                                                dataset_train._add_laplacian_positional_encodings(net_params['pos_enc_dim'])
                                                dataset_test._add_laplacian_positional_encodings(net_params['pos_enc_dim'])
                                                print('Time LapPE:',time.time()-st)
                                                
                                            if wl_pos_enc:
                                                st = time.time()
                                                print("[!] Adding WL positional encoding.")
                                                dataset_train._add_wl_positional_encodings()
                                                dataset_test._add_wl_positional_encodings()
                                                print('Time WL PE:',time.time()-st)
                                            
                                            trainset, valset = torch.utils.data.random_split(dataset_train, [0.8,0.2])
                                            
                                            fitter.fit(trainset, valset,dataset_train.collate)
                                            sq,a,r,residuals,_ = fitter.predict(testset,dataset_test.collate)
                                            printf('Transformer: encoder-decoder',fft,sq,a,r,L, attn_heads,mask_ratio,is_recurrent,args.imputation,is_normed,loss_type)
    else:
        print("You should define imputation type via 'imputation' parameter")



