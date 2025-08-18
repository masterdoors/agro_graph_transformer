import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer with edge features
    
"""

"""
    Util functions
"""
def src_dot_dst_sum(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func

def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func

# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}
    return func


def exp(field,mask):
    def func(edges):
        # clamp for softmax numerical stability
        if mask is not None:
            return {field: torch.exp((edges.data[field].sum(-1, keepdim=True) + mask).clamp(-5, 5))}
        else:
            return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func




"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True,dtype=torch.double)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True,dtype=torch.double)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True,dtype=torch.double)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True,dtype=torch.double)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False,dtype=torch.double)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False,dtype=torch.double)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False,dtype=torch.double)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False,dtype=torch.double)
    
    def propagate_attention(self, g, mask = None):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))
        
        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))
        
        # softmax
        g.apply_edges(exp('score',mask))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h, e, mask= None):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)
        
        self.propagate_attention(g, mask)
        
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6)) # adding eps to all values here
        e_out = g.edata['e_out']
        
        return h_out, e_out

   
#standard multihead attention
class EncoderDecoderAttentionLayer(nn.Module):
   def __init__(self, enc_dim, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.e_attention = nn.MultiheadAttention(in_dim, num_heads,dtype=torch.double,bias=False)
        self.h_attention = nn.MultiheadAttention(in_dim, num_heads,dtype=torch.double,bias=False) 

   def forward(self, dec_h, dec_e, enc_h,enc_e):
       e,w = self.e_attention(dec_e,enc_e,enc_e)
       # if not self.training:
       #     print("Encoder edges:")
       #     print(enc_e)
       #     print("Decoder edges:")
       #     print(dec_e)
       h,w = self.h_attention(dec_h,enc_h,enc_h)
       # if not self.training:
       #     print("Vert attention")
       #     print(w)
       #     print(w.max(axis=1))           
       return h, e


class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False, use_FFN = True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        self.ffn = use_FFN
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O_h = nn.Linear(out_dim, out_dim,dtype=torch.double)
        self.O_e = nn.Linear(out_dim, out_dim,dtype=torch.double)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim,dtype=torch.double)
            self.layer_norm1_e = nn.LayerNorm(out_dim, dtype=torch.double)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim,dtype=torch.double)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim,dtype=torch.double)
        
        if self.ffn:
            # FFN for h
            self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2,dtype=torch.double)
            self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim,dtype=torch.double)
            
            # FFN for e
            self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2,dtype=torch.double)
            self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim,dtype=torch.double)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim,dtype=torch.double)
            self.layer_norm2_e = nn.LayerNorm(out_dim,dtype=torch.double)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim,dtype=torch.double)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim,dtype=torch.double)
        
    def forward(self, g, h, e, mask = None):
        h_in1 = h # for first residual connection
        e_in1 = e # for first residual connection
        
        # multi-head attention out
        #print("h att in", h.shape)
        #print("e att in", e.shape)
        
        h_attn_out, e_attn_out = self.attention(g, h, e, mask)

        #print("h_attn",h_attn_out.shape)
        #print("e_attn",e_attn_out.shape)
        
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1[:,:self.out_channels] + h # residual connection
            e = e_in1[:,:self.out_channels] + e # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h # for second residual connection
        e_in2 = e # for second residual connection

        if self.ffn:
            # FFN for h
            h = self.FFN_h_layer1(h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.FFN_h_layer2(h)
    
            # FFN for e
            e = self.FFN_e_layer1(e)
            e = F.relu(e)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.FFN_e_layer2(e)

            if self.residual:
                h = h_in2 + h # residual connection       
                e = e_in2 + e # residual connection  

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)             

        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)

class EncoderDecoderLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, enc_dim, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False, use_FFN = True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        self.ffn = use_FFN
        
        self.attention = EncoderDecoderAttentionLayer(enc_dim, in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O_h = nn.Linear(out_dim, out_dim,dtype=torch.double)
        self.O_e = nn.Linear(out_dim, out_dim,dtype=torch.double)
        

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim,dtype=torch.double)
            self.layer_norm1_e = nn.LayerNorm(out_dim, dtype=torch.double)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim,dtype=torch.double)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim,dtype=torch.double)
        
        if self.ffn:
            # FFN for h
            self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2,dtype=torch.double)
            self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim,dtype=torch.double)
            
            # FFN for e
            self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2,dtype=torch.double)
            self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim,dtype=torch.double)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim,dtype=torch.double)
            self.layer_norm2_e = nn.LayerNorm(out_dim,dtype=torch.double)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim,dtype=torch.double)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim,dtype=torch.double)
        
    def forward(self, g, enc_h, enc_e, h, e):
        h_in1 = h # for first residual connection
        e_in1 = e
        
        # multi-head attention out
        #print("h att in", h.shape)
        #print("e att in", e.shape)

        #encoder-decoder attention
        
        h_attn_out, e_attn_out = self.attention(h,e,enc_h, enc_e)

        #print("h_attn",h_attn_out.shape)
        #print("e_attn",e_attn_out.shape)
        
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h # residual connection
            e = e_in1 + e # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)        
        
        h_in3 = h # for second residual connection
        e_in3 = e # for second residual connection

        if self.ffn:
            # FFN for h
            h = self.FFN_h_layer1(h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.FFN_h_layer2(h)
    
            # FFN for e
            e = self.FFN_e_layer1(e)
            e = F.relu(e)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.FFN_e_layer2(e)

            if self.residual:
                h = h_in3 + h # residual connection       
                e = e_in3 + e # residual connection  

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)             

        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)