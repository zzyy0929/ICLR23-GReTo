#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch_geometric.data import Data,Dataset
from torch_geometric.loader  import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd
import tqdm
import random
# from  torch.utils.data import TensorDataset,DataLoader


# In[2]:


torch.manual_seed(12345)
torch.cuda.manual_seed(12345)
np.random.seed(12345)
random.seed(12345)
torch.backends.cudnn.deterministic = True


# ### 参数

# In[3]:


BATCH_SIZE = 64
time_len = 12
learning_rate = 1e-2
EPOCH = 5

batch_first=True
bidirectional=False 
input_size= 1 #Long and latitude
hidden_size = 64
num_layers=2


# ###  Packages

# In[4]:


 # Normalization
class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# In[5]:


#Evaluation metrics
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


# In[6]:


# test(test_dataloader)


# ### Traffic volumes

# In[7]:


## Spatial-temporal
traffic = np.load('/home/hqh/DataSetFile/metrla/STdata.npy').transpose()
#traffic = np.load('/home/hqh/GWN/data/metr-la.h5').transpose()
print(type(traffic))
print(traffic.shape)
print(np.isnan(traffic).sum())


# In[8]:


x = []
y = []
for i in range(1,len(traffic[0])-time_len):
    x.append(traffic[:,i:i+time_len])#From [0.time_len) to [-1-time_len,-1) #4314,6
    y.append(traffic[:,i+time_len])#From time_len to -1 # 4314,1
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)
print(x.shape,y.shape)# batch Spatial - temporal


# ### ND_t

# In[9]:


ND_t = torch.load('/home/hqh/DataSetFile/metrla/ND_t.pt')


# In[10]:


x=torch.cat([x,ND_t],dim=-1)
print(x.shape,y.shape)# batch Spatial - temporal


# In[11]:


# dataset = TensorDataset(x,y)
# dataloader = DataLoader(dataset=data_set,batch_size=BATCH_SIZE,shuffle=True)


# In[12]:


# features, targets = next(iter(dataloader))


# ### Load adjacent matrix

# In[13]:


A=np.load('/home/hqh/DataSetFile/metrla/adj01.npy')
print(type(A))


# In[14]:


print(A)
print((A>0).sum(),'/',len(A)*len(A))


# In[15]:


adj = coo_matrix(A)
values = adj.data  
indices = np.vstack((adj.row, adj.col))  #  coo formation we required
adj = torch.LongTensor(indices)  # coo formation PyG provides
print(adj)


# ### M Positive  & Negative

# In[16]:


M_pos=torch.load('/home/hqh/DataSetFile/metrla/M_pos.pt')
M_neg=torch.load('/home/hqh/DataSetFile/metrla/M_neg.pt')


# In[17]:


M=torch.cat((M_pos.unsqueeze(-1),M_neg.unsqueeze(-1)),dim=-1) # concatenate to 2d dimension
print(M.shape)


# ### dataloader:train/valid/test Ratio: 7 1 2

# In[18]:


def construct_dataloader(x,y,M,loader_type):
    data_list = [] 
    for i in tqdm.tqdm(range(len(x))):
        data = Data(x=x[i], edge_index=adj,y=y[i],edge_attr=M[i])
        data_list.append(data)
    if loader_type=='train':
        dataloader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=True)# construct a large graph
    else:
        dataloader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False)# construct a large graph

    return dataloader


# In[19]:


num_samples = len(x)
train_ratio = 0.8
test_ratio = 0.1

num_test = round(num_samples * 0.2)
num_train = round(num_samples * 0.7)
num_val = num_samples - num_test - num_train
x_train, y_train,M_train = x[:num_train], y[:num_train],M[:num_train]
x_val, y_val,M_val =  x[num_train: num_train + num_val],y[num_train: num_train + num_val],M[num_train: num_train + num_val]
x_test, y_test,M_test = x[-num_test:], y[-num_test:],M[-num_test:]


scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

train_dataloader = construct_dataloader(x_train,y_train,M_train,'train')
val_dataloader = construct_dataloader(x_val,y_val,M_val,'valid')
test_dataloader = construct_dataloader(x_test,y_test,M_test,'test')


# In[20]:


print(train_dataloader.dataset.__len__())
print(val_dataloader.dataset.__len__())
print(test_dataloader.dataset.__len__())


# ### Model&Optimizer

# In[21]:


# class GCN(torch.nn.Module):
#     def __init__(self):
#         super(GCN, self).__init__()
#         self.lstm1 = torch.nn.LSTM(num_layers = 1,batch_first=batch_first,
#                                bidirectional=bidirectional,input_size=input_size,hidden_size = hidden_size)
#         self.conv1 = GCNConv(hidden_size, 64)
#         self.lstm2 = torch.nn.LSTM(num_layers = 1,batch_first=batch_first,
#                                bidirectional=bidirectional,input_size=input_size,hidden_size = hidden_size)
#         self.mlp = Linear(hidden_size,1)

#     def forward(self, batch):
#         x, edge_index=batch.x,batch.edge_index
#         out,(hn,cn) = self.lstm1(x.unsqueeze(-1))
#         out = out.relu()
#         out = self.conv1(out[:,-1,:], edge_index)
#         out = out.relu()
#         out,(hn,cn) = self.lstm2(x.unsqueeze(-1))
#         out = out[:,-1,:].relu()
#         out = self.mlp(out)
#         return out
    
# class GAT(torch.nn.Module):
#     def __init__(self):
#         super(GAT, self).__init__()
#         self.lstm1 = torch.nn.LSTM(num_layers = 1,batch_first=batch_first,
#                                bidirectional=bidirectional,input_size=input_size,hidden_size = hidden_size)
#         self.conv1 = GATConv(hidden_size, 64)
#         self.lstm2 = torch.nn.LSTM(num_layers = 1,batch_first=batch_first,
#                                bidirectional=bidirectional,input_size=input_size,hidden_size = hidden_size)
#         self.mlp = Linear(hidden_size,1)

#     def forward(self, batch):
#         x, edge_index=batch.x,batch.edge_index
#         out,(hn,cn) = self.lstm1(x.unsqueeze(-1))
#         out = out.relu()
#         out = self.conv1(out[:,-1,:], edge_index)
#         out = out.relu()
#         out,(hn,cn) = self.lstm2(x.unsqueeze(-1))
#         out = out[:,-1,:].relu()
#         out = self.mlp(out)
#         return out
    
# class GraphSAGE(torch.nn.Module):
#     def __init__(self):
#         super(GraphSAGE, self).__init__()
#         self.lstm1 = torch.nn.LSTM(num_layers = 1,batch_first=batch_first,
#                                bidirectional=bidirectional,input_size=input_size,hidden_size = hidden_size)
#         self.conv1 = SAGEConv(hidden_size, 64)
#         self.lstm2 = torch.nn.LSTM(num_layers = 1,batch_first=batch_first,
#                                bidirectional=bidirectional,input_size=input_size,hidden_size = hidden_size)
#         self.mlp = Linear(hidden_size,1)

#     def forward(self, batch):
#         x, edge_index=batch.x,batch.edge_index
#         out,(hn,cn) = self.lstm1(x.unsqueeze(-1))
#         out = out.relu()
#         out = self.conv1(out[:,-1,:], edge_index)
#         out = out.relu()
#         out,(hn,cn) = self.lstm2(x.unsqueeze(-1))
#         out = out[:,-1,:].relu()
#         out = self.mlp(out)
#         return out
    
# class GraphSAGE(torch.nn.Module):
#     def __init__(self):
#         super(GraphSAGE, self).__init__()
#         self.lstm1 = torch.nn.LSTM(num_layers = 1,batch_first=batch_first,
#                                bidirectional=bidirectional,input_size=input_size,hidden_size = hidden_size)
#         self.conv1 = SAGEConv(hidden_size, 64)
#         self.lstm2 = torch.nn.LSTM(num_layers = 1,batch_first=batch_first,
#                                bidirectional=bidirectional,input_size=input_size,hidden_size = hidden_size)
#         self.mlp = Linear(hidden_size,1)

#     def forward(self, batch):
#         x, edge_index=batch.x,batch.edge_index
#         out,(hn,cn) = self.lstm1(x.unsqueeze(-1))
#         out = out.relu()
#         out = self.conv1(out[:,-1,:], edge_index)
#         out = out.relu()
#         out,(hn,cn) = self.lstm2(x.unsqueeze(-1))
#         out = out[:,-1,:].relu()
#         out = self.mlp(out)
#         return out

#         outneg = self.convneg (x=x, edge_index=edge_index,edge_weight=(edge_attr[:,0]+1))
#         outpos = self.convpos (x=x, edge_index=edge_index,edge_weight=(edge_attr[:,1]+1))

#         out = torch.cat((outpos,outneg),dim=-1)    

# class GReTo(torch.nn.Module):
#     def __init__(self):
#         super(GReTo, self).__init__()
#         self.lstm1 = torch.nn.LSTM(num_layers = 1,batch_first=batch_first,
#                                bidirectional=bidirectional,input_size=input_size,hidden_size = hidden_size)
#         self.convneg = GCNConv(12, 32)
#         self.convpos = GCNConv(12, 32)
# #         self.mlp = Linear(hidden_size,1)
#         self.mlp = Linear(64,1)

#     def forward(self, batch):
#         x, edge_index,edge_attr=batch.x,batch.edge_index,batch.edge_attr
#         outneg = self.convneg (x=x, edge_index=edge_index,edge_weight=(edge_attr[:,0]+1))
#         outpos = self.convpos (x=x, edge_index=edge_index,edge_weight=(edge_attr[:,1]+1))
#         out = torch.cat((outpos,outneg),dim=-1)
#         out = out.relu()
#         out,(hn,cn) = self.lstm1(out.unsqueeze(-1))
#         out = out[:,-1,:].relu()

#         out = self.mlp(out)
#         return out


# In[22]:


# enable_padding
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        if enable_padding == True:
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[: , : , : -self.__padding]
        return result


# In[23]:


class First_Layer(torch.nn.Module):
    # temporal dimension 12->10
    def __init__(self,in_channels=1,out_channels=64):
        super(First_Layer, self).__init__()
        self.conv = CausalConv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=3)
    def forward(self, x):
        # x:bat*n  channels time
        out = self.conv(x)
        x=x.relu()
        return out
class Out_Layer(torch.nn.Module):
    # 
    def __init__(self,in_channels=64,out_channels=1):
        super(Out_Layer, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels=in_channels,out_channels=in_channels*2,kernel_size=2,padding=0)
        self.mlp=Linear(in_channels*2,out_channels)
    def forward(self, x):
        # x:bat*n  channels time
        out=self.conv(x).squeeze(-1).relu()
        out=self.mlp(out)
        return out  
class Time_Conv(torch.nn.Module):
    ### GTU tanh(x) · sigmoid(x)
    ### time -2
    def __init__(self,in_channels=16 ,out_channels=64,kernel_size=3):
        super(Time_Conv,self).__init__()
        self.conv1 = CausalConv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size)
        self.conv2 = CausalConv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size)
    def forward(self,x):
        out1=self.conv1(x).tanh()
        out2=self.conv2(x).sigmoid()
        out=out1*out2
        return out
        
class ST_Block(torch.nn.Module):
    def __init__(self,tlen_spconv,in_channels=64,hidden_channels=16,out_channels=64,gnn_type='GCN'):
        #1st timeconv -2 2nd timeconv-2
        self.gnn_type=gnn_type
        super(ST_Block, self).__init__()
        self.timeconv1 = Time_Conv(in_channels=in_channels,out_channels=hidden_channels,kernel_size=3)
        if gnn_type=='GCN':
            self.spatialconv = GCNConv(in_channels=hidden_channels*tlen_spconv, out_channels=hidden_channels*tlen_spconv)
        elif gnn_type=='GAT':
            self.spatialconv = GATConv(hidden_channels*tlen_spconv,  hidden_size/8,heads=8)
        elif gnn_type=='GReTo':
            self.spatialconv_pos = GCNConv(in_channels=hidden_channels*tlen_spconv, out_channels=hidden_channels*tlen_spconv)
            self.spatialconv_neg = GCNConv(in_channels=hidden_channels*tlen_spconv, out_channels=hidden_channels*tlen_spconv)
            self.gremlp = Linear(hidden_channels*2,hidden_channels)
            self.k = 3
            self.psimlp = torch.nn.Sequential(Linear(3,64),torch.nn.ReLU(),Linear(64,self.k))
        self.timeconv2 = Time_Conv(in_channels=hidden_channels,out_channels=out_channels,kernel_size=3)
        
    def forward(self, x,edge_index,edge_attr,ND_t):
        # x:bat*n  channels time
        out1 = self.timeconv1(x)
        out1 = self.timeconv1(x)
        d0,d1,d2=out1.shape[0],out1.shape[1],out1.shape[2]
        if self.gnn_type=='GReTo':
            # b*n channels timelen     
            psi = self.psimlp(ND_t).unsqueeze(-1).repeat_interleave(64,dim=-1).unsqueeze(-1).repeat_interleave(d2,dim=-1)
            out=[out1]
            for i in range(self.k):
                out[i]=out[i].reshape(d0,-1)
#                 print(out[i].shape)
                out_pos  =self.spatialconv_pos(edge_index=edge_index,x=out[i],edge_weight=(edge_attr[:,0]+1)).reshape(d0,d1,d2).relu()
                out.append(out_pos)
                if i == 0:
#                     print(psi.shape,out2.shape)
                    out_pos_psi = psi[:,i]*out_pos
                else:
                    out_pos_psi += psi[:,i]*out_pos
            out_neg  =self.spatialconv_neg(edge_index=edge_index,x=out[0],edge_weight=(edge_attr[:,1]+1)).reshape(d0,d1,d2).relu()
            out2 = torch.cat([out_pos_psi,out_neg],dim=1)
            out2 = self.gremlp(out2.transpose(1,2)).transpose(1,2).relu()
                       
        else:
            out1 = out1.reshape(d0,-1)
            out2   =self.spatialconv(edge_index=edge_index,x=out1).reshape(d0,d1,d2).relu()
        out3 = self.timeconv2(out2 )
        return out3

class GReTo(torch.nn.Module):
    def __init__(self,in_channels=64,hidden_channels=16,out_channels=64,gnn_type='GCN'):
        super(GReTo, self).__init__()
        self.firstlayer = First_Layer(in_channels=1,out_channels=in_channels)#t-2 t-4 t-4 t-1
        self.stblock1 = ST_Block(tlen_spconv=8,in_channels=in_channels,hidden_channels=hidden_channels,
                                 out_channels=out_channels,gnn_type=gnn_type)
        self.stblock2 = ST_Block(tlen_spconv=4,in_channels=in_channels,hidden_channels=hidden_channels,
                                 out_channels=out_channels,gnn_type=gnn_type)  
        self.outlayer = Out_Layer(in_channels=out_channels,out_channels=1)

    def forward(self, batch):
        x,ND_t, edge_index,edge_attr=batch.x[:,:time_len],batch.x[:,time_len:],batch.edge_index,batch.edge_attr
        out = self.firstlayer(x.unsqueeze(1))
        out=self.stblock1(x=out,edge_index=edge_index,edge_attr=edge_attr,ND_t=ND_t)
        out=self.stblock2(x=out,edge_index=edge_index,edge_attr=edge_attr,ND_t=ND_t)
        out=self.outlayer(out)
        return out


# In[24]:


model = GReTo(in_channels=64,hidden_channels=64,out_channels=64,gnn_type='GReTo')
print(model)


# In[25]:


optimizer =  torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)


# ### Training

# In[26]:


device = "cuda:1"
model = model.to(device)
loss_fn = torch.nn.L1Loss()


# In[27]:


def train_one_epoch(dataloader,optimizer,scheduler):
    model.train()
    total_loss = 0 
    total_batch = 0
    for batch in tqdm.tqdm(dataloader):
        batch = batch.to(device)
        out = model(batch)
        predict = scaler.inverse_transform(out)
        loss = loss_fn(batch.y,predict.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss+=loss.detach().cpu()
        total_batch+=1
    scheduler.step()
    return total_loss/total_batch


# In[28]:


def valid(dataloader):
    model.eval()
    total_loss = 0 
    total_batch = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch)
            predict = scaler.inverse_transform(out)
            loss = loss_fn(batch.y,predict.reshape(-1))
            total_loss+=loss.detach().cpu()
            total_batch+=1
    return total_loss/total_batch


# In[29]:


def test(dataloader):
    model.eval()
    total_loss = 0 
    total_batch = 0
    pred_total = []
    label_total = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch)
            predict = scaler.inverse_transform(out).reshape(-1)
            pred_total.append(predict)
            label_total.append(batch.y)
    pred_total = torch.cat(pred_total,dim=0)
    label_total = torch.cat(label_total,dim=0)
    metrics = metric(pred_total,label_total)
    log = 'Evaluate best model on test data, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format( metrics[0], metrics[1], metrics[2]))


# In[30]:


min_valid_loss = 1e9
MODEL_PATH='./model/best_valid_wwwk=3.pt'
for epoch in range(150):
    train_loss = train_one_epoch(train_dataloader,optimizer,scheduler)
    valid_loss = valid(val_dataloader)
    if min_valid_loss>valid_loss:
        torch.save(model,MODEL_PATH)
        min_valid_loss = valid_loss
    print('{}-th epoch | train_loss={} | valid_loss={} | best_valid_loss={}'
          .format(epoch+1,train_loss,valid_loss,min_valid_loss))
model = torch.load(MODEL_PATH)
test(test_dataloader)


# In[31]:


model = torch.load(MODEL_PATH)
test(test_dataloader)


# In[32]:


# GReTo
# with 错的M
# 100-th epoch | train_loss=2.380671977996826 | valid_loss=2.3669283390045166 | best_valid_loss=2.335008144378662
# Evaluate best model on test data, Test MAE: 2.6420, Test MAPE: 0.0630, Test RMSE: 5.2794

# with 对的M
# 100-th epoch | train_loss=2.1417477130889893 | valid_loss=2.127549648284912 | best_valid_loss=2.118959665298462
# Evaluate best model on test data, Test MAE: 2.4024, Test MAPE: 0.0594, Test RMSE: 4.2679

# 100-th epoch | train_loss=2.0668952465057373 | valid_loss=1.9732215404510498 | best_valid_loss=1.9636491537094116
# Evaluate best model on test data, Test MAE: 2.2462, Test MAPE: 0.0548, Test RMSE: 3.9049

# 3 10 0.9
# 100-th epoch | train_loss=1.9602874517440796 | valid_loss=1.9380944967269897 | best_valid_loss=1.9380944967269897
# Evaluate best model on test data, Test MAE: 2.2124, Test MAPE: 0.0543, Test RMSE: 3.8694

# 3 10 0.9 150
# 150-th epoch | train_loss=1.809723138809204 | valid_loss=1.804538607597351 | best_valid_loss=1.7984442710876465
# Evaluate best model on test data, Test MAE: 2.0633, Test MAPE: 0.0507, Test RMSE: 3.6912


# In[33]:


batch = next(iter(train_dataloader))
print(batch)
print(batch.edge_attr[:,0].shape)

