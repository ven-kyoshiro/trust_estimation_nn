import matplotlib.pyplot as plt
import torch
import pickle
import time
import torch.nn as nn
from utils import *


MAX_TIME=10
MAX_ITR = 1000000


# models
class TwoLayerNetRes(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNetRes, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)
    def forward(self,x):
        h_relu1 = self.linear1(x).clamp(min=0)
        h_relu2 = self.linear2(h_relu1).clamp(min=0)
        y_pred = self.linear3(h_relu2+h_relu1)
        return y_pred


class AbstModel(object):
    def __init__(self,device):
        self.device = device
        self.name = 'abst2_model'

    def pred_mean(self,x): # (10000)
        return torch.randn(x.shape[0]).to(self.device) # (10000)

    def pred_unce(self,x): # (10000)
        return torch.randn(x.shape[0]).to(self.device) # (10000)

    def learn(self,x,y): # x:(itr,1), y:(itr,1)
        # learn in 10 sec
        pass


class RandomChoice(AbstModel):
    def __init__(self,device,H=600,b_size=128):
        super().__init__(device)
        self.H = H
        self.b_size = b_size
        self.name = 'random'
        self.model = TwoLayerNetRes(1,H,1).to(self.device)
        self.optim = torch.optim.Adadelta(self.model.parameters())
        
    def pred_mean(self,x): # (10000)
        return self.model(x.unsqueeze(-1)).squeeze(1).detach() # (10000)

    def learn(self,x,y):
        st = time.time()
        for i in range(MAX_ITR):
            xx,yy = choice(x,y,self.b_size)
            y_ = yy.unsqueeze(-1)
            y_pred = self.model(xx.unsqueeze(-1))
            loss =torch.pow(y_pred-y_,2).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if time.time() - st > MAX_TIME:
                break

class EquallySpacedChoice(AbstModel):
    def __init__(self,device,H=600,b_size=128):
        super().__init__(device)
        self.H = H
        self.b_size = b_size
        self.name = 'equally_x'
        self.model = TwoLayerNetRes(1,H,1).to(self.device)
        self.optim = torch.optim.Adadelta(self.model.parameters())

    def pred_mean(self,x): # (10000)
        return self.model(x.unsqueeze(-1)).squeeze(1).detach() # (10000)

    def pred_unce(self,x):# (10000)
        if hasattr(self,'train_x'):
            ret = torch.zeros(x.shape)
            for i,xx in enumerate(x):
                ret[i] = (self.train_x - xx).abs().min()
            return ret
        else:
            min_id = x.argmin()
            ret = torch.zeros(x.shape)
            ret[min_id]=1.0
            return  ret

    def learn(self,x,y):# (itr,1), (itr,1)
        st = time.time()
        for i in range(MAX_ITR):
            xx,yy = choice(x,y,self.b_size)
            y_ = yy.unsqueeze(-1)
            y_pred = self.model(xx.unsqueeze(-1))
            loss =torch.pow(y_pred-y_,2).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if time.time() - st > MAX_TIME:
                break
        self.train_x = x


class EquallySpacedChoiceXY(AbstModel):
    def __init__(self,device,H=600,b_size=128):
        super().__init__(device)
        self.H = H
        self.b_size = b_size
        self.name = 'equally_xy'
        self.model = TwoLayerNetRes(1,H,1).to(self.device)
        self.optim = torch.optim.Adadelta(self.model.parameters())

    def pred_mean(self,x): # (10000)
        return self.model(x.unsqueeze(-1)).squeeze(1).detach() # (10000)

    def pred_unce(self,x):# (10000)
        if hasattr(self,'train_x'):
            y = self.model(x.unsqueeze(-1)).squeeze(1)
            ret = torch.zeros(x.shape)
            for i,(xx,yy) in enumerate(zip(x,y)):
                ret[i] = (torch.pow(self.train_x - xx,2)+torch.pow(self.train_y - yy,2)).min()
            return ret
        else:
            min_id = x.argmin()
            ret = torch.zeros(x.shape)
            ret[min_id]=1.0
            return  ret

    def learn(self,x,y):# (itr,1), (itr,1)
        st = time.time()
        for i in range(MAX_ITR):
            xx,yy = choice(x,y,self.b_size)
            y_ = yy.unsqueeze(-1)
            y_pred = self.model(xx.unsqueeze(-1))
            loss =torch.pow(y_pred-y_,2).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if time.time() - st > MAX_TIME:
                break
        self.train_x = x
        self.train_y = y


class SimpleEnsembleModel(AbstModel):
    def __init__(self,device,num_head,H=600,b_size=128):
        super().__init__(device)
        self.H = H
        self.b_size = b_size
        self.num_head = num_head
        self.name = 'ensemble_of_'+str(num_head)
        self.models = [TwoLayerNetRes(1,H,1).to(self.device) for _ in range(self.num_head)]
        self.optims = [torch.optim.Adadelta(mo.parameters()) for mo in self.models]

    def pred_mean(self,x): # (10000)
        heads = torch.cat([mo(x.unsqueeze(-1)).detach() for mo in self.models],dim=1)
        return heads.mean(dim=1) # (10000)

    def pred_unce(self,x):# (10000)
        heads = torch.cat([mo(x.unsqueeze(-1)).detach() for mo in self.models],dim=1)
        return  heads.std(dim=1) # (10000)

    def learn(self,x,y):# (itr,1), (itr,1)
        for mo,op in zip(self.models,self.optims):
            st = time.time()
            for i in range(MAX_ITR):
                xx,yy = choice(x,y,self.b_size)
                y_ = yy.unsqueeze(-1)
                y_pred = mo(xx.unsqueeze(-1))
                loss =torch.pow(y_pred-y_,2).mean()
                op.zero_grad()
                loss.backward()
                op.step()
                if time.time() - st > MAX_TIME:
                    break


def choice_mod(x,y,N,m,n): #mod m != n
    idx = torch.randint(0,x.size(0),(2*N,))
    idx_mod = []
    while True:
        idx_mod += [ii for ii in idx if ii%m!=n]
        if len(idx_mod)>=N:
            break
        idx = torch.randint(0,x.size(0),(2*N,))
    idx = idx_mod[:N]
    return x[idx],y[idx]

class SimpleEnsembleModelBoot(AbstModel):
    def __init__(self,device,num_head,H=600,b_size=128):
        super().__init__(device)
        self.H = H
        self.b_size = b_size
        self.num_head = num_head
        self.name = 'ensemble_of_'+str(num_head)+'_boot'
        self.models = [TwoLayerNetRes(1,H,1).to(self.device) for _ in range(self.num_head)]
        self.optims = [torch.optim.Adadelta(mo.parameters()) for mo in self.models]

    def pred_mean(self,x): # (10000)
        heads = torch.cat([mo(x.unsqueeze(-1)).detach() for mo in self.models],dim=1)
        return heads.mean(dim=1) # (10000)

    def pred_unce(self,x):# (10000)
        heads = torch.cat([mo(x.unsqueeze(-1)).detach() for mo in self.models],dim=1)
        return  heads.std(dim=1) # (10000)

    def learn(self,x,y):# (itr,1), (itr,1)
        for j, (mo,op) in enumerate(zip(self.models,self.optims)):
            if len(x)==1 and j==0:
                break
            st = time.time()
            for i in range(MAX_ITR):
                xx,yy = choice_mod(x,y,self.b_size,self.num_head,j)
                y_ = yy.unsqueeze(-1)
                y_pred = mo(xx.unsqueeze(-1))
                loss =torch.pow(y_pred-y_,2).mean()
                op.zero_grad()
                loss.backward()
                op.step()
                if time.time() - st > MAX_TIME:
                    break


EPS = 1e-7
class QRNNModel(AbstModel):
    def __init__(self,device,num_head,H=600,b_size=128):
        super().__init__(device)
        self.H = H
        self.device=device
        self.b_size = b_size
        assert num_head%2!=0, 'even number'
        self.quantiles = torch.tensor(
            [i/(num_head+1) for i in range(1,num_head+1)],dtype=torch.float).to(device)
        self.num_head = num_head
        self.name = 'qrnn_of_'+str(num_head)
        self.model = TwoLayerNetRes(1,H,num_head).to(self.device) # (10000,num_head)
        self.optim = torch.optim.Adadelta(self.model.parameters())
        self.center_id = int(num_head/2)
        # make q_head
        width = int((num_head-1)/2)
        self.q_head = torch.zeros((num_head,num_head),dtype=torch.float).to(device)
        self.q_head[self.center_id,:] = 1
        for i in range(0,width):
            self.q_head[i,:i+1] = 1
            self.q_head[num_head-i-1,num_head-i-1:] = 1
        self.q_head = self.q_head.to(device)
                              
    def pred_quantile(self,x):# (10000)
        model_out = self.model(x.unsqueeze(-1)).squeeze(1) # (10000,num_head)
        #l = model_out[:,:self.center_id].clamp(min=EPS)
        l = model_out[:,:self.center_id].abs() + EPS
        m = model_out[:,self.center_id].unsqueeze(-1)
        r = model_out[:,self.center_id+1:].abs() + EPS
        #r = model_out[:,self.center_id+1:].clamp(min=EPS)
        return torch.cat((-1*l,m,r),dim=1)@self.q_head #(10000,num_head)                
        
    def pred_mean(self,x): # (10000)
        model_out = self.model(x.unsqueeze(-1)).squeeze(1)
        return (model_out[:,self.center_id].detach()) # (10000)

    def pred_unce(self,x):# (10000)
        model_out = self.model(x.unsqueeze(-1)).squeeze(1)
        l = model_out[:,:self.center_id].abs() + EPS
        r = model_out[:,self.center_id+1:].abs() + EPS
        aa = torch.log(torch.cat((l,r),dim=1))
        bb = torch.log(EPS*torch.ones(1,dtype=torch.float)).to(self.device)
        a = (aa-bb).sum(dim=1)
        return a/100 # (10000)

    def learn(self,x,y):# (itr,1), (itr,1)
        st = time.time()
        for i in range(MAX_ITR):
            xx,yy = choice(x,y,self.b_size)
            quan_out = self.pred_quantile(xx)
            error = (yy.expand(yy.shape[0],self.num_head) - quan_out)
            loss = torch.max(error* self.quantiles ,error*(self.quantiles -1)).sum()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if time.time() - st > MAX_TIME*2:
                break

class DiscrimModel(AbstModel):
    def __init__(self,device,H=600,b_size=128):
        super().__init__(device)
        self.H = H
        self.b_size = b_size
        self.name = 'discrim_model'
        self.model = TwoLayerNetRes(1,H,1).to(self.device)
        self.discrim = TwoLayerNetRes(2,H,1).to(self.device) # gene =1, real=0
        self.optim = torch.optim.Adadelta(self.model.parameters())
        self.optim_discrim = torch.optim.Adadelta(self.discrim.parameters())
        
    def pred_mean(self,x): # (10000)
        return self.model(x.unsqueeze(-1)).squeeze(1).detach() # (10000)
    
    def discriminate(self,xy): #(10000,2)
         return 1/(1+torch.exp(-self.discrim(xy))) # sigmoid,(10000,1)

    def pred_unce(self,x):# (10000)
        y = self.model(x.unsqueeze(-1)).detach()
        xy = torch.cat((x.unsqueeze(-1),y),dim=1)
        if hasattr(self, 'gene_xy'):
            self.gene_xy = torch.cat((self.gene_xy,xy))
        else:
            self.gene_xy = xy
        return self.discriminate(xy).squeeze(1) # (10000)

    def learn(self,x,y):# (itr,1), (itr,1)
        st = time.time()
        # learn model
        for i in range(MAX_ITR):
            xx,yy = choice(x,y,self.b_size)
            y_ = yy.unsqueeze(-1)
            y_pred = self.model(xx.unsqueeze(-1))
            loss =torch.pow(y_pred-y_,2).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if time.time() - st > MAX_TIME:
                break
        st = time.time()
        # learn discrim
        for i in range(MAX_ITR):
            d_y1 = torch.ones(int(self.b_size/2),dtype=torch.float).to(self.device)
            d_x1 = self.gene_xy[torch.randint(0,self.gene_xy.shape[0],(int(self.b_size/2),))]
            
            d_xx,d_yy = choice(x,y,int(self.b_size/2))
            d_x2 = torch.cat((d_xx,d_yy),dim=1)
            d_y2 = torch.zeros(int(self.b_size/2),dtype=torch.float).to(self.device)
            d_y1pred = self.discriminate(d_x1).squeeze(1) #ret (10000,1)
            d_y2pred = self.discriminate(d_x2).squeeze(1)
            # BCE
            loss_discrim =(-d_y1*torch.log(d_y1pred.clamp(min=EPS))\
                       -(1-d_y1)*torch.log((1-d_y1pred).clamp(min=EPS))\
                           -d_y2*torch.log(d_y2pred.clamp(min=EPS))\
                       -(1-d_y2)*torch.log((1-d_y2pred).clamp(min=EPS))).mean()
            self.optim_discrim.zero_grad()
            loss_discrim.backward()
            self.optim_discrim.step()
            if time.time() - st > MAX_TIME:
                break


class DiscrimModelNormal(AbstModel):
    def __init__(self,device,H=600,b_size=128):
        super().__init__(device)
        self.H = H
        self.b_size = b_size
        self.name = 'discrim_model_normal'
        self.model = TwoLayerNetRes(1,H,1).to(self.device)
        self.discrim = TwoLayerNetRes(2,H,1).to(self.device) # gene =1, real=0
        self.optim = torch.optim.Adadelta(self.model.parameters())
        self.optim_discrim = torch.optim.Adadelta(self.discrim.parameters())
        
    def pred_mean(self,x): # (10000)
        return self.model(x.unsqueeze(-1)).squeeze(1).detach() # (10000)
    
    def discriminate(self,xy): #(10000,2)
         return 1/(1+torch.exp(-self.discrim(xy))) # sigmoid,(10000,1)

    def pred_unce(self,x):# (10000)
        y = self.model(x.unsqueeze(-1)).detach()
        xy = torch.cat((x.unsqueeze(-1),y),dim=1)
        if hasattr(self, 'gene_xy'):
            self.gene_xy = torch.cat((self.gene_xy,xy))
        else:
            self.gene_xy = xy
        return self.discriminate(xy).squeeze(1) # (10000)

    def learn(self,x,y):# (itr,1), (itr,1)
        st = time.time()
        # learn model
        for i in range(MAX_ITR):
            xx,yy = choice(x,y,self.b_size)
            y_ = yy.unsqueeze(-1)
            y_pred = self.model(xx.unsqueeze(-1))
            loss =torch.pow(y_pred-y_,2).mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if time.time() - st > MAX_TIME:
                break
        st = time.time()
        # learn discrim
        for i in range(MAX_ITR):
            d_y1 = torch.ones(int(self.b_size/2),dtype=torch.float).to(self.device)
            # false data:label=1
            # normal distribution
            if x.shape[0] == 1:
                nx = (torch.randn((int(self.b_size/2),1))+x.mean()).to(self.device)
                ny = (torch.randn((int(self.b_size/2),1))+y.mean()).to(self.device)
                d_x1 = torch.cat((nx,ny),dim=1)
            else:
                nx = (torch.randn((int(self.b_size/2),1))*x.std()+x.mean()).to(self.device)
                ny = (torch.randn((int(self.b_size/2),1))*y.std()+y.mean()).to(self.device)
                d_x1 = torch.cat((nx,ny),dim=1)
            
            # real data:label=0
            d_xx,d_yy = choice(x,y,int(self.b_size/2))
            d_x2 = torch.cat((d_xx,d_yy),dim=1)
            d_y2 = torch.zeros(int(self.b_size/2),dtype=torch.float).to(self.device)
            d_y1pred = self.discriminate(d_x1).squeeze(1) #ret (10000,1)
            d_y2pred = self.discriminate(d_x2).squeeze(1)
            # BCE
            loss_discrim =(-d_y1*torch.log(d_y1pred.clamp(min=EPS))\
                       -(1-d_y1)*torch.log((1-d_y1pred).clamp(min=EPS))\
                           -d_y2*torch.log(d_y2pred.clamp(min=EPS))\
                       -(1-d_y2)*torch.log((1-d_y2pred).clamp(min=EPS))).mean()
            self.optim_discrim.zero_grad()
            loss_discrim.backward()
            self.optim_discrim.step()
            if time.time() - st > MAX_TIME:
                break
