import matplotlib.pyplot as plt
import torch
import pickle
import time
import torch.nn as nn

# basic functions
def target_function1(x):
    return torch.exp(-x*4)*torch.cos(x*20)

def get_data(fn,N):
    x = torch.rand(N)
    y = fn(x)
    return x,y

def choice(x,y,N): # (num_train,1),(num_train,1),int
    idx = torch.randint(0,x.size(0),(N,))
    return x[idx],y[idx] # (N,1)

def nm(x):
    if 'cpu' in x.device.type:
        return x.detach().numpy()
    else:
        return x.detach().cpu().numpy()

def draw(load_name,save_name):
    with open(load_name,mode='rb') as f:
        load_log = pickle.load(f)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5),dpi=100)
    fig.suptitle(load_name)
    ax[0].plot(load_log['eval'])
    ax[0].set_xlabel('Number of data collected')
    ax[0].set_ylabel('MSE over 100k samples')
    ax[1].plot(load_log['all_x'],load_log['all_y_mean'],label='prediction_mean',color='red',lw=1)
    ax[1].fill_between(load_log['all_x'],load_log['all_y_mean']-load_log['all_y_unce'],
                       load_log['all_y_mean']+load_log['all_y_unce'],label='prediction_unce',color='pink',alpha=0.6)
    ax[1].scatter(load_log['train_x'],load_log['train_y'],label='train data',color='blue',s=2,zorder=100)
    ax[1].legend()
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    plt.savefig(save_name,dpi=100,format = 'png')
    plt.close()

def eval_model(model):
    x,y = get_data(target_function1,100000)
    x = x.to(model.device)
    y = y.to(model.device)
    y_pred = model.pred_mean(x)
    return torch.pow(y - y_pred,2).mean()

def make_log(save_name,eva,train_x,train_y,model):
    log = {}
    log['eval'] = eva
    log['train_x'] = nm(train_x)
    log['train_y'] = nm(train_y)
    log['all_x'] = nm(torch.linspace(0, 1, steps=10000))
    log['all_y'] = nm(target_function1(torch.linspace(0, 1, steps=10000)))
    log['all_y_mean'] = nm(model.pred_mean(torch.linspace(0, 1, steps=10000).to(model.device)))
    log['all_y_unce'] = nm(model.pred_unce(torch.linspace(0, 1, steps=10000).to(model.device)))
    with open(save_name,mode='wb') as f:
        pickle.dump(log,f)


