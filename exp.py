from utils import *
from models import RandomChoice,EquallySpacedChoice,EquallySpacedChoiceXY
from models import SimpleEnsembleModel, SimpleEnsembleModelBoot,QRNNModel,DiscrimModel,DiscrimModelNormal

import argparse
import requests
import os

parser = argparse.ArgumentParser()
parser.add_argument("--cuda",type=int,default=0)
args = parser.parse_args()


def notify(message = 'done'):
    line_notify_token = 'UhzRLKWEpku0nNCv0x1XqDEDRIOpLSqNun9AgGJOXS7'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    message += '@'+ os.uname()[1]
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)



device = torch.device("cuda:{}".format(int(args.cuda)))
if args.cuda==1:
    models =  [RandomChoice(device),EquallySpacedChoice(device),EquallySpacedChoiceXY(device),
               SimpleEnsembleModel(device,3),SimpleEnsembleModel(device,7)]
elif args.cuda==2:
    models =  [SimpleEnsembleModelBoot(device,3),SimpleEnsembleModelBoot(device,7),
               DiscrimModel(device)]
elif args.cuda==3:
    models =  [DiscrimModelNormal(device),QRNNModel(device,3),QRNNModel(device,7),
               QRNNModel(device,11),QRNNModel(device,21)]
else:
    raise

cand = 10000
trials = ['trial1','trial2','trial3','trial4','trial5',]
for t in trials:
    for model in models:
        notify(t+' start on '+str(args.cuda)+' :'+model.name)
        eva = []
        for i in range(40):
            x,y = get_data(target_function1,cand)
            x = x.to(device)
            y = y.to(device)
            max_id = model.pred_unce(x).argmax()
            if i ==0:
                train_x = x[max_id].view(1,1)
                train_y = y[max_id].view(1,1)
            else:
                train_x = torch.cat((train_x,x[max_id].view(1,1)))
                train_y = torch.cat((train_y,y[max_id].view(1,1)))
            model.learn(train_x,train_y)
            eva.append(eval_model(model))
            if i == 4 or i==14:
                save_head = 'data/'+model.name+'_itr'+str(i)+'_'+t
                make_log(save_head+'.pickle',eva,train_x,train_y,model)
                draw(save_head+'.pickle',save_head+'.png')
        save_head = 'data/'+model.name+'_itr'+str(i)+'_'+t
        make_log(save_head+'.pickle',eva,train_x,train_y,model)
        draw(save_head+'.pickle',save_head+'.png')
