from utils import *
from models import RandomChoice,EquallySpacedChoice,EquallySpacedChoiceXY
from models import SimpleEnsembleModel, SimpleEnsembleModelBoot,QRNNModel,DiscrimModel,DiscrimModelNormal


cand = 10000
device = torch.device("cuda:0")
num_head = 3
models = [RandomChoice(device),EquallySpacedChoice(device),EquallySpacedChoiceXY(device),
        SimpleEnsembleModel(device,num_head), SimpleEnsembleModelBoot(device,num_head),
        QRNNModel(device,num_head),DiscrimModel(device),DiscrimModelNormal(device)] 
models = [DiscrimModel(device)]
for model in models:
    print(model.name)
    eva = []
    for i in range(5):
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
    save_name = 'data/test_'+model.name+'.pickle'
    make_log(save_name,eva,train_x,train_y,model)
    draw(save_name,'data/'+model.name+'.png')
