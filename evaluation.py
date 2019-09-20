from bayes_opt import BayesianOptimization
import torch
from Networks import train_eval_Net, Net
from data import getSTL10
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

hyperpar_domains = {'learning_rate': (0.0001, 0.01), 'weight_decay': (0.0001, 0.1)}
trainloader, validloader, _ = getSTL10(True)

post_train_losses = []
post_train_val = []

def evaluateBayes(learning_rate, weight_decay):

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)

    _, validation_loss, validation_acc = train_eval_Net(model, 80, trainloader, validloader, learning_rate, weight_decay, device)

    post_train_losses.append(validation_loss)
    post_train_val.append(validation_acc)

    return -validation_loss





# Bayesian Optimization
#opt_bys = BayesianOptimization(f=evaluateBayes, pbounds=hyperpar_domains)
#opt_bys.maximize(3, 30)
#print('Result with Bayes Optimizer:'+str(opt_bys.max))
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
train_eval_Net(model, 200, trainloader, validloader, 0.0001, 0.0, device)
print('vector of min loss for every iteration:'+post_train_losses)
print('vector of max accuracy for every iteration:'+post_train_val)
