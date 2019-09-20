from bayes_opt import BayesianOptimization
import torch
from Networks import train_eval_Net, Net
from data import getSTL10

hyperpar_domains = {'learning_rate': (0.0001, 0.01), 'weight_decay': (0.0, 0.001)}
trainloader, validloader, _ = getSTL10(True)

post_train_losses = []
post_train_acc = []
post_val_losses = []
post_val_acc = []

def evaluateBayes(learning_rate, weight_decay):

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)

    _, validation_loss, validation_acc, train_loss, train_acc = train_eval_Net(model, 80, trainloader, validloader, learning_rate, weight_decay, device)

    post_train_losses.append(train_loss)
    post_train_acc.append(train_acc)
    post_val_losses.append(validation_loss)
    post_val_acc.append(validation_acc)

    return -validation_loss





# Bayesian Optimization
opt_bys = BayesianOptimization(f=evaluateBayes, pbounds=hyperpar_domains)
opt_bys.maximize(3, 30)
print('Result with Bayes Optimizer:'+str(opt_bys.max))
#device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#model = Net().to(device)
#train_eval_Net(model, 200, trainloader, validloader, 0.0001, 0.0, device)
print('vector of min train loss for every iteration:'+str(post_train_losses))
print('vector of max train accuracy for every iteration:'+str(post_train_acc))
print('vector of max validation loss for every iteration:'+str(post_val_losses))
print('vector of max validation accuracy for every iteration:'+str(post_val_acc))


