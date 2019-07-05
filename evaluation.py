from bayes_opt import BayesianOptimization
import torch
from Networks import train_eval_Net, Net
from data import getSTL10


hyperpar_domains = {'learning_rate': (0.00001, 0.1), 'weight_decay': (0, 0.001)}
trainloader, validloader, _ = getSTL10(True)


def evaluateBayes(learning_rate, weight_decay):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)

    _, validation_loss = train_eval_Net(model, 30, trainloader, validloader, learning_rate, weight_decay, device)

    return -validation_loss


# Bayesian Optimization
opt_bys = BayesianOptimization(f=evaluateBayes, pbounds=hyperpar_domains)
opt_bys.maximize(3, 20)
print('Result with Bayes Optimizer:'+str(opt_bys.max))
