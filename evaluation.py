from bayes_opt import BayesianOptimization
import torch
from Networks import train_eval_Net, Net
from data import getSTL10
import pyKriging
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

hyperpar_domains = {'learning_rate': (0.00001, 0.1), 'weight_decay': (0, 0.001)}
trainloader, validloader, _ = getSTL10(True)

post_train_losses = []

def evaluateBayes(learning_rate, weight_decay):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)

    _, validation_loss = train_eval_Net(model, 10, trainloader, validloader, learning_rate, weight_decay, device)

    post_train_losses.append(validation_loss)

    return -validation_loss


#Kriging Optimization
sp = samplingplan(2)
X = sp.optimallhc(20)
testfun = pyKriging.testfunctions().branin
y = testfun(X)
k = kriging(X,y, testfunction=-evaluateBayes, name='simple')
k.train()




# Bayesian Optimization
opt_bys = BayesianOptimization(f=evaluateBayes, pbounds=hyperpar_domains)
opt_bys.maximize(3, 10)
print('Result with Bayes Optimizer:'+str(opt_bys.max))
print(post_train_losses)
