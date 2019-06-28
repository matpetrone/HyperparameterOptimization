import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter


class Net(nn.Module):


    def __init__(self):
        self.conv1 = nn.Conv2d(3,16,3)
        self.conv2 = nn.Conv2d(16, 16,3)
        self.maxP1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16,32,3)
        self.conv4 = nn.Conv2d(32,32,3)
        self.maxP2 = nn.MaxPool2d(2,2)
        self.

