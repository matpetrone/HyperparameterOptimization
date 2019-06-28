from torch.utils.data import Dataset, DataLoader
import torchvision

def getImageNet(val = True):



    trainset = torchvision.datasets.ImageNet