from torch.utils.data import Dataset, DataLoader, sampler
import torchvision
import torch
from torchvision import transforms


class IndexSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


def getSTL10(val=True):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainSize = 5000
    if val:
        trainSize = 5000
        valSize = 4000

    trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size= 32,shuffle=False, sampler=IndexSampler(trainSize, 0))

    testset = torchvision.datasets.STL10(root="./data", split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, sampler=IndexSampler(4000, 0))

    if val:
        validationloader = torch.utils.data.DataLoader(testset, batch_size=60, shuffle=False, sampler=IndexSampler(valSize, 4000))
        return trainloader, validationloader, testloader

    return trainloader, testloader
