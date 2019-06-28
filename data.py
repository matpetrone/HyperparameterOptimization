from torch.utils.data import Dataset, DataLoader
import torchvision

def getImageNet(val = True):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainSize = 10000
    if val:
        trainSize = 7500
        valSize = 2500

    trainset = torchvision.datasets.ImageNet(root='./data', split='train', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, sampler=ChunkSampler(trainSize, 0))

    testset = torchvision.datasets.ImageNet(root="./data", split='val', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False)

    if val:
        validationloader = torch.utils.data.DataLoader(trainset, shuffle=False, sampler=ChunkSampler(valSize, trainSize))
        return trainloader, validationloader, testloader

    return trainloader, testloader