import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter


class Net(nn.Module):

    def __init__(self):
        self.conv1 = nn.Conv2d(3,16,3)
        self.conv2 = nn.Conv2d(16, 16,3)
        self.maxP1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16,32,3)
        self.conv4 = nn.Conv2d(32,32,3)
        self.maxP2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32 * 21 * 21, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.maxP1(self.conv2(self.conv1(x))))
        x = F.relu(self.maxP2(self.conv4(self.conv3(x))))
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def trainingNet(net, epochs, trainloader, learn_rate, momentum):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=momentum)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

def valNet(net, validloader):

    correct = 0
    total = 0
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print('Accuracy of the network: %d %%' % (accuracy))