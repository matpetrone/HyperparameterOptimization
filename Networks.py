import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.maxP1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.maxP2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 21 * 21, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.maxP1(self.conv2(self.conv1(x))))
        x = F.relu(self.maxP2(self.conv4(self.conv3(x))))
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.reset_parameters()
        if isinstance(m, nn.Linear):
            m.weight.data.fill_(0.01)
            m.bias.data.fill_(0.01)
        # net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

    def init_w(self):
        self.apply(self.weights_init)


def train_eval_Net(net, epochs, trainloader, validloader, learn_rate, weight_decay, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learn_rate, weight_decay=weight_decay)
    net.init_w() #initialization of every weights
    #net.train()
    tensorboard = SummaryWriter('runs/' + 'lr=' + str(learn_rate) + ',wd=' + str(weight_decay))
    for epoch in range(epochs):  # loop over the dataset multiple times
        val_accuracy = []
        val_losses = []
        train_losses = []
        train_acc = []
        total = 0
        correct = 0
        running_loss = 0.0
        num_minibatch = 0
        train_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            train_loss += loss.item()
            num_minibatch += 1
            prints = 200
            if i % prints == (prints-1):  # print every 200 mini-batches
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, i + 1, running_loss / prints))
                print('Accuracy: %d %%' % (100*correct/total))
                running_loss = 0.0

        # Save Metrics
        _, validation_accuracy, validation_loss = valNet(net, validloader, device)
        val_losses.append(validation_loss)
        val_accuracy.append(validation_accuracy)
        train_loss /= num_minibatch
        train_losses.append(train_loss)
        train_accuracy = 100 * correct/total
        train_acc.append(train_accuracy)

        # Print tensorboard
        tensorboard.add_scalar('data/train_loss', train_loss, epoch)
        tensorboard.add_scalar('data/train_acc', train_accuracy, epoch)
        tensorboard.add_scalar('data/valid_loss', validation_loss, epoch)
        tensorboard.add_scalar('data/valid_acc', validation_accuracy, epoch)
    tensorboard.close()

    final_val_loss = min(val_losses)
    index = val_losses.index(final_val_loss)
    final_val_acc = val_accuracy[index]
    final_train_loss = train_losses[index]
    final_train_acc = train_acc[index]
    return net, final_val_loss, final_val_acc, final_train_loss, final_train_acc


def valNet(net, validloader, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    num_minibatch = 0
    valid_loss = 0.0
    #net.eval()
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            labels = labels.to(device)
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            num_minibatch += 1

    accuracy = 100 * correct / total
    valid_loss /= num_minibatch
    #print('Accuracy of the network: %d %%' % accuracy+' Validation Loss: %.7f' % valid_loss)
    return net, accuracy, valid_loss
