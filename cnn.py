from time import time
import json
# PyTorch imports
import torch
from torch.autograd import Variable
from torch import nn, optim
# import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from torchvision import transforms, datasets

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])

"""
MNIST Dataset Dataloaders
"""
mnist_trainset = datasets.MNIST(root='mnist_dataset', download=True, train=True, transform=transforms.ToTensor())
mnist_testset = datasets.MNIST(root='mnist_dataset', download=True, train=False, transform=transforms.ToTensor())
mnist_trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=True)

"""
Log file Handling
Open a log file if it exits and get the last epoch number to continue from
"""
log_dictionary = []
epoch_pad = 0
try:
    with open('cnn.log', 'r', encoding='utf-8') as f:
        for line in f:
            pass
        last_line = line
        epoch_pad = int(line.split(",")[0])
        f.close()
except(FileNotFoundError):
    print("No Log file found")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(5,5),
                stride=(1,1),
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (5,5), (1,1), 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization

model_cnn = CNN()

try:
    model_cnn.load_state_dict(torch.load("cnn.sav"))
    model_cnn.eval()
except:
    print("Could not load old model")

criterion = nn.CrossEntropyLoss()
images, labels = next(iter(mnist_trainloader))

optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)
time0 = time()
epochs = 15

for e in range(epochs):
    running_loss = 0
    for images, labels in mnist_trainloader:
        # Don't flatten MNIST images into a 784 long vector
        # images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model_cnn(Variable(images))[0]
        loss = criterion(output, Variable(labels))

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        mean_loss = running_loss/len(mnist_trainloader)
        print("Epoch {} - Training loss: {}".format(epoch_pad + e +1, mean_loss))
        log_dictionary.append((epoch_pad+e+1,mean_loss))

timestamp = (time() - time0)
print("\nTraining Time (in minutes) =", timestamp / 60)

images, labels = next(iter(mnist_testloader))
correct_count, all_count = 0, 0
model_cnn.eval()
accuracy = 0

for i in range(len(labels)):
    with torch.no_grad():
        test_output, last_layer = model_cnn(images)

    pred_label = torch.max(test_output, 1)[1].data.squeeze()
    accuracy = (pred_label == labels).sum().item() / float(labels.size(0))

print("\nNumber Of Images Tested =", 10000)
print("Model Accuracy =", accuracy * 100, "%")

torch.save(model_cnn.state_dict(), "cnn.sav")
with open('cnn.log', 'a', encoding='utf-8') as f:
    for x in log_dictionary:
        f.write(str(x[0]) + "," + str(x[1]) + "\n")
    f.close()
with open('cnn.time', 'a', encoding='utf-8') as f:
    f.write(str(timestamp) + "," + str(accuracy*100) + "\n")
    f.close()