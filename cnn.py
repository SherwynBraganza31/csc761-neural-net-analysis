from time import time
import torch
from torch.autograd import Variable
from torch import nn, optim
from torchvision import transforms, datasets

class DatasetsMNIST:
    """
    MNIST Dataset Dataloaders Class

        Attributes
        ----------
        trainset : torch.datasets
            torchvision datasets class representing the training set

        testset : torch.datasets
            torchvision datasets class representing the testing set

        trainloader : torch.utils.data.DataLoader
            torchvision datasets class representing the training set labels

        testloader : torch.utils.data.DataLoader
            torchvision datasets class representing the testing set labels

        batch_size : int
            the size of sample set to be loaded in when testing and training

    """

    def __init__(self, batch_size=64):
        """
        :param batch_size: int
            the size of the sample set to load in at a time
        """
        self.trainset = datasets.MNIST(root='mnist_dataset', download=True, train=True, transform=transforms.ToTensor())
        self.testset = datasets.MNIST(root='mnist_dataset', download=True, train=False, transform=transforms.ToTensor())
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=True)
        pass

    # def transformConfig(self):
    #     self.transform = transforms.Compose([transforms.ToTensor(),
    #                                          transforms.Normalize((0.5,), (0.5,))])
    #     return self.transform


class LogDict:
    """
    Log Dictionary Class
    Creates a list of dictionaries that serves as a log file.
    Checks for existing log files to append to or creates a new one
    if none exit

        Attributes
        ----------
        dictionary: list
            list object of dictionaries acting as epoch logs

        epoch_pad: int
            the last read epoch # from the file

        filename: string
            the name of the old logfile

    """
    def __init__(self, filename: str):
        """
        :param filename: str
            name of existing logfile
        """
        self.dictionary = []
        self.epoch_pad = 0
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    pass
                last_line = line
                self.epoch_pad = int(line.split(",")[0])
                f.close()
        except(FileNotFoundError):
            print("No Log file found")

    def save_log_file(self, filename='cnn.log'):
        """
        Save log to a file
        :param filename: string
            name of the file to save to
        :return: None
        """
        with open(filename, 'a', encoding='utf-8') as f:
            for x in self.dictionary:
                f.write(str(x[0]) + "," + str(x[1]) + "\n")
            f.close()
        pass


class ModelCNN(nn.Module):
    """
    CNN Model Definition
    Inherits from superclass nn.Module.
    4 Layers -> 2 Convolutional

        Attributes
        ----------
        conv1: torch.nn.Sequential
            First Convolutional Layer. 1 input (unchangeable), 16 outputs.

        conv2: torch.nn.Sequential
            Second Convolutional Layer. 16 inputs, 32 outputs. Changeable with changes made to 1st layer

        output: torch.nn.Sequential
            Output Layer. Implements a softmax ish output.

        optimizer: torch.optim
            Optimizer module

        criterion: torch.nn
            Loss method object

        Methods
        -------
        forward(x)
            inherits from parent class. Links all NN modules created

        load_model_from_file(filename)
            loads a torch.state_dict from the saved model specified by
            filename and updates the model params

        config_optimizer(learning_rate)
            configures the optimizer object model

        config_loss_function()
            configures loss function parameters

        save_state(filename)
            saves the torch.state_dict corresponding to the models current
            params
    """
    def __init__(self):
        """
        Initializes attributes of parent class and creates optimizer and loss calculator objects
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 1 input channel
                out_channels=16,  # 16 output channels
                kernel_size=(5, 5),  # sliding window matrix of size (5,5)
                stride=(1, 1),  # Move the window by the stride value
                padding=(2, 2)  # pad the data to make it 30x30 instead of 28x28 (auto calculates if missing)
            ),
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2),  # pooling function with window matrix of size (2,2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=(5, 5),
                      stride=(1, 1),
                      padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Define a fully connected layer to link the 32x49 output from the Conv Layer to the Output Layer
        # performs the same function as a Softmax activator function but is better as it takes a dynamic
        # learning approach rather than a probabilistic one.
        self.out = nn.Linear(32 * 7 * 7, 10)

        self.optimizer = self.config_optimizer()
        self.criterion = self.config_loss_function()
        pass

    def forward(self, x):
        """
        inherited from parent class.
        :param x: nn.Module
            current state of NN Module
        :return: output:
            calculated output of the NN
        :return x: nn.Module
            state of the neural network (returned for visualization)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization

    def load_model_from_file(self, filename:str):
        """
        Loads an a torch.state_dict from file <filename>. If no file was found,
        it initializes the optimizer and loss_function and returns the original
        model.
        :param filename: str
            name of the file to open from
        :return: self: ModelCNN
            the ModelCNN recovered from the file
        """
        try:
            self.load_state_dict(torch.load(filename))
            self.eval()
        except FileNotFoundError:
            print("Could not load old model")
            self.config_optimizer()
            self.config_loss_function()

        return self

    def config_optimizer(self, learning_rate=0.001):
        """
        Configures an optimizer or learner for the model.
        Defaults to Adam Optimizer with learning rate of 0.001
        :param learning_rate: float
            learning rate of the model
        :return: self.optimizer
            the optimizer object created
        """
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return self.optimizer

    def config_loss_function(self):
        """
        Configures a loss function for the model.
        Defaults to CrossEntropyLoss.
        :return: self.criterion
            the loss function object created.
        """
        self.criterion = nn.CrossEntropyLoss()
        return self.criterion

    def save_state(self, filename='cnn.sav'):
        """
        :param filename: str
            The name of the file to save.
        :return: None
        """
        torch.save(self.state_dict(), filename)


cnn = ModelCNN()
cnn = cnn.load_model_from_file('cnn.sav')
dataset = DatasetsMNIST()
log = LogDict(filename='cnn.log')

epochs = 15
images, labels = next(iter(dataset.trainset))
time0 = time()

for e in range(epochs):
    running_loss = 0
    for images, labels in dataset.trainloader:

        # Training pass
        cnn.optimizer.zero_grad()

        output = cnn(Variable(images))[0]
        loss = cnn.criterion(output, Variable(labels))

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        cnn.optimizer.step()

        running_loss += loss.item()
    else:
        mean_loss = running_loss/len(dataset.trainloader)
        print("Epoch {} - Training loss: {}".format(log.epoch_pad + e +1, mean_loss))
        log.dictionary.append((log.epoch_pad+e+1,mean_loss))

timestamp = (time() - time0)
print("\nTraining Time (in minutes) =", timestamp / 60)

images, labels = next(iter(dataset.testloader))
correct_count, all_count = 0, 0
cnn.model.eval()
accuracy = 0

for i in range(len(labels)):
    with torch.no_grad():
        test_output, last_layer = cnn(images)

    pred_label = torch.max(test_output, 1)[1].data.squeeze()
    accuracy = (pred_label == labels).sum().item() / float(labels.size(0))

print("\nNumber Of Images Tested =", 10000)
print("Model Accuracy =", accuracy * 100, "%")

# save the model
cnn.save_state()

# log records
log.save_log_file()

with open('cnn.time', 'a', encoding='utf-8') as f:
    f.write(str(timestamp) + "," + str(accuracy*100) + "\n")
    f.close()