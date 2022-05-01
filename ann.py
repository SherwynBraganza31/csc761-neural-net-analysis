import torch
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

# set the parameters to transform the dataset for input
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])

"""
MNIST Dataset Dataloaders

Load with batch size of 64 samples at a time. 
"""
mnist_trainset = datasets.MNIST(root='mnist_dataset', download=True, train=True, transform=transform)
mnist_testset = datasets.MNIST(root='mnist_dataset', download=True, train=False, transform=transform)
mnist_trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=True)

"""
Log file Handling
Open a log file if it exits and get the last epoch number to continue from
"""
log_dictionary = []
epoch_pad = 0
try:
    with open('ann.log', 'r', encoding='utf-8') as f:
        for line in f:
            pass
        last_line = line
        epoch_pad = int(line.split(",")[0])
        f.close()
except(FileNotFoundError):
    print("No Log file found")

# Layer HyperParameters
input_size = 784 # input layer
hidden_sizes = [128, 64] # 2 hidden Layers
output_size = 10 # output layer

"""
Model Declaration
Declare a model with 4 fully connected Layers using the above hyperparams

nn.Linear is a method of defining a fully conected layer from param1 to param2,
the following argument has to be the activation function for that layer.

The general methodology is to follow a Linear Layer Definition with a activation 
function and in the output case, and output function
"""
model_ann = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                  nn.ReLU(),
                  nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                  nn.ReLU(),
                  nn.Linear(hidden_sizes[1], output_size),
                  nn.LogSoftmax(dim=1))

# try to load old model with saved weights and parameterrs
try:
    model_ann.load_state_dict(torch.load("ann.sav"))
    model_ann.eval()
except:
    print("Could not load old model")

# choose the loss criterion
criterion = nn.CrossEntropyLoss()
images, labels = next(iter(mnist_trainloader))

# optimizer/learner
optimizer = optim.Adam(model_ann.parameters(), lr=0.001)
time0 = time() # take note of start time for timing of the process
epochs = 15

for e in range(epochs):
    running_loss = 0
    for images, labels in mnist_trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        # get output and calculate loss based on chosen criterion
        output = model_ann(images)
        loss = criterion(output, labels)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        mean_loss = running_loss/len(mnist_trainloader)
        print("Epoch {} - Training loss: {}".format(e+1+epoch_pad, mean_loss))
        log_dictionary.append((e+1+epoch_pad, mean_loss))

timestamp = time() - time0
print("\nTraining Time (in minutes) =", timestamp/ 60)

images, labels = next(iter(mnist_testloader))
correct_count, all_count = 0, 0


"""
Testing phase
"""
for images, labels in mnist_testloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model_ann(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if true_label == pred_label:
            correct_count += 1
        all_count += 1

print("\nNumber Of Images Tested =", all_count)
print("Model Accuracy =", (correct_count / all_count) * 100, "%")

torch.save(model_ann.state_dict(), "ann.sav")
with open('ann.log', 'a', encoding='utf-8') as f:
    for x in log_dictionary:
        f.write(str(x[0]) + "," + str(x[1]) + "\n")
    f.close()
with open('ann.time', 'a', encoding='utf-8') as f:
    f.write(str(timestamp) + "," + str((correct_count / all_count)*100) + "\n")
    f.close()