import math
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.axes as ax
ann_vals = [[],[],[]]
cnn_vals = [[],[],[]]
try:
    with open('ann.log', 'r', encoding='utf-8') as f:
        for line in f:
            ann_vals[0].append(float(line.split(",")[1][0:-1]))
except(FileNotFoundError):
    print("No Log file found")

try:
    with open('cnn.log', 'r', encoding='utf-8') as f:
        for line in f:
            cnn_vals[0].append(float(line.split(",")[1][0:-1]))
except(FileNotFoundError):
    print("No Log file found")

try:
    with open('ann.time', 'r', encoding='utf-8') as f:
        for line in f:
            temp_list = line.split(",")
            ann_vals[1].append(float(temp_list[0])/60)
            ann_vals[2].append(float(temp_list[1][0:-1]))
except(FileNotFoundError):
    print("No Time file found")

try:
    with open('cnn.time', 'r', encoding='utf-8') as f:
        for line in f:
            temp_list = line.split(",")
            cnn_vals[1].append(float(temp_list[0])/60)
            cnn_vals[2].append(float(temp_list[1][0:-1]))
except(FileNotFoundError):
    print("No Time file found")

plt.title("Cross-Entropy Loss")
plt.xlabel("# of epochs")
plt.plot(ann_vals[0], label='ANN Loss')
plt.plot(cnn_vals[0], label='CNN Loss')
plt.legend()
plt.show()

plt.title("Training Time - Artificial Neural Network")
plt.xlabel("time in minutes")
plt.boxplot(ann_vals[1],vert=False)
plt.show()
plt.title("Training Time - Convolutional Neural Network")
plt.boxplot(cnn_vals[1],vert=False)
plt.show()

plt.title("Accuracy - Artificial Neural Network")
plt.xlabel("% accuracy")
plt.boxplot(ann_vals[2],vert=False)
plt.show()
plt.title("Accuracy - Convolutional Neural Network")
plt.boxplot(cnn_vals[2],vert=False)
plt.show()

