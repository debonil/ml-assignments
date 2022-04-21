
import seaborn as sns
from sklearn import metrics
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])


trainset = datasets.ImageFolder('train', transform=transform)
valset = datasets.ImageFolder('val', transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)


figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')


input_size = 1024
hidden_sizes = [512, 64]
output_size = 10


def Gurmukhi(input_size=1024, hidden_sizes=[512, 64], output_size=10):

    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1))

    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)

    logps = model(images)  # log probabilities

    criterion = nn.NLLLoss()

    #tlabls = labelTransform(labels,10)
    loss = criterion(logps, labels)  # calculate the loss

    loss.backward()

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = 15
    for e in range(epochs):
        running_loss = 0
        cnt = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            cnt += images.shape[0]
            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            # This is where the model learns by backpropagating
            loss.backward()

            # And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()

    correct_count, all_count = 0, 0
    pred_label_arr = []
    true_label_arr = []
    for images, labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 1024)
            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1
            pred_label_arr.append(pred_label)
            true_label_arr.append(true_label)
    #print("Number Of Images Tested =", all_count)
    accuracy = (correct_count/all_count)
    print(f"\n Hidden Layer = {hidden_sizes} Model Accuracy = {accuracy}")
    return accuracy


sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})


def confusionMatrixAndAccuracyReport(Y_test, Y_pred):
    cm = metrics.confusion_matrix(Y_test, Y_pred)
    overallAccuracy = np.trace(cm)/sum(cm.flatten())

    classwiseAccuracy = np.zeros(len(cm))
    for n in range(len(cm)):
        for i in range(len(cm)):
            for j in range(len(cm)):
                if (i != n and j != n) or (i == n and j == n):
                    classwiseAccuracy[n] += cm[i][j]

    classwiseAccuracy /= sum(cm.flatten())

    plt.figure(figsize=(6, 6))
    plt.title('Accuracy Score: {0:3.3f}'.format(overallAccuracy), size=12)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    sns.heatmap(data=cm, annot=True, square=True,  cmap='Blues')

    plt.show()
    print('Overall Accuracy Score: {0:3.3f}'.format(overallAccuracy))
    print('Classwise Accuracy Score: {0}'.format(classwiseAccuracy))


accs = []
accsLbl = []
for i in range(10):
    for j in range(10):
        accs.append(Gurmukhi(hidden_sizes=[2**i, 2**j]))
        accsLbl.append([2**i, 2**j])

ij = np.argmax(np.array(accs))

print(f'\n\nBest model :: Hidden Layer = {accsLbl[ij]} Model Accuracy = {accs[ij]}')
