
import seaborn as sns
from sklearn import metrics
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


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


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])


trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET',
                          download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET',
                        download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()


input_size = 784
hidden_sizes = [128, 64]
output_size = 10

given_model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                            nn.ReLU(),
                            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                            nn.ReLU(),
                            nn.Linear(hidden_sizes[1], output_size),
                            nn.LogSoftmax(dim=1))


def trainAndPredictNeuralNet(criterion=nn.NLLLoss(), lr=0.003, hidden_sizes=hidden_sizes,  model=given_model):
    print(
        f"\n\nTraining with \tLoss Function = {criterion}, \tLearning Rate = {lr}, \tHidden Layer = {hidden_sizes}\n")

    print(model)

    #criterion = nn.NLLLoss()
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)

    logps = model(images)  # log probabilities
    loss = criterion(logps, labels)  # calculate the NLL loss

    #print('Before backward pass: \n', model[0].weight.grad)
    loss.backward()
    #print('After backward pass: \n', model[0].weight.grad)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    time0 = time()
    epochs = 15
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            # This is where the model learns by backpropagating
            loss.backward()

            # And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        """ else:
            print("Epoch {} - Training loss: {}".format(e,
                running_loss/len(trainloader))) """
    print("\nTraining Time (in minutes) =", (time()-time0)/60)

    pred_label_arr = []
    true_label_arr = []
    for images, labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            pred_label_arr.append(pred_label)
            true_label_arr.append(true_label)
    return true_label_arr, pred_label_arr


""" true_label_arr, pred_label_arr = trainAndPredictNeuralNet(
    criterion=nn.NLLLoss(), lr=0.003, hidden_sizes=[128, 64])
confusionMatrixAndAccuracyReport(true_label_arr, pred_label_arr)


true_label_arr, pred_label_arr = trainAndPredictNeuralNet(
    criterion=nn.CrossEntropyLoss(), lr=0.003, hidden_sizes=[128, 64])
confusionMatrixAndAccuracyReport(true_label_arr, pred_label_arr)


true_label_arr, pred_label_arr = trainAndPredictNeuralNet(
    criterion=nn.NLLLoss(), lr=0.99, hidden_sizes=[128, 64])
confusionMatrixAndAccuracyReport(true_label_arr, pred_label_arr)


true_label_arr, pred_label_arr = trainAndPredictNeuralNet(
    criterion=nn.NLLLoss(), lr=0.5, hidden_sizes=[128, 64])
confusionMatrixAndAccuracyReport(true_label_arr, pred_label_arr) """


hidden_sizes = [128]
modelNew = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[0], output_size),
                         nn.LogSoftmax(dim=1))

true_label_arr, pred_label_arr = trainAndPredictNeuralNet(
    criterion=nn.NLLLoss(), lr=0.01,hidden_sizes=hidden_sizes, model=modelNew)
confusionMatrixAndAccuracyReport(true_label_arr, pred_label_arr)

hidden_sizes = [16]
modelNew = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[0], output_size),
                         nn.LogSoftmax(dim=1))

true_label_arr, pred_label_arr = trainAndPredictNeuralNet(
    criterion=nn.NLLLoss(), lr=0.01,hidden_sizes=hidden_sizes, model=modelNew)
confusionMatrixAndAccuracyReport(true_label_arr, pred_label_arr)

hidden_sizes = [256, 128]

modelNew = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[1], output_size),
                         nn.LogSoftmax(dim=1))

true_label_arr, pred_label_arr = trainAndPredictNeuralNet(
    criterion=nn.NLLLoss(), lr=0.01, hidden_sizes=hidden_sizes, model=modelNew)
confusionMatrixAndAccuracyReport(true_label_arr, pred_label_arr)

hidden_sizes = [64, 16, 16, 16]

modelNew = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[2], hidden_sizes[3]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[3], output_size),
                         nn.LogSoftmax(dim=1))

true_label_arr, pred_label_arr = trainAndPredictNeuralNet(
    criterion=nn.NLLLoss(), lr=0.01,hidden_sizes=hidden_sizes, model=modelNew)
confusionMatrixAndAccuracyReport(true_label_arr, pred_label_arr)

hidden_sizes = [256, 128, 64, 32]

modelNew = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[2], hidden_sizes[3]),
                         nn.ReLU(),
                         nn.Linear(hidden_sizes[3], output_size),
                         nn.LogSoftmax(dim=1))

true_label_arr, pred_label_arr = trainAndPredictNeuralNet(
    criterion=nn.NLLLoss(), lr=0.01,hidden_sizes=hidden_sizes, model=modelNew)
confusionMatrixAndAccuracyReport(true_label_arr, pred_label_arr)
