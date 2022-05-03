# %%
import numpy as np
import torch
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
time1 = time()

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# %%
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.ImageFolder('cifar/train', transform=transform)
valset = datasets.ImageFolder('cifar/test', transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# %%
dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

# %%
class_list = trainset.classes
figure = plt.figure(figsize=(15, 10))
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(
        np.rot90(np.rot90(np.rot90(images[index].numpy().T))), interpolation="nearest")
    plt.title(class_list[labels[index]])
plt.show()
print('Classes : ', class_list)

# %%
input_size = 3072
hidden_sizes = [1024, 512, 128]
output_size = len(class_list)

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(hidden_sizes[2], output_size),
                      nn.LogSoftmax(dim=1))
print(model)

# %%

images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images)  # log probabilities

criterion = nn.NLLLoss()

#tlabls = labelTransform(labels,10)
loss = criterion(logps, labels)  # calculate the loss


# %%
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8)
time0 = time()
epochs = 50


def train(model, epochs):
    loss_list, acc_list = [], []
    for epoch in range(epochs):
        t_loss, v_loss, t_n_correct, v_n_correct = 0, 0, 0, 0
        for i, (images, labels) in enumerate(trainloader):

            model.train()

            # Forward pass

            images, labels = images.to(device), labels.to(device)
            images = images.view(images.shape[0], -1)
            x = model(images)
            train_loss = criterion(x, labels)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            t_loss += train_loss.item()
            # acc
            _, predicted = torch.max(x, 1)
            t_n_correct += (predicted == labels).sum().item()
        model.eval()
        with torch.no_grad():
            for images, labels in valloader:

                images, labels = images.to(device), labels.to(device)
                images = images.view(images.shape[0], -1)
                x = model(images)
                valid_loss = criterion(x, labels)
                v_loss += valid_loss.item()

                # acc
                _, predicted = torch.max(x, 1)
                v_n_correct += (predicted == labels).sum().item()

            print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {t_loss/len(trainset):.4f}, Validation Loss: {v_loss/len(valset):.4f}, Training Accuracy: {t_n_correct/len(trainset):.4f}, Validation Accuracy: {v_n_correct/len(valset):.4f}')

        loss_list.append([t_loss/len(trainset), v_loss/len(valset)])
        acc_list.append([t_n_correct/len(trainset), v_n_correct/len(valset)])
    return np.array(loss_list), np.array(acc_list)


loss_list, acc_list = train(model, epochs)
print("\nTraining Time (in minutes) =", (time()-time0)/60)

# %%


def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(np.rot90(np.rot90(np.rot90(img.numpy().T))))
    ax1.axis('off')
    ax2.barh(np.arange(len(class_list)), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(class_list)))
    ax2.set_yticklabels(class_list)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


# %%
images, labels = next(iter(valloader))

img = images[0].view(1, input_size)
with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
print("Predicted Class =", ps)
print("Predicted Class =", class_list[np.argmax(ps)])
print("Actual Class =", class_list[labels[0]])
view_classify(img.view(3, 32, 32), ps)

# %%
plt.title("Training vs Validation loss")
plt.plot(loss_list[:, 0], linestyle='--', label="Training Loss")
plt.plot(loss_list[:, 1], linestyle='-', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.title("Training vs Validation Accuracy")
plt.plot(acc_list[:, 0], linestyle='--', label="Training Accuracy")
plt.plot(acc_list[:, 1], linestyle='-', label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %%
sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})


def confusionMatrixAndAccuracyReport(Y_test, Y_pred):
    cm = metrics.confusion_matrix(Y_test, Y_pred)
    overallAccuracy = np.trace(cm)/sum(cm.flatten())

    classwiseAccuracy = cm.diagonal()/cm.sum(axis=1)

    plt.figure(figsize=(10, 10))
    plt.title('Accuracy Score: {0:3.3f}'.format(overallAccuracy), size=14)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    sns.heatmap(data=cm, annot=True, square=True,  cmap='Blues', fmt='g')

    plt.show()
    print('Overall Accuracy Score: {0:3.3f}'.format(overallAccuracy))
    print('Classwise Accuracy Score: {0}'.format(classwiseAccuracy))


# %%
model.eval()
predicted_list = np.array([])
label_list = np.array([])
with torch.no_grad():

    for images, labels in valloader:

        images, labels = images.to(device), labels.to(device)
        prob = model(images.view(images.shape[0], -1))
        # max returns (value ,index)
        _, predicted = torch.max(prob, 1)

        predicted_list = np.append(predicted_list, predicted.numpy())
        label_list = np.append(label_list, labels.numpy())


# %%

confusionMatrixAndAccuracyReport(predicted_list, label_list)
