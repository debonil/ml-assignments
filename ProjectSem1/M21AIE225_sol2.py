# %%
import numpy as np
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn  # provide modules and classes for implementing neural networks
import torch.optim as optim  # provide optimization algorithms
# for visual trnasformation: normalzation, converting into tensors, augmentation
import torchvision.transforms as transforms
from torchvision import datasets, transforms


# %% [markdown]
# ### Creating dataset and dataloader

# %%
transform = transforms.Compose([transforms.ToTensor()
                                ])

# %%
# Creating dataloader

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

# %% [markdown]
# ### Creating Dataset using "ImageFolder"

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
print('Classes : ', class_list)

# %%
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# %% [markdown]
# ## 2: CNN

# %%


class CNN(nn.Module):

    def __init__(self, input_channel, num_classes):
        super(CNN, self).__init__()

        self.conv_model = nn.Sequential(

            nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3,
                      padding_mode='replicate', padding=1),  # img_size = (inp_img - F + 2P)/S + 1
            # out_img = (32-3+2*1)/1 + 1 = 32 (32*32)
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # out_img = (32 - 2)/2 + 1 = 16 (16 * 16)


            # img_size = (inp_img - F + 2P)/S + 1
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      padding_mode='replicate', padding=1),
            # out_img = (16-3+2*1)/1 + 1 = 32 (32*32)
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # out_img = (62 - 2)/2 + 1 = 31 (31 * 31)


            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                      padding_mode='replicate', padding=1),  # img_size = (inp_img - F + 2P)/S + 1
            # out_img = (64-3)/1 + 1 = 62 (62*62)
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # out_img = (62 - 2)/2 + 1 = 31 (31 * 31)

        )

        self.classification_model = nn.Sequential(
            nn.Dropout(0.4),
            # in_features = 64*6*6 = 2304
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_model(x)
        # Flattening
        x = x.view(x.size(0), -1)
        x = self.classification_model(x)
        return x

# %%
# Hyperparameters


in_channels = 3  # for CNN

num_classes = 6

lr = 0.0002

num_epochs = 20

# %%
model = CNN(in_channels, num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# %%
print(model)

# %%
# train CNN network


def train(model, epochs):
    loss_list, acc_list = [], []
    for epoch in range(epochs):
        t_loss, v_loss, t_n_correct, v_n_correct = 0, 0, 0, 0
        for i, (images, labels) in enumerate(trainloader):

            model.train()

            # Forward pass

            images, labels = images.to(device), labels.to(device)
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


loss_list, acc_list = train(model, num_epochs)

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
# CNN model evaluation
model.eval()
predicted_list = np.array([])
label_list = np.array([])
with torch.no_grad():

    for images, labels in valloader:

        images, labels = images.to(device), labels.to(device)
        prob = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(prob, 1)

        predicted_list = np.append(predicted_list, predicted.numpy())
        label_list = np.append(label_list, labels.numpy())


# %%

confusionMatrixAndAccuracyReport(predicted_list, label_list)
