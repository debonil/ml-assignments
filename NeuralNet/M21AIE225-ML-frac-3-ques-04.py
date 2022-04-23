# %%
# Ref : https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02

import seaborn as sns
from sklearn import metrics
import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from sklearn import svm

# %%
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# %%

trainset = datasets.ImageFolder('train', transform=transform)
valset = datasets.ImageFolder('val', transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1000, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=1000, shuffle=True)

# %%
dataiter = iter(trainloader)
X_train, y_train = dataiter.next()


X_test, y_test = iter(valloader).next()
X_test = X_test.view(X_test.shape[0], -1)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

# %%
figure = plt.figure()
num_of_images = 24
for index in range(1, num_of_images + 1):
    plt.subplot(4, 6, index)
    plt.axis('off')
    plt.imshow(X_train[index].numpy().T.squeeze())

# %%
X_train = X_train.view(X_train.shape[0], -1)

linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo',
                 probability=True).fit(X_train, y_train)
rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo',
              probability=True).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1,
               decision_function_shape='ovo', probability=True).fit(X_train, y_train)
sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo',
              probability=True).fit(X_train, y_train)

# %%
# retrieve the accuracy and print it for all 4 kernel functions
accuracy_lin = linear.score(X_test, y_test)
accuracy_poly = poly.score(X_test, y_test)
accuracy_rbf = rbf.score(X_test, y_test)
accuracy_sig = sig.score(X_test, y_test)
print("Accuracy Linear Kernel:", accuracy_lin)
print("Accuracy Polynomial Kernel:", accuracy_poly)
print("Accuracy Radial Basis Kernel:", accuracy_rbf)
print("Accuracy Sigmoid Kernel:", accuracy_sig)

# %%


def view_classify(img, ps, title):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 32, 32).numpy().squeeze())
    ax1.axis('off')
    ax1.set_title(title)
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


# %%
# create the title that will be shown on the plot
titles = ['Linear kernel', 'RBF kernel', 'Polynomial kernel', 'Sigmoid kernel']

# %%
images, labels = next(iter(valloader))
img = images[0].view(1, 3072)
for i, clf in enumerate((linear, rbf, poly, sig)):
    print("Model =", titles[i])
    with torch.no_grad():
        probab = clf.predict_proba(img)

    print("Predicted Digit =", np.argmax(probab))
    view_classify(img.view(3, 32, 32), torch.tensor(probab), titles[i])
    print("  -  -  -  -  -")

# %%
sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})


def confusionMatrixAndAccuracyReport(Y_test, Y_pred, label):
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
    plt.title(label+' Accuracy : {0:3.3f}'.format(overallAccuracy), size=12)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    sns.heatmap(data=cm, annot=True, square=True,  cmap='Blues')

    plt.show()
    print('Overall Accuracy Score: {0:3.3f}'.format(overallAccuracy))
    print('Classwise Accuracy Score: {0}'.format(classwiseAccuracy))


# %%
for i, clf in enumerate((linear, rbf, poly, sig)):
    confusionMatrixAndAccuracyReport(y_test, clf.predict(X_test), titles[i])
    print('_____________________________________________________')
