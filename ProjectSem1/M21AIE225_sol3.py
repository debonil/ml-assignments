# %%
import seaborn as sns
from sklearn import metrics
import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
time1 = time()

# %%
transform = transforms.Compose([transforms.ToTensor()])

# %%

trainset = datasets.ImageFolder('cifar/train', transform=transform)
valset = datasets.ImageFolder('cifar/test', transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=30000, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=6000, shuffle=True)

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

class_list = trainset.classes
figure = plt.figure(figsize=(15, 10))
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(
        np.rot90(np.rot90(np.rot90(X_train[index].numpy().T))), interpolation="nearest")
    plt.title(class_list[y_train[index]])
print('Classes : ', class_list)

# %%
X_train = X_train.view(X_train.shape[0], -1)

linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo',
                 probability=True).fit(X_train, y_train)

poly = svm.SVC(kernel='poly', degree=3, C=1,
               decision_function_shape='ovo', probability=True).fit(X_train, y_train)

logistic = LogisticRegression().fit(X_train, y_train)

bayes = GaussianNB().fit(X_train, y_train)

# %%


def view_classify(img, ps, title):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(
        np.rot90(np.rot90(np.rot90(images[index].numpy().T))), interpolation="nearest")
    ax1.axis('off')
    ax1.set_title(title)
    ax2.barh(np.arange(len(class_list)), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(class_list)))
    ax2.set_yticklabels(class_list)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


# %%
titles = ['Linear kernel', 'Polynomial kernel', 'Logistic Regression', 'Bayes']

# %%
images, labels = next(iter(valloader))
img = images[0].view(1, 3072)
for i, clf in enumerate((linear, poly, logistic, bayes)):
    print("Model =", titles[i])
    with torch.no_grad():
        probab = clf.predict_proba(img)

    print("Predicted Class =", class_list[np.argmax(probab)])
    view_classify(img.view(3, 32, 32), torch.tensor(probab), titles[i])
    print("  -  -  -  -  -")

images, labels = next(iter(valloader))


# %%
sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})


def confusionMatrixAndAccuracyReport(Y_test, Y_pred, label):
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


for i, clf in enumerate((linear, poly, logistic, bayes)):
    confusionMatrixAndAccuracyReport(y_test, clf.predict(X_test), titles[i])
    print('_____________________________________________________')
print("\nAll done (in minutes) =", (time()-time1)/60)
