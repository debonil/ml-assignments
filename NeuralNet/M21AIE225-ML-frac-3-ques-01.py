# %%
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})

# %%


def plotDb(w, label='Decision Boundary'):
    xx = np.linspace(-1, 1, 100)
    yy = -(w[0]*xx+w[2])/w[1]
    xx = xx[yy >= -1]
    yy = yy[yy >= -1]
    xx = xx[yy <= 1]
    yy = yy[yy <= 1]
    plt.plot(xx, yy, label=label)
    plt.scatter(np.array(data[:, 0]), np.array(data[:, 1]), c=y, marker='o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

# %%


data = np.array([[1, 1], [-1, -1], [0, 0.5],
                [0.1, 0.5], [0.2, 0.2], [0.9, 0.5]])
# data = (data - np.mean(data,axis=0)) # will make data mean = 0
print(data)
y = [1, -1, -1, -1, 1, 1]
print(y)

# %%
plt.scatter(np.array(data[:, 0]), np.array(data[:, 1]), c=y, marker='*')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

# %%
correctClassified = 0
# w = np.random.rand(2,1) # Randomly initialize w
# w = np.append(w,1) # append w0=1
w = np.ones(3)
print(w)

# %%
xx = np.linspace(-1, 1, 100)
yy = -(w[0]*xx+w[2])/w[1]
plt.plot(xx, yy, '-r', label='Initial Decision Boundary')
plt.scatter(np.array(data[:, 0]), np.array(data[:, 1]), c=y, marker='*')
plt.xlabel('redness')
plt.ylabel('weight')
plt.legend()
plt.show()

# %%
count = 1
while (correctClassified != len(data)):  # Until everything is classified
    print(f'Iteration is : {count}')
    for sample in range(len(data)):
        x = np.append(data[sample, 0:2], 1)
        print(f'Sample is : {x}')
        print(f'Class is : {y[sample]}')
        wx = np.dot(np.transpose(w), x)
        print(f'Dot Product of W & X is : {wx}')
        if y[sample] == 1:  # Sample is positive
            if wx >= 0:  # WX >= 0
                correctClassified = correctClassified+1
                print("Positive Sample is correctly clasified")
            else:
                print("Positive Sample is classified negative")
                w = w+x
                print(f'Updated W is : {w}')
                plotDb(w)
        else:  # Sample is Negative
            if wx < 0:  # WX < 0
                correctClassified = correctClassified+1
                print("Negative Sample is Correctly classified")
            else:
                print("Negative Sample is classified positive")
                w = w-x
                print(f'Updated W is : {w}')
                plotDb(w)
    count = count+1
    if(correctClassified != len(data)):
        correctClassified = 0

print(w)

# %%
print((np.dot(data, w[:-1].T)))
print(np.array(y))
