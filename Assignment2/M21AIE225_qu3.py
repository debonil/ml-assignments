# %% [markdown]
# #### Question 3:
# Download the dataset containing two columns X and Y. Fix an appropriate threshold and consider Y > threshold as 1 otherwise 0. Perform the following tasks.
# #
#     1. Create the labels from the given data.
#     2. Plot the distribution of samples using the histogram.
#     3. Determine the prior probability for both classes.
#     4. Determine the class conditional probabilities (likelihood) for the classes.
#     5. Plot the count of each unique element for each class.
#     6. Calculate the posterior probability of both classes and plot them.
#     7. Consider the dataset of question 1 and select any one feature and class label. Repeat
#     1 to 6.
#

# %% [markdown]
# - Data load

# %%
import numpy as np
import pandas as pd
data = pd.read_csv('data-ques-3/Data.csv')
data.head()


# %% [markdown]
# 1. Create the labels from the given data.

# %%
data['Y'].hist()


# %% [markdown]
# Choosen thresold 700 as there were clear division in histogram at y = 700

# %%
data["label"] = data.Y.apply(lambda x: 0 if x <= 700 else 1)
data.head()


# %% [markdown]
# 2. Plot the distribution of samples using the histogram.

# %%
data[data['label'] == 0]['X'].hist()
data[data['label'] == 1]['X'].hist()
data.hist()


# %% [markdown]
# 3. Determine the prior probability for both classes.

# %%
classes = data['label'].unique()
classCounts = data['label'].value_counts()
print(f'Class count :\n{classCounts.sort_index()}')

priors = classCounts/data['label'].size
for c in classes:
    print(f'Priors for class {c} : {priors[c]}')


# %% [markdown]
# 4. Determine the class conditional probabilities (likelihood) for the classes.

# %%

likelihood = data.groupby(data['X'].values).apply(
    lambda x: x['label'].value_counts()/len(x))

print(likelihood)


# %% [markdown]
# 5. Plot the count of each unique element for each class.

# %%
uniqueCntPerClass = data.groupby(data['label'].values).apply(
    lambda x: x['X'].value_counts())

uniqueCntPerClass[0].hist()
uniqueCntPerClass[1].hist()

print(uniqueCntPerClass)


# %% [markdown]
# 6. Calculate the posterior probability of both classes and plot them.

# %%

uniqeFeatures = data['X'].unique()
uniqeFeatures.sort()
evidenceProb = data['X'].value_counts()/len(data)
posterior = []


for c in classes:
    classPost = []
    for f in uniqeFeatures:
        classPost.append((likelihood[f].get(c, 0)*priors[c])/evidenceProb[f])
    posterior.append(classPost)

print(posterior)


# %%

postDf = pd.DataFrame(posterior, columns=uniqeFeatures).T
print(postDf)
postDf.plot()


# %% [markdown]
# 7. Consider the dataset of question 1 and select any one feature and class label. Repeat
# 1 to 6.

# %%
dataQ1 = pd.read_csv('data-ques-1/iris_dataset.csv')
dataQ1.head()


# %%
dataQ1['variety'].hist()


# %% [markdown]
# - Selected sepal.length as seceted feature

# %%
x = 'sepal.length'
label = 'variety'

# %% [markdown]
# - Plot the distribution of samples using the histogram.

# %%
classes = dataQ1[label].unique()
classCounts = dataQ1[label].value_counts()
print(f'Class count :\n{classCounts.sort_index()}')

for cls in classes:
    dataQ1[dataQ1[label] == cls][x].hist()


# %% [markdown]
# - Determine the prior probability for both classes

# %%
priors = classCounts/dataQ1[label].size
for c in classes:
    print(f'Priors for class {c} : {priors[c]}')

# %% [markdown]
# - Determine the class conditional probabilities (likelihood) for the classes.

# %%
likelihood = dataQ1.groupby(dataQ1[x].values).apply(
    lambda x: x[label].value_counts()/len(x))

print(likelihood)

# %% [markdown]
# -  Plot the count of each unique element for each class.

# %%
uniqueCntPerClass = dataQ1.groupby(dataQ1[label].values).apply(
    lambda a: a[x].value_counts())

for c in classes:
    uniqueCntPerClass[c].hist()

print(uniqueCntPerClass)

# %% [markdown]
# - Calculate the posterior probability of both classes and plot them.

# %%

uniqeFeatures = dataQ1[x].unique()
uniqeFeatures.sort()
evidenceProb = dataQ1[x].value_counts()/len(dataQ1)
posterior = []


for c in classes:
    classPost = []
    for f in uniqeFeatures:
        classPost.append((likelihood[f].get(c, 0)*priors[c])/evidenceProb[f])
    posterior.append(classPost)

print(posterior)

# %%

postDf = pd.DataFrame(posterior, columns=uniqeFeatures).T
print(postDf)
postDf.plot()
