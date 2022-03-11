#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import seaborn as sns
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# In[2]:


admission = pd.read_csv('./OBH Admission Report.csv',encoding='cp1252')
zip_codes = pd.read_csv('./zip_codes_ohio_state.csv',encoding='cp1252')
admission['zip_code'] = 43210
admission.sample(5)
zip_codes.sample(5)
health = admission.merge(zip_codes, how='left')
health.shape


# In[3]:


print('Rows with any missing values: {} out of {}'.format(len(health)-len(health.dropna()), len(health)))


# In[4]:


health_df = pd.get_dummies(health)
print(health_df.sample(5))


# In[5]:


Y = health_df['Primary drug of choice_Heroin']
X = health_df.drop('Primary drug of choice_Heroin', axis=1)


# In[6]:


drugs = pd.read_csv('./drugs.csv',encoding='cp1252')
drugs_df = drugs[['Heroin', 'Primary drug of choice', 'Secondary drug of choice', 'Tertiary drug of choice']]
print(drugs_df)


# In[7]:


all_drugs = set(drugs_df['Primary drug of choice'].unique()) | set(drugs_df['Secondary drug of choice'].unique()) | set(drugs_df['Tertiary drug of choice'].unique()) 
print(all_drugs)
#for i in range(drugs_df.shape[0]):


# In[8]:


a1 = drugs_df.groupby(['Heroin', 'Primary drug of choice']).size().reset_index(name ='Number of individuals')
print(a1)


# In[9]:


a2 = drugs_df.groupby(['Heroin', 'Secondary drug of choice']).size().reset_index(name ='Number of individuals')
print(a2)


# In[10]:


a3 = drugs_df.groupby(['Heroin', 'Tertiary drug of choice']).size().reset_index(name ='Number of individuals')
print(a3)


# In[11]:


sns.set_style("whitegrid")
ax = sns.barplot(x="Primary drug of choice", y="Number of individuals", hue = "Heroin", data=a1)
sns.set(rc={'figure.figsize':(50,50)})

#modify individual font size of elements
plt.legend(loc=2, prop={'size': 30}, title='Heroin', title_fontsize=30)
plt.xlabel('Primary drug of choice', fontsize=30, labelpad=20);
plt.ylabel('Number of individuals', fontsize=30, labelpad=20);
plt.tick_params(axis='both', which='major', labelsize=20, pad=15)


# In[12]:


X = X.replace(np.nan, 0)

X_train = X.head(int(len(X.index)*0.9))
X_test = X.tail(int(len(X.index)*0.1))

Y_train = Y.head(int(len(Y.index)*0.9))
Y_test = Y.tail(int(len(Y.index)*0.1))


# In[13]:


print(health_df.info())


# In[14]:


sns.factorplot('Client Age', 'zip_code', data=health_df, size=4, aspect=3)
sns.barplot(x='zip_code',y='Client Age',data=health_df,order=[1,0])


# In[15]:


sns.countplot(x='Gender_Male', data=health_df)
sns.countplot(x='Gender_Female', data=health_df)


# In[16]:


print(Y.mean())


# In[17]:


kf = KFold(n_splits = 3)
finalscore = 0
trainscore = 0
for train, test in kf.split(X):
    X_train = X.iloc[train]
    X_test = X.iloc[test]
    Y_train = Y.iloc[train]
    Y_test = Y.iloc[test]
    logreg = LogisticRegression()
    logreg.fit(X_train,Y_train)
    preds_test = logreg.predict_proba(X_test)[:, 1]
    preds_train = logreg.predict_proba(X_train)[:, 1]
    
    finalscore += roc_auc_score(Y_test, preds_test)
    trainscore += roc_auc_score(Y_train, preds_train)
    
print(finalscore/3, trainscore/3)


# In[18]:


conf_matrix = confusion_matrix(np.array(Y_test), np.round(preds_test))

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[19]:


kf = KFold(n_splits = 3)
finalscore = 0
trainscore = 0

for train, test in kf.split(X):
    X_train = X.iloc[train]
    X_test = X.iloc[test]
    Y_train = Y.iloc[train]
    Y_test = Y.iloc[test]
    random_forest = RandomForestClassifier(n_estimators=10)
    random_forest.fit(X_train, Y_train)
    preds_test = random_forest.predict_proba(X_test)[:, 1]
    preds_train = random_forest.predict_proba(X_train)[:, 1]
    
    finalscore += roc_auc_score(Y_test, preds_test)
    trainscore += roc_auc_score(Y_train, preds_train)
    
print(finalscore/3, trainscore/3)


# In[20]:


conf_matrix = confusion_matrix(np.array(Y_test), np.round(preds_test))

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[21]:


kf = KFold(n_splits = 3)
finalscore = 0
trainscore = 0

for train, test in kf.split(X):
    X_train = X.iloc[train]
    X_test = X.iloc[test]
    Y_train = Y.iloc[train]
    Y_test = Y.iloc[test]
    gradient_boosting = GradientBoostingClassifier(n_estimators=10)
    gradient_boosting.fit(X_train, Y_train)
    preds_test = gradient_boosting.predict_proba(X_test)[:, 1]
    preds_train = gradient_boosting.predict_proba(X_train)[:, 1]
    
    finalscore += roc_auc_score(Y_test, preds_test)
    trainscore += roc_auc_score(Y_train, preds_train)
    
print(print(finalscore/3, trainscore/3))


# In[22]:


conf_matrix = confusion_matrix(np.array(Y_test), np.round(preds_test))

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[23]:


coeff_df = DataFrame(health_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
coeff_df

