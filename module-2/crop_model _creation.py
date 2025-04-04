#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[6]:


dataset = pd.read_csv(r'./dataset/Preprocessed_Dataset.csv')


# In[7]:


dataset.head()


# In[4]:


type(dataset)


# In[5]:


dataset.shape


# In[6]:


dataset.info()


# In[7]:


dataset.describe().T


# In[8]:


dataset.isnull().sum()


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


labelencoder = LabelEncoder()


# In[13]:


dataset['class'] = labelencoder.fit_transform(dataset['label'])
dataset.head(5)


# # feature selection

# In[14]:


dataset.corr()
sns.heatmap(dataset.corr(), annot = True)
plt.show()


# In[15]:


X = dataset.iloc[:, [1,2,3,4]].values
Y = dataset.iloc[:, [5]].values


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42 )


# In[17]:


# Checking dimensions
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# In[18]:


print(X_train)


# # Support Vector Classifier Algorithm

# In[19]:


from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)


# # Decision tree Algorithm

# In[20]:


from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dectree.fit(X_train, Y_train)


# # Random forest Algorithm

# In[21]:


from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
ranfor.fit(X_train, Y_train)


# # Making predictions on test dataset

# In[22]:


Y_pred_svc = svc.predict(X_test)
Y_pred_dectree = dectree.predict(X_test)
Y_pred_ranfor = ranfor.predict(X_test)


#  # Model Evaluation

# In[23]:


from sklearn.metrics import accuracy_score


# In[24]:


accuracy_svc = accuracy_score(Y_test, Y_pred_svc)
accuracy_dectree = accuracy_score(Y_test, Y_pred_dectree)
accuracy_ranfor = accuracy_score(Y_test, Y_pred_ranfor)


# In[25]:


print("Support Vector Classifier: " + str(accuracy_svc * 100))
print("Decision tree: " + str(accuracy_dectree * 100))
print("Random Forest: " + str(accuracy_ranfor * 100))


# In[26]:


c=accuracy_svc * 100
d=accuracy_dectree * 100
e=accuracy_ranfor * 100


# In[27]:


scores = [c,d,e]
algorithms = ["Support Vector Machine","Decision Tree","Random Forest"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


# In[28]:


sns.set(rc={'figure.figsize':(8,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(algorithms,scores)


# In[29]:


import joblib 
joblib.dump(ranfor, r'Models\ranfor_model1.pkl') 
ranfor_from_joblib = joblib.load(r'Models\ranfor_model1.pkl')  
print("Model successfully created...!")


# # Confusion Matrix

# In[30]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_ranfor)


# In[31]:


plt.figure(figsize = (10,10))
sns.heatmap(pd.DataFrame(cm), annot=True)


# In[32]:



X1 = dataset.iloc[:, [1,2,3,4]].values
Y1 = dataset.iloc[:, [6]].values


# In[33]:


from sklearn.model_selection import train_test_split
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size = 0.20, random_state = 42 )


# In[34]:


from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
ranfor.fit(X_train1, Y_train1)


# In[35]:

import joblib 
joblib.dump(ranfor, r'Models\ranfor_model2.pkl') 
ranfor_from_joblib = joblib.load(r'Models\ranfor_model2.pkl')  
print("Model successfully created...!")

