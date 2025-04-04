import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('./dataset/Preprocessed_Dataset.csv')

dataset.head()

type(dataset)

dataset.shape

dataset.info()

dataset.describe().T

dataset.isnull().sum()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

dataset['class'] = labelencoder.fit_transform(dataset['label'])
dataset.head(5)

dataset.corr()
sns.heatmap(dataset.corr(), annot = True)
plt.show()

X = dataset.iloc[:, [1,2,3,4]].values
Y = dataset.iloc[:, [5]].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42 )

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

print(X_train)


from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,Y_train)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)

Y_pred_svc = svc.predict(X_test)
Y_pred_nb = nb.predict(X_test)
Y_pred_knn = knn.predict(X_test)

accuracy_svc = accuracy_score(Y_test, Y_pred_svc)
accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)

print("Support Vector Classifier: " + str(accuracy_svc * 100))
print("Naive bayes: " + str(accuracy_nb * 100))
print("KNN: " + str(accuracy_knn * 100))

c=accuracy_svc * 100
d=accuracy_nb * 100
e=accuracy_knn * 100

scores = [c,d,e]
algorithms = ["Support Vector Machine","Naive bayes","KNN"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")

sns.set(rc={'figure.figsize':(8,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(algorithms,scores)

import joblib 
joblib.dump(nb, r'Models\nb_model1.pkl') 
nb_from_joblib = joblib.load(r'Models\nb_model1.pkl')  


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_nb)

plt.figure(figsize = (10,10))
sns.heatmap(pd.DataFrame(cm), annot=True)

X1 = dataset.iloc[:, [1,2,3,4]].values
Y1 = dataset.iloc[:, [6]].values

from sklearn.model_selection import train_test_split
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size = 0.20, random_state = 42 )

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train1,Y_train1)

import joblib 
joblib.dump(knn, r'Models\nb_model2.pkl') 
ranfor_from_joblib = joblib.load(r'Models\nb_model2.pkl')  
print("Models successfully created...!")
print(" \n \n ")
print(" \n \n ")
print("###########################################################################################")

print("##                Crop Yield Prediction Based on Indian agriculture                      ##")
print("###########################################################################################")

Location=str  (input("          Location                                 |             "))
print("--------------------------------------------------------------------------------------------")
date=str      (input("           Date                                    |             "))
print("--------------------------------------------------------------------------------------------")
Temp=float    (input("      Temperature level                            |             "))
print("--------------------------------------------------------------------------------------------")
Humidity=float(input("       Humidity value                              |             "))
print("--------------------------------------------------------------------------------------------")
pH=float      (input("          pH level                                 |             "))
print("--------------------------------------------------------------------------------------------")
Rainfall=float(input("       Rainfall level                              |             "))
print("--------------------------------------------------------------------------------------------")

test_data=[Temp,Humidity,pH,Rainfall]

ranfor_from_joblib = joblib.load(r'Models\nb_model1.pkl')  

prediction_crop=ranfor_from_joblib.predict([test_data])
x=str(prediction_crop)
print(" \n \n ")
print("###########################################################################################")
print("                               Predicted crop:  "+ x                   ,"                        ")
#print(prediction_crop)

ranfor_from_joblib = joblib.load(r'Models\nb_model2.pkl')  

prediction_price=ranfor_from_joblib.predict([test_data])
y=str(prediction_price)
print("###########################################################################################")
print("                               prediction price:  "+ y                   ,"                        ")
print("###########################################################################################")
#print(prediction_price)


