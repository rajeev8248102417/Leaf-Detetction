#Importing LIbraries
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

#Uploading dataset
dataset = load_iris()

#Summarize the data
print(dataset.data)  #print the data
print(dataset.data.shape) #shape of the data
print(dataset.target)  #target of the data

#Segreting dataset into X and Y
X = pd.DataFrame(dataset.data,columns=dataset.feature_names)

Y = dataset.target

#Splitting dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X ,Y, test_size=0.25,random_state= 0)
print(x_train.shape)
print(x_test.shape)

#Checking Max_depth
Accuracy = []
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

for i in range(1,10):
  model = DecisionTreeClassifier(max_depth=i,random_state=0)
  model.fit(x_train,y_train)
  pred = model.predict(x_test)
  score = accuracy_score(y_test,pred)
  Accuracy.append(score)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), accuracy, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
plt.title('Finding best Max_Depth')
plt.xlabel('pred')
plt.ylabel('score')

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy',max_depth = 3, random_state = 0)
model.fit(x_train,y_train)

#predcition of model
y_pred = model.predict(x_test)

#Accuracy of the mnodel
from sklearn.metrics import accuracy_score
print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred)*100))
