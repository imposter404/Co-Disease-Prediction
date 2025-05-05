import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import ( 
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    f1_score, 
    classification_report, 
    precision_score,
    recall_score,
)
import os



os.system('cls')


df = pd.read_csv("diabetes.csv")
X = df.drop('label',axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1,random_state=0)
 
model= RandomForestClassifier()
model=model.fit(X_train, y_train)
# y_pred=model.predict(X_test)
y_pred=model.predict([[0,0,0,5,0,8.6,15,0]])
# y_true=1
# print(y_pred)
# print(classification_report(y_true, y_pred))
# print(model.predict_proba(X_test))
# print('Yes' if predict==1 else 'No')

from sklearn.model_selection import RandomizedSearchCV,train_test_split
from scipy.stats import randint

param_dist = {'n_estimators': randint(50,500), 'max_depth': randint(1,20)} 
# Create a random forest classifier 
rf = RandomForestClassifier() 
# Use random search to find the best hyperparameters 
rand_search = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5) 
# Fit the random search object to the data 
rand_search.fit(X_train, y_train)
y_pred = model.predict_proba(X_test) 
# accuracy = accuracy_score(y_test, y_pred) 
# precision = precision_score(y_test, y_pred) 
# recall = recall_score(y_test, y_pred) 
print('prob',y_pred)

# print("Accuracy:", accuracy) 
# print("Precision:", precision) 
# print("Recall:", recall)