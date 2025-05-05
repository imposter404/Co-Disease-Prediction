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
print(y_pred)
# print(classification_report(y_true, y_pred))
# print(model.predict_proba(X_test))
# print('Yes' if predict==1 else 'No')



