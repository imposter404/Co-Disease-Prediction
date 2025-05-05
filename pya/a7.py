import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import metrics
from scipy.stats import randint
# from sklearn.tree import export_graphviz
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
 

# '''
model= RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_true=y_test
print(classification_report(y_true, y_pred))
print(model.predict([[6,148,72,35,0,33.6,0.627,50]]))
# '''





'''
param_dist = {'n_estimators': randint(50,500), 'max_depth': randint(1,20)}
rf = RandomForestClassifier()
rand_search = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)
rand_search.fit(X_train, y_train)
best_rf = rand_search.best_estimator_
print('Best hyperparameters:', rand_search.best_params_)

# model=model.fit(X_train, y_train)
# y_pred=model.predict(X_test)
y_pred=best_rf.predict(X_test)



print(y_pred)
print(y_test)

import matplotlib.pyplot as plt
# y_pred = best_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred) 
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()
'''



'''
           0       0.86      0.86      0.86        51
           1       0.73      0.73      0.73        26

    accuracy                           0.82        77
   macro avg       0.80      0.80      0.80        77
weighted avg       0.82      0.82      0.82        77

'''