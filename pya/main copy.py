import pandas as pd 
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import os



os.system('cls')






def decisionTree():
    df = pd.read_csv("diabetes.csv")
    features = ['pregnant','glucose','blood_pressure','skin_thickness','insulin','BMI','Diabetes_pedigree','Age']
    X = df[features]
    y = df['label']
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)
    predict=dtree.predict([[0,0,0,5,0,8.6,15,0]])
    print(dtree.predict_proba([[0,0,0,5,0,8.6,15,0]]))
    # print('Yes' if predict==1 else 'No')
decisionTree()

