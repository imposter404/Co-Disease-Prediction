import pandas as pd 
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import os
import warnings
warnings.filterwarnings('ignore')
os.system('cls')


features = ['pregnant','glucose','blood_pressure','skin_thickness','insulin','BMI','Diabetes_pedigree','Age']

def decisionTree():
    df = pd.read_csv("diabetes.csv")
    features = ['pregnant','glucose','blood_pressure','skin_thickness','insulin','BMI','Diabetes_pedigree','Age']
    X = df[features]
    y = df['label']
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)
    predict=dtree.predict([[0,0,0,5,0,8.6,15,0]])
    print('Yes' if predict==1 else 'No')

decisionTree()

# a=np.zeros(len(features))
# print(a[0])

'''# output------------------------'''
'''
[0]
'''