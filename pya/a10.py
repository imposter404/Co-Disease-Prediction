import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import warnings

import os
os.system('cls')
warnings.filterwarnings('ignore')

df = pd.read_csv("Training.csv")
# pima = pd.read_csv("pima-indians-diabetes.csv")
col_names = df.head(0)
# print(col_names)

# pima.head()
feature_cols =list(df.columns)
feature_cols.pop()


X = df[feature_cols] 
y=df['prognosis']
y=pd.get_dummies(y,drop_first=False)

print(y)

# y = pima[['Go','Go1']]

# for i in range(len(y)):
#     if(y[i]=="NO"):
#         y[i]=0
#     else :
#         y[i]=1

# print(y)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=0)

dtree = DecisionTreeClassifier() 
multi_output_tree=MultiOutputClassifier(dtree)
multi_output_tree.fit(X,y) 
y_pred = multi_output_tree.predict([[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
y_pred=y_pred[0]

# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

y=list(y.head(0))
print(y[0])

# for i in range(len(y_pred)):
#     if(y_pred[i]==True):
#         print(y[i])



