import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics

import os
os.system('cls')

pima = pd.read_csv("diabetes.csv")
# pima = pd.read_csv("pima-indians-diabetes.csv")
col_names = pima.head(0)
# print(col_names)

# pima.head()
feature_cols = ['pregnant','glucose','blood_pressure','skin_thickness','insulin','BMI','Diabetes_pedigree'] 
X = pima[feature_cols] 
y = pima.label

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=0)

clf = DecisionTreeClassifier() 
clf = clf.fit(X_train,y_train) 
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



'''# Visualizing Decision Trees-----------------'''

# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# from IPython.display import Image
# import pydotplus

# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data, 
#                 filled=True, 
#                 rounded=True, 
#                 special_characters=True,
#                 feature_names = feature_cols,
#                 class_names= [0,1])

# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('diabetes.png') 
# Image(graph.create_png())

# ---------------------------

'''# Optimizing Decision Tree Performance'''

# clf = DecisionTreeClassifier(criterion="entropy", max_depth=3) 
# clf = clf.fit(X_train,y_train) 
# y_pred = clf.predict(X_test) 

# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# from sklearn.metrics import ( 
#     accuracy_score, 
#     confusion_matrix, 
#     ConfusionMatrixDisplay, 
#     f1_score, 
#     classification_report, 
# )

# y_pred = clf.predict(X_test) 
# accuray = accuracy_score(y_pred, y_test) 
# f1 = f1_score(y_pred, y_test, average="weighted") 

# y_true=y_test
# print(classification_report(y_true, y_pred))
# print("Accuracy:", accuray) 
# print("F1 Score:", f1)

print(clf.predict([6,148,72,35,0,33.6,0.627]))
