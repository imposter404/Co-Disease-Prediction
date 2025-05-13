'''# SVM --------------------'''

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import datasets 
import warnings
from sklearn.multioutput import MultiOutputClassifier
warnings.filterwarnings('ignore')


# ------------------------------------------------------------
df=pd.read_csv('Training.csv')
symptom=["weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "blurred_and_distorted_vision", "obesity", "excessive_hunger", "increased_appetite", "polyuria","itching", "skin_rash", "fatigue", "lethargy", "high_fever", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "malaise", "red_spots_over_body","skin_rash", "high_fever", "blister", "red_sore_around_nose"]
Features=df.columns.to_list()
Features.pop()
Labels=df["prognosis"]
# ------------------------------------------------------------


# ------------------------------------------------------------

# ------------------------------------------------------------









# ------------------------------------------------------------

def make_symptom():
    global symptom
    global test

    test=np.zeros(len(df.head(0).columns)-1)
    for i in range(len(symptom)):
        test[Features.index(symptom[i])]=1
make_symptom()
# ------------------------------------------------------------




# ------------------------------------------------------------

X=df[Features]
y=Labels
y = pd.get_dummies(y,drop_first=False)

clf = svm.SVC(kernel='linear')
multi_output_tree=MultiOutputClassifier(clf)
multi_output_tree.fit(X, y)
y_pred = multi_output_tree.predict([test])
# ------------------------------------------------------------



# ------------------------------------------------------------

y=y.columns.to_list()
print(y_pred[0])
for i in range(len(y_pred[0])):
    if y_pred[0][i]==True :
        print(y[i])
# ------------------------------------------------------------
