import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import metrics
from scipy.stats import randint
import random
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
# df = pd.read_csv("diabetes.csv")

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss, accuracy_score
from sklearn.datasets import make_multilabel_classification

X, y = make_multilabel_classification(n_samples=10, n_features=4,n_classes=20,random_state=5,n_labels=2)
print(X)

