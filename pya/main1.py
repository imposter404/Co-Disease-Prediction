import pandas as pd 
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
import os
import array
import warnings
from sklearn.metrics import ( 
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    f1_score, 
    classification_report, 
)
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
os.system('cls')


# training------------------------
df = pd.read_csv("Training.csv")
test=""
symptom=['itching','skin_rash','nodal_skin_eruptions']
# symptom=['itching','skin_rash','nodal_skin_eruptions','dischromic _patches','shivering','chills','stomach_pain']
# symptom=["shivering","skin_rash", "blackheads", "scurring"]
# symptom=["continuous_sneezing", "shivering", "watering_from_eyes"]
# symptom=["weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "blurred_and_distorted_vision", "obesity", "excessive_hunger", "increased_appetite", "polyuria","itching", "skin_rash", "fatigue", "lethargy", "high_fever", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "malaise", "red_spots_over_body","skin_rash", "high_fever", "blister", "red_sore_around_nose"]

symptom=list(set(symptom))

disease=set(df['prognosis'])
disease=list(disease)
disease.sort()

feature_cols =list(df.columns)
feature_cols.pop()

# head=df.columns.to_list()




# -------------------------
def make_symptom():
    global symptom
    global test

    test=np.zeros(len(df.head(0).columns)-1)
    for i in range(len(symptom)):
        test[feature_cols.index(symptom[i])]=1
# -------------------------
# make_symptom()





# -------------------------
# decisionTree_final=[]
# RandomForest_final=[]
# Bays_final=[]
#--------------------------




# '''
multi_output_tree=""
# decisionTree--------------------------------
def decisionTree():
    global multi_output_tree
    global feature_cols
    global test
    X = df[feature_cols] 
    y = list(df['prognosis'])
    y = pd.get_dummies(y,drop_first=False)

    dtree = DecisionTreeClassifier() 
    multi_output_tree=MultiOutputClassifier(dtree)
    multi_output_tree.fit(X,y) 
    # print(classification_report(y, y_predict))
    # predict = multi_output_tree.predict([test])
    # predict=predict[0]

    # y=list(y.head(0))
    # for i in range(len(predict)):
    #     if(predict[i]==True):
    #         decisionTree_final.append(y[i])
    # print(decisionTree_final)
# -------------------------------------------------


def decisionTree_Output():
    global test
    global multi_output_tree

    decisionTree_final=[]
    predict = multi_output_tree.predict(test)
    # predict=predict[0]
    print(predict)

    '''### work from here -=-=-=-=-'''
    ## i am workinf on classifiaation report
    # we need to make Y_true = y_pred









    # ==================================
    # y=disease
    # for i in range(len(predict)):
    #     if(predict[i]==True):
    #         decisionTree_final.append(y[i])
    # print("DecisionTree :",decisionTree_final)
    # ============================================

# -------------------------------------------------


# decisionTree() 
# decisionTree_Output()



# ''' 
multi_output_Forest=""
# RandomForest------------------------------------------------------
def RandomForest():
    global multi_output_Forest
    global feature_cols

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint
    global multi_output_tree

    feature_cols =list(df.columns)
    feature_cols.pop()
    X = df[feature_cols] 
    y = list(df['prognosis'])
    y = pd.get_dummies(y,drop_first=False)

    # -------------------------
    rf = RandomForestClassifier()
    multi_output_Forest=MultiOutputClassifier(rf)
    multi_output_Forest.fit(X,y) 
#-----------------------------------------------------------------------
# '''



def RandomForest_Output():
    global test
    global multi_output_Forest

    RandomForest_final=[]
    predict = multi_output_Forest.predict([test])
    predict=predict[0]

    y=disease
    for i in range(len(predict)):
        if(predict[i]==True):
            RandomForest_final.append(y[i])
    print("RandomForest :",RandomForest_final)

# -------------------------------------------------


# RandomForest()
# RandomForest_Output()









multi_output_bays=""
def Bays():
    from sklearn.model_selection import train_test_split 
    from sklearn.naive_bayes import GaussianNB 

    global multi_output_bays
    global feature_cols

    feature_cols =list(df.columns)
    feature_cols.pop()
    X = df[feature_cols] 
    y = list(df['prognosis'])
    y = pd.get_dummies(y,drop_first=False)

    model = GaussianNB()
    multi_output_bays=MultiOutputClassifier(model)
    multi_output_bays.fit(X,y)

# # -------------------------

def Bays_Output():
    global test
    global multi_output_bays

    Bays_final=[]
    predict = multi_output_bays.predict([test])
    predict=predict[0]

    y=disease
    for i in range(len(predict)):
        if(predict[i]==True):
            Bays_final.append(y[i])
    print("Bays         :",Bays_final)
# -------------------------------------------------


# Bays()
# Bays_Output()



#-----------------------------------------  
def patient():
    global test
    patient_data = pd.read_csv("Testing.csv")
    patient_data=patient_data.drop("prognosis", axis=1)
    # x=patient_data.iloc[0]
    # print(tuple(patient_data))
    # print(x[0])

    test=patient_data
    # print(patient_data)
    # for i in range(len(patient_data)):
    #     test=patient_data.iloc[i]
    #     test_model()
    #     print("\n")
#-----------------------------------------

# patient()




#-------------------
def train_model():
    decisionTree() 
    # RandomForest()
    # Bays()
#-------------------

#-------------------
def test_model():
    decisionTree_Output()    
    # RandomForest_Output()
    # Bays_Output()
#-------------------  

# make_symptom()

train_model()
## test_model()
patient()

# make_symptom()
test_model()





# patient_data = pd.read_csv("Testing.csv")
# z=patient_data['prognosis']
# z = pd.get_dummies(z,drop_first=False)

# print(z)
# patient_data=patient_data.drop("prognosis", axis=1)
# test=patient_data.iloc[0]





# y_predict=multi_output_tree.predict([test])
# print(y_predict)

# print(classification_report(y_test, y_pred))








# # Output--------------------------
# model can give 100% accure output to known samples (training.csv)
# but is the symptom is for more than one disease then bays give no output
#-----------------------------

