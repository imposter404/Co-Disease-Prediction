import pandas as pd 
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
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
test=np.zeros(len(df.head(0).columns)-1)
# symptom=['itching','skin_rash','nodal_skin_eruptions']
# symptom=['itching','skin_rash','nodal_skin_eruptions','dischromic _patches','shivering','chills','stomach_pain']
# symptom=["shivering","skin_rash", "blackheads", "scurring"]
symptom=["weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "blurred_and_distorted_vision", "obesity", "excessive_hunger", "increased_appetite", "polyuria","itching", "skin_rash", "fatigue", "lethargy", "high_fever", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "malaise", "red_spots_over_body","skin_rash", "high_fever", "blister", "red_sore_around_nose"]

symptom=list(set(symptom))
disease=set(df['prognosis'])
disease=list(disease)

head=df.columns.to_list()
for i in range(len(symptom)):
    test[head.index(symptom[i])]=1


# -------------------------
decisionTree_final=[]
RandomForest_final=[]
Bays_final=[]
#--------------------------




# '''
# decisionTree--------------------------------
def decisionTree(e):
    features = df.columns.to_list()
    features.pop()
    X_train=pd.get_dummies(df,columns=['prognosis'],drop_first=False)
    # df=pd.DataFrame(X_train)
    # df.to_csv('csvt.csv',index=False)

    # print(X_train)
    # df=np.column_stack(df,)
    X = df.drop("prognosis", axis=1)
    y = X_train['prognosis_'+e]

    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)
    predict=dtree.predict([test])
    if predict==True :
        decisionTree_final.append(e)
        # print(e)
    # print(e)
    # print(dtree.predict_proba([[0,0,0,5,0,8.6,15,0]]))
    # print('Yes' if predict==1 else 'No')


    
'''
# decisionTree('prognosis_'+'(vertigo) Paroymsal  Positional Vertigo')
for x in disease:
    # print('prognosis_'+x)
    decisionTree(x)
print(decisionTree_final)

# -------------------------------------------------------------------
# '''


# ''' 
# RandomForest------------------------------------------------------
def RandomForest(e):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint
    X_train=pd.get_dummies(df,columns=['prognosis'],drop_first=False)
    X = df.drop("prognosis", axis=1)
    y = X_train['prognosis_'+e]

    # -------------------------
    rf = RandomForestClassifier()
    rf = rf.fit(X, y)
    predict=rf.predict([test])
    if predict==True :
        print(e)
        RandomForest_final.append(e)

        
    # --------------------------
    # # param_dist = {'n_estimators': randint(50,500), 'max_depth': randint(1,20)}
    # param_dist = {'n_estimators': randint(1,10), 'max_depth': randint(1,10)}
    # rf = RandomForestClassifier()
    # rand_search = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)
    # rand_search.fit(X, y)
    # best_rf = rand_search.best_estimator_

    # # print('Best hyperparameters:', rand_search.best_params_)

    # # model=model.fit(X_train, y_train)
    # # y_pred=model.predict(X_test)
    # predict=best_rf.predict([test])
    # if predict==True :
    #     print(e,'multi')
    #     RandomForest_final.append(e)



# for x in disease:
#     RandomForest(x)

# print(RandomForest_final,'RandomForest')

#-----------------------------------------------------------------------
# '''



''' 
# do not execute for random Forest --------------------------------
d = pd.read_csv("training.csv")
for i in range(2): 
    row=d.loc[i].to_list()
    row.pop()
    test=row
    temp=[]
    for x in disease:
        RandomForest(x)
    if len(temp)>1 :
        print(i,temp)
    else :
        print(i)
    # RandomForest_final.append(temp)

#----------------------------------------------------------------------------
'''

def Bays(e):
    from sklearn.model_selection import train_test_split 
    from sklearn.naive_bayes import GaussianNB 
    X_train=pd.get_dummies(df,columns=['prognosis'],drop_first=False)
    X = df.drop("prognosis", axis=1)
    y = X_train['prognosis_'+e]
    # print(y)
    model = GaussianNB()
    model=model.fit(X,y)
    predict=model.predict([test])
    # print(predict)
    if predict==True :
        Bays_final.append(e)


    # accuray = accuracy_score(predict,test)
    # print(accuray)
    # f1 = f1_score(predict, y, average="weighted") 
    # y_true=y
    # print(classification_report(y_true, predict))
    # print("Accuracy:", accuray) 
    # print("F1 Score:", f1)
    # labels = ["Not Fungal infection","Fungal infection"] 
    # cm = confusion_matrix(y, predict)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels) 
    # disp.plot();
    # plt.show()



# decisionTree('Fungal infection')
# RandomForest('Fungal infection')
# Bays('Fungal infection')
# print(decisionTree_final)
# print(RandomForest_final)

# for x in disease:
    # decisionTree(x)
    # RandomForest(x)
    # print(x)
    # Bays(x)

# print(decisionTree_final)
# print(RandomForest_final)
# print(Bays_final)
import json
l=pd.read_csv('b3_binary.csv')
with open('b_pair.json','r') as openfile:
    pair=json.load(openfile)

p=0
n=0

for i in range(len(l)):
    test=l.iloc[i]
    decisionTree_final=[]
    for x in disease:
        decisionTree(x)

    decisionTree_final.sort()
 
    if(pair[i]!=decisionTree_final):
        print(i,len(pair[i]),len(decisionTree_final),pair[i],decisionTree_final)
    
    if(len(pair[i])<=len(decisionTree_final)):
        p+=1
    else:
        n+=1

print('p',p,'n',n)
# output -----------------------------
# p 95 n 5
# -------------------------------



# print(pair[0])
# print(decisionTree_final)
