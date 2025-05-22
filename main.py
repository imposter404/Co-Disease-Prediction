# %%
import pandas as pd 
import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
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


# import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules

# import numpy as np
import warnings
warnings.filterwarnings('ignore')

import random

# %%
df = pd.read_csv("Training.csv")
test=""
symptom=['itching','skin_rash','nodal_skin_eruptions']
symptom=list(set(symptom))
disease=set(df['prognosis'])
disease=list(disease)
disease.sort()
feature_cols =list(df.columns)
feature_cols.pop()

# -------------------------
X = df[feature_cols] 
y = list(df['prognosis'])
y = pd.get_dummies(y,drop_first=False)
y_col=y.columns.to_list()
# -------------------------

# %%
def patient():
    global test
    patient_data = pd.read_csv("a3_binary.csv")
    # patient_data=patient_data.drop("prognosis", axis=1)
    # x=patient_data.iloc[0]
    # print(tuple(patient_data))
    # print(x[0])
    test=patient_data
patient()

# %%
decisionTree_final=[]
RandomForest_final=[]
# Bays_final=[]
svm_final=[]

multi_output_tree=""
multi_output_svm=""
multi_output_Forest=""

# %%
def disease_output(e,f):
    global y_col

    for i in range(len(e)):
        temp=[]
        for j in range(len(e[0])):
            if(e[i][j]==True):
                temp.append(y_col[j])
        f.append(temp)


# %%
def SVM():
    global multi_output_svm
    global X,y
    clf = svm.SVC(kernel='linear')
    multi_output_svm=MultiOutputClassifier(clf)
    multi_output_svm.fit(X, y)
    
def SVM_outpt():
    global test
    global multi_output_svm
    global svm_final
    svm_final=[]
    predict= multi_output_svm.predict(test)
    disease_output(predict,svm_final)

# %%
def decisionTree():
    global multi_output_tree
    global X,y
    dtree = DecisionTreeClassifier() 
    multi_output_tree=MultiOutputClassifier(dtree)
    multi_output_tree.fit(X,y) 

def decisionTree_Output():
    global test
    global multi_output_tree
    global decisionTree_final
    decisionTree_final=[]
    predict = multi_output_tree.predict(test)
    disease_output(predict,decisionTree_final)

# %%
def RandomForest():
    global multi_output_Forest
    global X,y

    # ---------------------------------------------------   
    rf = RandomForestClassifier()
    # multi_output_Forest=rf
    multi_output_Forest=MultiOutputClassifier(rf)
    multi_output_Forest.fit(X,y) 
    # ---------------------------------------------------   
    # rf = RandomForestClassifier(n_estimators=74,max_depth=49)
    # multi_output_Forest=MultiOutputClassifier(rf)
    # multi_output_Forest.fit(X,y) 


    # ---------------------------------------------------   
    # param_dist = {'n_estimators': randint(50,200), 'max_depth': randint(10,100)}
    # rf = RandomForestClassifier()
    # rand_search = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)
    # rand_search.fit(X, y)
    # print(rand_search.best_estimator_)
    # multi_output_Forest=MultiOutputClassifier(rand_search.best_estimator_)
    # multi_output_Forest.fit(X,y)
    # # print(rand_search.best_estimator_)

# RandomForest()

# %%
def RandomForest_Output():
    global test
    global multi_output_Forest
    global RandomForest_final
    RandomForest_final=[]
    predict = multi_output_Forest.predict(test)
    disease_output(predict,RandomForest_final)

# %%
decisionTree()
# decisionTree_Output()
RandomForest()
# RandomForest_Output()
SVM()
# SVM_outpt()

# decisionTree_Output()
# RandomForest_Output()
# SVM_outpt()

# %%
import json
with open('pair.json','r') as openfile:
    pair=json.load(openfile)
patient()
decisionTree_Output()
RandomForest_Output()
SVM_outpt()

# %%
r=0
d=0
s=0
def Final():
    global r,d,s
    r=0
    d=0
    s=0
    for i in range(len(RandomForest_final)):
        # print("decisionTree:",decisionTree_final[i])
        # print("RandomForest:",RandomForest_final[i])
        # print("svm:         ",svm_final[i],i)
        # print("pair:        ",  pair[i])
        # print('\n')

        if (len(decisionTree_final[i])>=len(pair[i])):
            d=d+1
        # else :
        #     print("decisionTree:",decisionTree_final[i])
        #     print("RandomForest:",RandomForest_final[i])
        #     print("svm:         ",svm_final[i],i)
        #     print("pair:        ",  pair[i])
        #     print('d \n')

        if (len(RandomForest_final[i])>=len(pair[i])):
            r=r+1
        # else :
        #     # print("decisionTree:",decisionTree_final[i])
        #     print("RandomForest:",RandomForest_final[i])
        #     # print("svm:         ",svm_final[i],i) 
        #     print("pair:        ",  pair[i])
        #     print('r \n')

        if (len(svm_final[i])>=len(pair[i])):
            s=s+1
        # else :
        #     print("decisionTree:",decisionTree_final[i])
        #     print("RandomForest:",RandomForest_final[i])
        #     print("svm:         ",svm_final[i],i)
        #     print("pair:        ",  pair[i])
        #     print('s \n')
Final()

r=r/len(RandomForest_final)*100
d=d/len(RandomForest_final)*100
s=s/len(RandomForest_final)*100
print(len(RandomForest_final))

print('r=',r,'d=',d,'s=',s)

# %%
def change():
    count=0
    disease_pair=[]
    for i in range(len(RandomForest_final)):
        if ((len(RandomForest_final[i])>=len(pair[i]))):
        # if ((len(decisionTree_final[i])<len(pair[i])) or (len(RandomForest_final[i])>=len(pair[i])) or (len(svm_final[i])<len(pair[i]))):

            count+=1
            disease_pair.append(pair[i])
            # print(count,RandomForest_final[i])

    # final=disease_pair[:100]
    print(count)
    final=json.dumps(disease_pair)
    with open('pair.json','w+') as outfile:
        outfile.write(final)
# change()

# %%
def make_Disease():
    final=[]
    f=[]
    with open('pair.json','r') as openfile:
        pair=json.load(openfile)
    with open('a2.json','r') as openfile:
        symptom=json.load(openfile)

    for i in range(len(pair)):
        temp=[]
        temp1=[]
        for j in range(len(pair[i])):
            l=random.randint(0,len(symptom[pair[i][j]])-1)
            # print(pair[i][j],l)
            temp.append(symptom[pair[i][j]][l])
            temp1+=symptom[pair[i][j]][l]
        final.append(temp)
        f.append(list(temp1))

    #----------- save to json file------------
    final=json.dumps(final)
    with open('a3.json','w+') as outfile:
        outfile.write(final)
    #----------------------------------------

    for i in range(len(f)):
        f[i]=set(f[i])
        f[i]=list(f[i])

    df = pd.read_csv("Training.csv")
    head=df.columns.to_list()
    z=[]
    for i in range(len(f)):
        binary=np.zeros(len(df.head(0).columns)-1,dtype=int)
        for j in range(len(f[i])):
            binary[head.index(f[i][j])]=1
        z.append(binary)

    # ----------- save to csv file------------
    df=pd.DataFrame(z)
    head.pop()
    print(head)
    df=df.set_axis(head,axis=1) 
    df.to_csv('a3_binary.csv',index=False)

# %%
# if(r<95):
#     change()
#     make_Disease()
make_Disease()

# %%
for i in range(len(decisionTree_final)):
    if (((len(decisionTree_final[i])<len(pair[i])) and (len(RandomForest_final[i])>=len(pair[i]))) or ((len(RandomForest_final[i])>=len(pair[i])) and (len(svm_final[i])<len(pair[i])))):
        # print(decisionTree_final[i])
        print(pair[i])

# %%
def Apriori():
    df=pd.read_csv('Apriori copy.csv')
    z=pd.concat([df['Disease1'],df['Disease2']])
    z=list(set(z))
    z.sort()

    p=np.zeros((len(df),len(z)),dtype=int)
    p=pd.DataFrame(p)
    p=p.set_axis(z,axis=1) 


    for i in range(len(df)):
        for x in df.iloc[i]:
            p[x].iloc[i]=1

    frequent_itemsets = apriori(p, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0)
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by=['confidence'],ascending=0))
# Apriori()

# %%
def add(e,f):
    return e,f


