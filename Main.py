import pandas as pd 
import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import os
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')
os.system('cls')

#-----------------------------------------
df = pd.read_csv("data/Training.csv")
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
# -----------------------------------------

decisionTree_final=[]
RandomForest_final=[]
svm_final=[]

multi_output_tree=""
multi_output_svm=""
multi_output_Forest=""

import json
with open('data/test_data/test_pair.json','r') as openfile:
    pair=json.load(openfile)




def patient(e):
    global test
    patient_data = pd.read_csv("data/test_pair_binary.csv")
    test=patient_data
    if e!='all':
        test=[patient_data.iloc[e]]



def disease_output(e,f):
    global y_col
    
    for i in range(len(e)):
        temp=[]
        for j in range(len(e[0])):
            if(e[i][j]==True):
                temp.append(y_col[j])
        f.append(temp)



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



def RandomForest_Output():
    global test
    global multi_output_Forest
    global RandomForest_final
    RandomForest_final=[]
    predict = multi_output_Forest.predict(test)
    disease_output(predict,RandomForest_final)



def Accuracy():
    r=0
    d=0
    s=0
    for i in range(len(RandomForest_final)):
        if (len(decisionTree_final[i])>=len(pair[i])):
            d=d+1
        if (len(RandomForest_final[i])>=len(pair[i])):
            r=r+1
        if (len(svm_final[i])>=len(pair[i])):
            s=s+1
    r=r/len(RandomForest_final)*100
    d=d/len(RandomForest_final)*100
    s=s/len(RandomForest_final)*100
    print('Test Pair length : ',len(RandomForest_final))
    print('r=',r,'d=',d,'s=',s)
    print('---------------------------------------')



def Apriori():
    global rules
    df=pd.read_csv('data/disease_disease.csv')
    z=pd.concat([df['Disease1'],df['Disease2']])
    z=list(set(z))
    z.sort()
    p=np.zeros((len(df),len(z)),dtype=int)
    p=pd.DataFrame(p)
    p=p.set_axis(z,axis=1) 
    for i in range(len(df)):
        for x in df.iloc[i]:
            p[x].iloc[i]=1
    frequent_itemsets = apriori(p, min_support=0.001, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0)
    # print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by=['confidence'],ascending=0))



def predict_apriori(intersection):
    global rules    
    apriori_final=[]
    for x in intersection:
        for i in range(len(rules)):
            if(x in rules.iloc[i]['antecedents']):
                apriori_final.append(rules.iloc[i])
    apriori_final=pd.DataFrame(apriori_final)
    if len(apriori_final)==0:
        print('No Disease Found in History')
        return []
    else:
        # print(apriori_final[['antecedents', 'consequents', 'support', 'confidence', 'lift','jaccard']].sort_values(by=['confidence'],ascending=0))
        return apriori_final[['antecedents', 'consequents', 'support', 'confidence', 'lift','jaccard']].sort_values(by=['confidence'],ascending=0)




def Disease_percentage():
    global intersection
    print('RandomForest_final',RandomForest_final)
    print('decisionTree_final',decisionTree_final)
    print('svm_final         ',svm_final)
    print('---------------------------------------')
    union=[]
    union=RandomForest_final[0].copy()
    union.extend(decisionTree_final[0])
    union.extend(svm_final[0])
    intersection=list(set(RandomForest_final[0]).intersection(set(decisionTree_final[0]) , set(svm_final[0])))

    u_min_n=[x for x in union if x not in intersection]
    u_min_n_set=list(set(u_min_n))
    final=[[x,100] for x in intersection]
    for x in u_min_n_set:
        final.append([x,u_min_n.count(x)/(3*len(u_min_n_set))*100])
    final=pd.DataFrame(final)
    final.columns=['Disease','%']
    print(final.sort_values(by='%',ascending=0))
    print('---------------------------------------')
    apriori_final=predict_apriori(intersection)
    if len(apriori_final)!=0: #return value
        final_apriori=[]
        for i in range(len(apriori_final)):
            final_apriori.append([apriori_final.iloc[i]['consequents'], apriori_final.iloc[i]['confidence']*100])
        final_apriori=pd.DataFrame(final_apriori)
        final_apriori.columns=['Disease','%']
        print(final_apriori)
    print('---------------------------------------')





# Train Models------------------------------
decisionTree()
RandomForest()
SVM()
# Disease from History----------------------
Apriori()
#-------------------------------------------



# Test Models-------------------------------
patient('all')
decisionTree_Output()
RandomForest_Output()
SVM_outpt()
Accuracy()
#-------------------------------------------


# Disease Of Single Patient-----------------
patient(4)
decisionTree_Output()
RandomForest_Output()
SVM_outpt()
Disease_percentage()
# ------------------------------------------