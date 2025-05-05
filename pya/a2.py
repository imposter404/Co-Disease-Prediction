import os
os.system('cls')


'''-------------multilevel--------------'''
#------------- Training ------------
import pandas as pd
df = pd.read_csv('Training.csv') 
# df = pd.read_csv('loan_data.csv') 
# df.head()
# df.info()
disease=df['prognosis']
# pre_df = pd.get_dummies(df,columns=['prognosis'],drop_first=True) 
# pre_df.info()
# --------------------------------
# print(pre_df)



# --------------- graph plot-----------------------
import seaborn as sns 
import matplotlib.pyplot as plt
# sns.countplot(data=pre_df,x='chills')
# plt.xticks(rotation=0, ha='right')
# plt.show()
# --------------------------------------------------


from sklearn.model_selection import train_test_split 
X = df
y = disease 
from sklearn.naive_bayes import GaussianNB 
model = GaussianNB()

X_train,X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=5)
X_train=pd.get_dummies(X_train,columns=['prognosis'],drop_first=True)
X_test=pd.get_dummies(X_test,columns=['prognosis'],drop_first=True)
model.fit(X_train, y_train);
# print(y_train)


#------------- Testing -------------
# df = pd.read_csv('Testing.csv')
# pre_df = pd.get_dummies(df,columns=['prognosis'],drop_first=True)
# X_test= pre_df.drop("prognosis_Fungal infection", axis=1) 
# y_test= pre_df["prognosis_Fungal infection"] 

# X_train,X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=5)
# --------------------------------------



#----------- save to csv file------------
# df=pd.DataFrame(X_test)
# df.to_csv('csv.csv',index=False)
# -----------------------------------



### --------------- accuracy ----------- 

from sklearn.metrics import ( 
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    f1_score, 
    classification_report, 
)

y_pred = model.predict(X_test) 
accuray = accuracy_score(y_pred, y_test) 
f1 = f1_score(y_pred, y_test, average="weighted") 

y_true=y_test
print(classification_report(y_true, y_pred))
print("Accuracy:", accuray) 
print("F1 Score:", f1)




# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm) 
# disp.plot();
# plt.show()

''' # Output -----------------------'''
'''
                                         precision    recall  f1-score   support

(vertigo) Paroymsal  Positional Vertigo       1.00      1.00      1.00        56
                                   AIDS       1.00      1.00      1.00        60
                                   Acne       1.00      1.00      1.00        60
                    Alcoholic hepatitis       1.00      1.00      1.00        58
                                Allergy       1.00      1.00      1.00        59
                              Arthritis       1.00      1.00      1.00        67
                       Bronchial Asthma       1.00      1.00      1.00        59
                   Cervical spondylosis       1.00      1.00      1.00        49
                            Chicken pox       1.00      1.00      1.00        63
                    Chronic cholestasis       1.00      1.00      1.00        58
                            Common Cold       1.00      1.00      1.00        56
                                 Dengue       1.00      1.00      1.00        63
                               Diabetes       1.00      1.00      1.00        61
           Dimorphic hemmorhoids(piles)       1.00      1.00      1.00        58
                          Drug Reaction       1.00      1.00      1.00        60
                       Fungal infection       1.00      1.00      1.00        63
                                   GERD       1.00      1.00      1.00        57
                        Gastroenteritis       1.00      1.00      1.00        64
                           Heart attack       1.00      1.00      1.00        67
                            Hepatitis B       1.00      1.00      1.00        56
                            Hepatitis C       1.00      1.00      1.00        53
                            Hepatitis D       1.00      1.00      1.00        65
                            Hepatitis E       1.00      1.00      1.00        61
                           Hypertension       1.00      1.00      1.00        61
                        Hyperthyroidism       1.00      1.00      1.00        62
                           Hypoglycemia       1.00      1.00      1.00        55
                         Hypothyroidism       1.00      1.00      1.00        58
                               Impetigo       1.00      1.00      1.00        64
                               Jaundice       1.00      1.00      1.00        66
                                Malaria       1.00      1.00      1.00        62
                               Migraine       1.00      1.00      1.00        59
                        Osteoarthristis       1.00      1.00      1.00        65
           Paralysis (brain hemorrhage)       1.00      1.00      1.00        57
                    Peptic ulcer diseae       1.00      1.00      1.00        65
                              Pneumonia       1.00      1.00      1.00        52
                              Psoriasis       1.00      1.00      1.00        59
                           Tuberculosis       1.00      1.00      1.00        67
                                Typhoid       1.00      1.00      1.00        63
                Urinary tract infection       1.00      1.00      1.00        56
                         Varicose veins       1.00      1.00      1.00        57
                            hepatitis A       1.00      1.00      1.00        59

                               accuracy                           1.00      2460
                              macro avg       1.00      1.00      1.00      2460
                           weighted avg       1.00      1.00      1.00      2460

Accuracy: 1.0
F1 Score: 1.0
'''