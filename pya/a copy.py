import os
os.system('cls')


import pandas as pd
df = pd.read_csv('Training.csv') 
# df = pd.read_csv('Test2.csv') 
# df = pd.read_csv('loan_data.csv') 
# df.head()
# df.info()
# print (df)

pre_df = pd.get_dummies(df,columns=['prognosis'],drop_first=True) 
# print(pre_df)

# pre_df.info()


import seaborn as sns 
import matplotlib.pyplot as plt

sns.countplot(data=pre_df,x='itching')
plt.xticks(rotation=0, ha='right')
# plt.show()


from sklearn.model_selection import train_test_split 
X = pre_df.drop("prognosis_Fungal infection", axis=1) 
y = pre_df["prognosis_Fungal infection"] 
# X=pre_df
# print(X)
# # X_train, X_test, y_train, y_test = train_test_split( X, y,)

from sklearn.naive_bayes import GaussianNB 
model = GaussianNB() 
model.fit(X, y);


from sklearn.metrics import ( 
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    f1_score, 
    classification_report, 
)



df = pd.read_csv('Test1.csv')
pre_df = pd.get_dummies(df,columns=['prognosis'],drop_first=True)
# print(pre_df)

X = pre_df.drop("prognosis_Fungal infection", axis=1) 
y = pre_df["prognosis_Fungal infection"] 
# print(X)
# print(y)

y_pred = model.predict(X)
accuray = accuracy_score(y_pred, y) 
f1 = f1_score(y_pred, y, average="weighted") 
print("Accuracy:", accuray) 
print("F1 Score:", f1)

labels = ["Fungal infection", "Not Fungal infection"] 
cm = confusion_matrix(y, y_pred) 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels) 
disp.plot();
plt.show()


# output------------------------
# Accuracy: 0.8333333333333334
# F1 Score: 0.8380952380952381
#-------------------------------