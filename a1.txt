import os
os.system('cls')



#------------- Training ------------
import pandas as pd
df = pd.read_csv('Training.csv') 
# df = pd.read_csv('loan_data.csv') 
# df.head()
# df.info()
pre_df = pd.get_dummies(df,columns=['prognosis'],drop_first=True) 
# pre_df.info()
# --------------------------------




# --------------- graph plot-----------------------
import seaborn as sns 
import matplotlib.pyplot as plt
# sns.countplot(data=pre_df,x='chills')
# plt.xticks(rotation=0, ha='right')
# plt.show()
# --------------------------------------------------



from sklearn.model_selection import train_test_split 
X = pre_df.drop("prognosis_Fungal infection", axis=1) 
y = pre_df["prognosis_Fungal infection"] 
from sklearn.naive_bayes import GaussianNB 
model = GaussianNB() 
model.fit(X, y);




#------------- Testing -------------
df = pd.read_csv('Testing.csv')
# X_train,X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=5)

pre_df = pd.get_dummies(df,columns=['prognosis'],drop_first=True)
X_test= pre_df.drop("prognosis_Fungal infection", axis=1) 
y_test= pre_df["prognosis_Fungal infection"] 
# --------------------------------------



#----------- save to csv file------------
# df=pd.DataFrame(X_test)
# df.to_csv('csv.csv',index=False)
# -----------------------------------



#--------------- accuracy ----------- 
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
y_true=pre_df["prognosis_Fungal infection"]
print(classification_report(y_true, y_pred))
print("Accuracy:", accuray) 
print("F1 Score:", f1)

labels = ["Not Fungal infection","Fungal infection"] 
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels) 
disp.plot();
plt.show()

