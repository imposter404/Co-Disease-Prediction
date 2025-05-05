import pandas as pd 
import numpy as np
import os
import random
import json

os.system('cls')




df = pd.read_csv("Training.csv")
disease=set(df['prognosis'])
disease=list(disease)
disease.sort()
final=[]


# -----------------------------------
def Random_disease():
    global final
    for i in range(100):
        m=random.randint(2,3)
        temp=[]
        for j in range(m):
            temp.append(disease[random.randint(0,len(disease)-1)])
        temp=list(set(temp))
        temp.sort()
        final.append(temp)
    print(final)
    #----------- save to json file------------
    final=json.dumps(final)
    with open('b_pair.json','w+') as outfile:
        outfile.write(final)
    #-----------------------------------
# -----------------------------------

# Random_disease()




def Disease_symptom():
    sy={}
    for x in disease:
        sy[x]=[]
        for i in range(len(df)):
            if(df['prognosis'][i]==x):
                temp=[]
                for y in df.columns:
                    if(df[y][i]==1):
                        temp.append(y)
                temp.sort()
                sy[x].append(temp)

    #----------- save to csv file------------
    sy=json.dumps(sy)
    with open('b1.json','w+') as outfile:
        outfile.write(sy)
    # -----------------------------------
# -----------------------------------

# Disease_symptom()





# remove duplicate values--------------------------
def Disease_symptom_duplicate():
    with open('b1.json','r') as openfile:
        df=json.load(openfile)

    for x in disease:
        df[x]=list(set(tuple(r) for r in df[x]))
    
    #----------- save to csv file------------
    df=json.dumps(df)
    with open('b2.json','w+') as outfile:
        outfile.write(df)
    #--------------------------------------
#-------------------------------------------------------

# Disease_symptom_duplicate()






def make_Disease():
    final=[]
    f=[]
    with open('b_pair.json','r') as openfile:
        pair=json.load(openfile)
    with open('b2.json','r') as openfile:
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
    with open('b3.json','w+') as outfile:
        outfile.write(final)
    #----------------------------------------

    for i in range(len(f)):
        f[i]=set(f[i])
        f[i]=list(f[i])

    df = pd.read_csv("Training.csv")
    head=df.columns.to_list()
    z=[]
    for i in range(len(f)):
        binary=np.zeros(len(df.head(0).columns)-1)
        for j in range(len(f[i])):
            binary[head.index(f[i][j])]=1
        z.append(binary)

    # ----------- save to csv file------------
    df=pd.DataFrame(z)
    df.to_csv('b3_binary.csv',index=False)
    # -----------------------------------


# make_Disease()



# Random_disease()
# Disease_symptom()
# Disease_symptom_duplicate()
# make_Disease()