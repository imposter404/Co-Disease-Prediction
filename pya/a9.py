import pandas as pd 
import numpy as np
import os
import random
import json
# from scipy.stats import randint
os.system('cls')




df = pd.read_csv("Training.csv")
disease=set(df['prognosis'])
disease=list(disease)
disease.sort()
# print(disease)
final=[]



# =======================
def Random_disease():
    for i in range(100):
        m=random.randint(2,3)
        # print(n)
        temp=[]
        for j in range(m):
            temp.append(disease[random.randint(0,len(disease)-1)])
            temp.sort()
        temp=list(set(temp))
        final.append(temp)
    print (final)
# print(b)
# Random_disease()



#----------- save to json file------------
# final=json.dumps(final)
# with open('pair.json','w+') as outfile:
#     outfile.write(final)
# -----------------------------------



# # for i in range(len(df)):
# #     if(df['prognosis'][i]==disease[0]):
# #         count.append(i)




sy={}
def Disease_symptom():
    for x in disease:
        sy[x]=[]
        for i in range(len(df)):
            if(df['prognosis'][i]==x):
                temp=[]
                for y in df.columns:
                    if(df[y][i]==1):
                        temp.append(y)
                sy[x].append(temp)
    # print(sy['AIDS'][0])

# Disease_symptom()







# #----------- save to csv file------------
# sy=json.dumps(sy)
# def save_json():
#     with open('aa1.json','w+') as outfile:
#         outfile.write(sy)
# # -----------------------------------
# # save_json()

# # read Json file --------------------------------

# # def read_json():
# with open('aa1.json','r') as openfile:
#     df=json.load(openfile)

# #-------------------------------------------------
# # read_json()
# # print(df)



# # remove duplicate values--------------------------
# def Duplicate_Remove():
#     for x in disease:
#         df[x]=list(set(tuple(r) for r in df[x]))

# Duplicate_Remove()
# # print(len(df['AIDS']))
# print(df)
# df=json.dumps(df)
# with open('aa2.json','w+') as outfile:
#     outfile.write(df)

# #-------------------------------------------------------

# p=np.unique(np.array(df['AIDS']))


with open('pair.json','r') as openfile:
    pair=json.load(openfile)
with open('aa2.json','r') as openfile:
    symptom=json.load(openfile)

final=[]
f=[]
def make_Disease():
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
    # final=json.dumps(final)
    # with open('aa4.json','w+') as outfile:
    #     outfile.write(final)

    # for x in final:
    #     for i in range

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

    print(z)
    # ----------- save to csv file------------
    df=pd.DataFrame(z)
    df.to_csv('aa5_binary.csv',index=False)
    # -----------------------------------


make_Disease()
# final=final[0][0]+final[0][1]
# final=list(final[0]
# print(f)
# print(len(f))


# final=json.dumps(f)
# with open('aa5.json','w+') as outfile:
#     outfile.write(final)

# print(symptom['AIDS'][5])
# print(final)


























































# # ---------------------------------------------
# for i in range(len(df)):
#     if(df['prognosis'][i]==disease[0]):
#         # sy[x].append(x)
#         # temp=[]
#         for y in df.columns:
#             if(df[y][i]==1):
#                 print(y)
#         # # sy[x].append(temp)
#         # count+=1
#     # print(df['prognosis'][i],x)
#     # print(i)



# # print(sy)
# # print(len(df))
# import json
# sy=json.dumps(sy)
# print(sy)
# -------------------------------------------------



# #----------- save to csv file------------
# # df=pd.DataFrame(sy)
# # df.to_json('aa.json',index=False,lines=False)
# with open('aa.json','w') as outfile:
#     outfile.write(sy)
# # -----------------------------------




# # read Json file --------------------------------
# import json
# with open('aa.json','r') as openfile:
#     df=json.load(openfile)
# #-------------------------------------------------







# # remove duplicate values--------------------------
# i=0
# j=1
# for x in disease:
#     while i<(len(df[x])-1):
#         while j<(len(df[x])):
#             if(df[x][i]==df[x][j]):
#                 # print(df[x][i],df[x][j],i,j,len(df[x]))
#                 del df[x][j]
#             else:
#                 j+=1
#         i+=1
#         j=i+1
#         # print(i,j)
# print(len(df['AIDS']))
# print(df['AIDS'])
# #-------------------------------------------------------
