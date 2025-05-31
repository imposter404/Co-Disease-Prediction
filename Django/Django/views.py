from django.shortcuts import render
import sys,os
# current=os.path.dirname(os.path.abspath(__file__))
# parent=os.path.dirname(current)
# sys.path.append(parent)

import Disease as Co
# import aaa 



def index(request):
    context={
        # 'n': pkg.a
        'n': "val"
    }
    return render(request,'index.html',context)



count=0

def load_co():
    global count
    if(count==0):
        Co.decisionTree()
        Co.RandomForest()
        Co.SVM()
        Co.Apriori()
    print(count)
    count+=1
load_co()



# ----------------------------------------------------


import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt



symptom=[]

# count=0

@csrf_exempt  # Only for demonstration; handle CSRF properly in production
def receive_data(request):
    global symptom
    
    if request.method == 'POST':
        data = json.loads(request.body)
        value = data.get('get')
        if value=='symptom':
            response={
                'response':200,
                'body':{
                    'disease':Co.feature_cols
                }
            }
        elif value=='CoDisease':
            symptom=data.get('body')
            Co.patient(symptom)
            Co.decisionTree_Output()
            Co.RandomForest_Output()
            Co.SVM_outpt()
            Co.Disease_percentage()
            response={
                'response':200,
                'body':{
                    # 'symptom': [Co.decisionTree_final,Co.RandomForest_final,Co.svm_final],
                    'forest':Co.final,
                    'apriori':Co.final_apriori
                }
            }
        return JsonResponse(response)