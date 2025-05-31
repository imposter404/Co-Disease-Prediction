# Co-Disease Prediction 
## Using The Classification Ensemble Technique

# Description
The disease orediction project is a machine learning application designed to predict the likelyhood of a various disease based on patient symptoms and medical history.
the project leverage data analysis and predictive modeling to assist healthcare professionals in making imformed decisisons

## Written In Python

<div align="left">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" height="50px" alt="Python" />
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/django/django-plain-wordmark.svg""height="50px" alt="Django" />  
 

</div>



---
# Setup
## Python Dependency 
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pypi/pypi-original.svg" height="50px" alt="pypi" />


> ``` console
> pip install pandas
> pip install numpy
> pip install sklearn
> pip install mlxtend
> ```






---
# Approach
Co Disease Predicted using two ways 
1. Symptom Analysis 
2. Histiorical Data Analysis

## 1. Symptom Analysis 
Here patients with symptoms is Analysed for one or more disease.

By simple disease prediction we predict a single disease
here we are using multilevel to predict more than one disease.

Suppose a patient has disease ``A`` and also disease `B` which is predicted using simple disease prediction

but if we union the two disease `A` & `B` symptoms we get a underlaying disease `C`. 

Where `C` is a subset of `A` union `B`

```math
A \cup B \implies \subset C
```


## 2. Histiorical Data Analysis
We Gather histiorical data of the disease pair that ocurred together and we check the frequency of that pair of disease to predict the likelyhood of a disease pair to occure

### Assosiation Rule Mining
- **Support** : Support of item x is nothing but the ratio of the number of transactions in which item x appears to the total number of transactions

- **Confidence** : Confidence (x => y) signifies the likelihood of the item y being purchased when item x is purchased. This method takes into account the popularity of item x. 

- **Lift** : Lift (x => y) is nothing but the ‘interestingness’ or the likelihood of the item y being purchased when item x is sold. Unlike confidence (x => y), this method takes into account the popularity of the item y.


---

## Algorithm Used
- Decision Tree
- Random Forest 
- Support Vector Machine
- Apriori 



---
# Code
## import library 
> ``` python
> import pandas as pd 
> import numpy as np
> from sklearn import tree
> from sklearn import svm
> from sklearn.tree import DecisionTreeClassifier
> from sklearn.multioutput import MultiOutputClassifier
> from sklearn.ensemble import RandomForestClassifier
> from sklearn.model_selection import RandomizedSearchCV
> from mlxtend.frequent_patterns import apriori, association_rules
> import os
> import warnings
> import json
> ```


## Train Models

>``` python
> decisionTree()
> RandomForest()
> SVM()
> Apriori()
>```

## Test Model

Co Disease Of All Patient
> ```python
> patient('all')
> decisionTree_Output()
> RandomForest_Output()
> SVM_outpt()
> Accuracy()
> ```
## Accuracy
> ```python
> Accuracy()
> ```

```
Accuracy:  r= 96.78111587982833 d= 99.57081545064378 s= 95.92274678111588
```

Co Disease Of Single Patient
> ``` python
> patient(4)
> decisionTree_Output()
> RandomForest_Output()
> SVM_outpt()
> Disease_percentage()
> ```



## Predected Co Disease
```
RandomForest_final [['Common Cold', 'Dengue', 'Tuberculosis']]
decisionTree_final [['Common Cold', 'Dengue', 'Malaria', 'Tuberculosis', 'hepatitis A']]
svm_final          [['Common Cold', 'Dengue', 'Tuberculosis']]
---------------------------------------
        Disease           %
0   Common Cold  100.000000
1        Dengue  100.000000
2  Tuberculosis  100.000000
3   hepatitis A   16.666667
4       Malaria   16.666667
---------------------------------------
                                          Disease     %
0                   (Hypertension Cardiovascular)  15.0
1            (Raised Total Cholesterol Endocrine)  14.0
2                               (Dermatitis Skin)  10.0
3                  (Enthesopathy Musculoskeletal)  10.0
4               (Osteoarthristis Musculoskeletal)  10.0
5                          (Raised LDL Endocrine)   9.0
6                (Bacterial Infection Infections)   8.0
7                        (Depression Psychiatric)   8.0
8  (Infection Lower Respiratory Tract Infections)   8.0
9                             (Obesity Endocrine)   8.0
---------------------------------------
```



---
# Notes

The `requirements.txt` file should list all Python libraries that your code
depend on, and they will be installed using:

```
pip install -r requirements.txt
```


---
# Data
Training data

## Disease - Symptom data
- Kaggle 
    - [data](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)

## Disease - Disease data 
- data have been collected from various journals 
    - catalog.data.gov
    - journals.lww.com
    - journals.sagepub.com
    - multimorbidity.caliberresearch.org
    - pmc.ncbi.nlm.nih.gov
    - abs.gov.au
    - cdc.gov
    - frontiersin.org
    - nature.com
    - sciencedirect.com
    - scielo.br
    - thelancet.com
    - who.int
    - ccwdata.org
    -  ... 
    - ...

[Refferences](data/ReadMe.md)



--- 
## Run

Python
``` python 
Main.py
```
Jupyter Notebook
``` python 
Main.ipynb
```