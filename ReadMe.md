# Co-Disease Prediction 
## Using The Classification Ensemble Technique

# Description
The disease orediction project is a machine learning application designed to predict the likelyhood of a various disease based on patient symptoms and medical history.
the project leverage data analysis and predictive modeling to assist healthcare professionals in making imformed decisisons

## Written In Python

<div align="left">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" height="50px" alt="Python" />  

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

### Apriori Algorithm
- **Support** : Support of item x is nothing but the ratio of the number of transactions in which item x appears to the total number of transactions

- **Confidence** : Confidence (x => y) signifies the likelihood of the item y being purchased when item x is purchased. This method takes into account the popularity of item x. 

- **Lift** : Lift (x => y) is nothing but the ‘interestingness’ or the likelihood of the item y being purchased when item x is sold. Unlike confidence (x => y), this method takes into account the popularity of the item y.




## Algorithm Used
- Decision Tree
- Random Forest 
- Support Vector Machine
- Apriori 



---
# Code

> ``` python
> import pandas as pd 
> import numpy as np
> from sklearn import tree
> from sklearn import svm
> from sklearn.tree import DecisionTreeClassifier
> from sklearn.multioutput import MultiOutputClassifier
> from sklearn.ensemble import RandomForestClassifier
> from sklearn.model_selection import RandomizedSearchCV
> ```


## Train Models
> ``` python
> decisionTree()
> RandomForest()
> SVM()
> ```

## Test Model

> ``` python
> patient()
> decisionTree_Output()
> RandomForest_Output()
> SVM_outpt()
> ```

## Accuracy
> ``` python
> Final()
> ```

```
Accuracy= r=87.71 d=92.13 s=90.15
```

---
# Data
Training data
- Kaggle 
    - [data]()

Disease Pair data for `Apriori`
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

[Refferences](Data.md)



--- 
## Run
``` python 
Main.ipynb
```