# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:59:56 2018

@author: MOBASSIR
"""

"""

code for this research work was taken from catboost's(A fast, scalable, high performance Gradient Boosting on Decision Trees library, used for ranking, classification, regression and other machine learning tasks for Python, R, Java, C++. Supports computation on CPU and GPU. https://catboost.ai)
doccumentation.

ref link : https://github.com/catboost/catboost


"""

# Importing the librariesfrom 

import pandas as pd
import numpy as np
from catboost import  CatBoostClassifier,Pool,cv
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
 
rnd_state = 63
 


#Importing the dataset
dataset = pd.read_csv('disease.csv')
X =  dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#X_test[104][10] = "yes"


cat_featuresind=[0,1,2]

clf = CatBoostClassifier (iterations=10,random_seed=rnd_state, custom_metric='Accuracy')

clf.fit(X_train, y_train, cat_features=cat_featuresind,plot = True)


clf.score(X_test, y_test)





from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = clf.predict(X_test)

#print(clf.predict(X_test[104]))
cm = confusion_matrix (y_test, y_pred)


from sklearn.metrics import recall_score,precision_score

print(recall_score(y_test,y_pred,average='macro'))

print(precision_score(y_test, y_pred, average='micro'))


print(accuracy_score(y_test,y_pred))





#cr0ss validati0n

cv_params = clf.get_params()
cv_params.update({
    'loss_function': 'Logloss'
})
cv_data = cv(
    Pool(X, y, cat_features=cat_featuresind),
    cv_params,
    plot=True
)


print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
    np.max(cv_data['test-Accuracy-mean']),
    cv_data['test-Accuracy-std'][np.argmax(cv_data['test-Accuracy-mean'])],
    np.argmax(cv_data['test-Accuracy-mean'])
))


print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))


"""

importances = clf.feature_importances_
print(clf.feature_importances_)
plt.title('Feature Importances ')
plt.barh(range(len(cat_featuresind)), importances[cat_featuresind], color='b', align='center')
#plt.yticks(dataset[i][0] for i in cat_featuresind)
plt.xlabel('Relative Importance')
plt.show()
 





plt.bar(range(len(clf.feature_importances_)))
plt.show()

from sklearn import metrics
y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.xlabel("False Positive Rate (FPR)", fontsize=14)
plt.ylabel("True Positive Rate (TPR)", fontsize=14)
plt.title("ROC Curve (appendix)", fontsize=14)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()



from sklearn.metrics import recall_score,precision_score

recall_score(y_test,y_pred,average='macro')

precision_score(y_test, y_pred, average='micro')

plt(clf)

print(accuracy_score(y_test,y_pred))




pool1 = Pool(data=X, label=y, cat_features=cat_featuresind)


importances = clf.get_feature_importance(prettified=True)
print (importances)
"""