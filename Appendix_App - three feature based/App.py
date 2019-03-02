# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:59:56 2018

@author: MOBASSIR
"""

from flask import Flask, render_template, request

app = Flask(__name__)

from catboost import CatBoostClassifier,Pool,cv

# Importing the libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

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

clf = CatBoostClassifier (iterations=100,random_seed=rnd_state, custom_metric='Accuracy')

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


"""


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


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/feedback', methods=['POST'])
def feedback():
  

    if request.method == 'POST':
        
        list = []
        comment = request.form['fever']
        data = int(comment)
        list.append(data)
        comment = request.form['painq']
        data = int(comment)
        list.append(data)
        comment = request.form['nausea']
        data = int(comment)
        list.append(data)
        if(list[0] is 0):
            X_test[59][0] = "3 to 5 times"
        elif (list[0] is 1):
            X_test[59][0] = "Less than 2"
        elif (list[0] is 2):
            X_test[59][0] = "More than above"
        else:
            X_test[59][0] = "None"
            
         
        if(list[1] is 0):
            X_test[59][1] = "High"
        elif (list[1] is 1):
            X_test[59][1] = "Medium"
        elif (list[1] is 2):
            X_test[59][1] = "Low"
        else:
            X_test[59][1] = "None"
            
        
        
        if(list[2] is 0):
            X_test[59][2] = "Normal"
        elif (list[2] is 1):
            X_test[59][2] = "Not tolerable"
        elif (list[2] is 2):
            X_test[59][2] = "So much"
        elif (list[2] is 3):
            X_test[59][2] = "Little bit"
        else:
            X_test[59][2] = "None"
            
           
        
        check = clf.predict(X_test)
        check = check[59]
        print('ans = ',check, list)
        #return render_template('index.html')
       
        #comment = request.form['rating']
        #data = [comment]
    return render_template('result.html', prediction=check)

if __name__ == '__main__':
    app.run()