'''
The code contains noted parts that are intended for analysis by child.
In order to analyze by child, eliminate the comments and define the parts of the adults as comments.
'''

import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # train_test_split function
from sklearn.linear_model import Perceptron

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt  
from sklearn import metrics
import seaborn as sns 

imported_data = open('data to analyse.csv', encoding='utf-8')
csv_data = csv.reader(imported_data)
data = list(csv_data)
imported_data.close()

# will contain the data for analysis
data_for_analysis = []
for item in data:  # For an adult
    if item[3] != 'NA':
        data_for_analysis.append(item)

        # for item in data_lines:  # For a child
#     if item[4] != 'NA' and item[5] != 'NA':
#         data_for_analysis.append(item)

#  Will contain the data as DataFrame in order to extract the relevant date when needed
df = pd.DataFrame(data_for_analysis, columns=['Patient number', 'Age', 'Sex', 'Adult BMI (kg/m2)', 'Child Weight (kg)',
                                              'Child Height (cm)', 'Diagnosis'])

"""
 ****************** By now all the data is set and ready to import to the algorithm ******************
"""


X = df[['Age', 'Sex', 'Adult BMI (kg/m2)']]  # Features for an adult
# X = df[['Age', 'Sex', 'Child Weight (kg)', 'Child Height (cm)']] # Features for a child
y = df['Diagnosis']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)



clf = Perceptron(alpha=0.1, random_state=0 ) #Wx0 in function tol=1e-3

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy: {:.2f}%".format(clf.score(X_test, y_test) * 100 ))
print("Precision: {:.2f}%".format(metrics.precision_score(y_test, y_pred, average="micro") * 100 ))
print("Recall: {:.2f}%".format(metrics.recall_score(y_test, y_pred, average="micro") * 100 ))

rfc_predict = clf.predict(X_test)
report = classification_report(y_test, rfc_predict, zero_division='warn')
print("The report is: ")
print(report)

conf_matrix = metrics.confusion_matrix(y_test, rfc_predict)
print("The confusion matrix is: ")
print(conf_matrix)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()