'''
The code contains noted parts that are intended for analysis by child.
In order to analyze by child, eliminate the comments and define the parts of the adults as comments.
'''

import csv

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # train_test_split function
from sklearn import svm  # SVM Model
from sklearn import metrics  # module for accuracy calculation
import matplotlib.pyplot as plt  # Visualizing the important features
import seaborn as sns  # Bar plot for the important features Visualization
from sklearn.metrics import classification_report


# Import the data from csv file into python
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
 SVM algorithm code in this project is based on the guidance of the tutorial in
 https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
"""

X = df[['Age', 'Sex', 'Adult BMI (kg/m2)']]  # Features for an adult
# X = df[['Age', 'Sex', 'Child Weight (kg)', 'Child Height (cm)']] # Features for a child
y = df['Diagnosis']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) # 70% training and 30% test
# random_state=109????????????????????????????
"""
 ******************************************************************
"""


#Create a svm Classifier
# kernel options: {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy: {:.2f}%".format(clf.score(X_test, y_test) * 100 ))


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision: {:.2f}%".format(metrics.precision_score(y_test, y_pred, average="micro") * 100 ))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall: {:.2f}%".format(metrics.recall_score(y_test, y_pred, average="micro") * 100 ))

rfc_predict = clf.predict(X_test)
report = classification_report(y_test, rfc_predict, zero_division='warn')
print("The report is: ")
print(report)

#  Building and presenting of the confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, rfc_predict)
print("The confusion matrix is: ")
print(conf_matrix)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()