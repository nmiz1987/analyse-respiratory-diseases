'''
The code contains noted parts that are intended for analysis by child.
In order to analyze by child, eliminate the comments and define the parts of the adults as comments.
'''

import csv
import pandas as pd
from sklearn.model_selection import train_test_split  # train_test_split function
from sklearn.ensemble import RandomForestClassifier  # Random Forest Model
from sklearn import metrics  # module for accuracy calculation
import matplotlib.pyplot as plt  # Visualizing the important features
import seaborn as sns  # Bar plot for the important features Visualization
from sklearn.metrics import classification_report


def predict_by_personal_data_adult(clf, age, sex, BMI):  # Prediction according to personal data
    return "The predict diagnosis is:" + clf.predict([[age, sex, BMI]])[0]


def predict_by_personal_data_child(clf, age, sex, weight, height):  # Prediction according to personal data
    return "The predict diagnosis is:" + clf.predict([[age, sex, weight, height]])[0]


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
 The Random Forest algorithm code in this project is based on the guidance of the tutorial in
 https://www.datacamp.com/community/tutorials/random-forests-classifier-python
"""

X = df[['Age', 'Sex', 'Adult BMI (kg/m2)']]  # Features for an adult
# X = df[['Age', 'Sex', 'Child Weight (kg)', 'Child Height (cm)']] # Features for a child
y = df['Diagnosis']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 20% test and 80% training

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=1_000, criterion='gini')  # 1,000 trees

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Finding Important Features in Scikit-learn
feature_imp = pd.Series(clf.feature_importances_, index=['Age', 'Sex', 'Adult BMI (kg/m2)']).sort_values(
    ascending=False)  # For an adult
# feature_imp = pd.Series(clf.feature_importances_,index=['Age', 'Sex', 'Child Weight (kg)', 'Child Height (cm)']).sort_values(ascending=False)  # For a child

print("Those are the important features that affect mostly on the result:")
print(feature_imp)
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

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
