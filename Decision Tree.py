#!/usr/bin/env python
# coding: utf-8

# In[4]:


#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv(r'C:\Users\hung\Downloads\titanic\train.csv')
test = pd.read_csv(r'C:\Users\hung\Downloads\titanic\test.csv')



# Splitting the dataset into the Training set and Test set
X_train=train[['Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
X_train["Sex"].replace({"male": 1, "female": 0}, inplace=True)


from missingpy import MissForest
imputer = MissForest()
X_train.replace([np.inf, -np.inf], np.nan)
X_imputed = imputer.fit_transform(X_train)

Y_train=train['Survived']

# Training the Random Forest Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy',  max_depth=3, random_state = 0)
classifier.fit(X_imputed, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_imputed)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_train, Y_pred)
print(cm)
accuracy_score(Y_train, Y_pred)
from sklearn.metrics import precision_score
print(precision_score(Y_train, Y_pred))


X_test=test[['Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
X_test["Sex"].replace({"male": 1, "female": 0}, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan)
X_imputed_test = imputer.fit_transform(X_test)
y_pred = classifier.predict(X_imputed_test)



test = pd.read_csv(r'C:\Users\hung\Downloads\titanic\test.csv')
test['Survived']=y_pred
test=test[['PassengerId','Survived']]
test.to_csv(r'C:\Users\hung\Downloads\test.csv', index=False)


# In[48]:


from sklearn import tree
tree.plot_tree(classifier)

#fn=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
fn=['Sex', 'Age', 'SibSp', 'Parch', 'Fare']
cn=['0', '1']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(classifier,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')


# In[22]:



pd.crosstab(index=X_train['SibSp'], columns='Count')

