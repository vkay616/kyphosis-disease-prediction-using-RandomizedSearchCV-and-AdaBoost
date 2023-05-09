import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from glob import glob
import seaborn as sns
import random
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression # logistic regression
from sklearn import svm # support vector machine
from sklearn.ensemble import RandomForestClassifier #Random_forest
from sklearn.tree import DecisionTreeClassifier #Decision tree
from sklearn.naive_bayes import GaussianNB #Naive_bayes
from sklearn.neighbors import KNeighborsClassifier #K nearest neighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv('data.csv')

data

data.info()

data.isnull().sum()

data.columns

w = data['Kyphosis'].value_counts()
print(w.index)
print(w.values)
w = pd.DataFrame(w)
print(w)

sns.countplot(data['Kyphosis'])
plt.grid()
plt.legend()
plt.title(' absent vs present ')
plt.show()
print(' ')
plt.pie([64,17],labels=['absent','present'],autopct='%.2f%%')
plt.legend(loc=(1,0.5))
plt.title('absent vs present ')
plt.show()

sns.pairplot(data,hue='Kyphosis',vars= ['Age','Number','Start'])

from sklearn.utils import resample
# Separate majority and minority classes
df_majority = data[data['Kyphosis']== 'absent']
df_minority = data[data['Kyphosis']== 'present']
 
# Downsample majority class and upsample the minority class
df_minority_upsampled = resample(df_minority, replace=True,n_samples=64,random_state=123) 
#df_majority_downsampled = resample(df_majority, replace=True,n_samples=500,random_state=123) 
 
# Combine minority class with downsampled majority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])
 
# Display new class counts
df_upsampled['Kyphosis'].value_counts()

sns.countplot(df_upsampled['Kyphosis'])
plt.grid()
plt.legend()
plt.title(' absent vs present ')
plt.show()
print(' ')
plt.pie([500,500],labels=['absent','present'],autopct='%.2f%%')
plt.legend(loc=(1,0.5))
plt.title('absent vs present ')
plt.show()

df_upsampled

# shuffle the DataFrame rows 
data= df_upsampled.sample(frac = 1)

x = data.iloc[:,1:] 
y = data.iloc[:,0]

x

y

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
X_trans = pt.fit_transform(x)
print(X_trans.shape)

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=5,interaction_only=False,include_bias=True)
x_pol=pf.fit_transform(X_trans)

x_train,x_test,y_train,y_test = train_test_split(x_pol,y,test_size=0.20,stratify=y ,random_state=30)

x_train

x_test

y_train

y_test

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

y_train

y_test

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV

dept = [1, 5, 10, 50, 100, 500,800, 1000]
min_samples =  [5, 10, 100, 500]


param_grid={'min_samples_split':min_samples , 'max_depth':dept}
clf = DecisionTreeClassifier()
model = GridSearchCV(clf,param_grid,scoring='accuracy',n_jobs=-1,cv=5)
model.fit(x_train,y_train)
print("optimal min_samples_split",model.best_estimator_.min_samples_split)
print("optimal max_depth",model.best_estimator_.max_depth)
optimal_min_samples_split = model.best_estimator_.min_samples_split
optimal_max_depth = model.best_estimator_.max_depth

#Testing AUC on Test data
dt = DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth =optimal_max_depth,min_samples_split =optimal_min_samples_split)

dt.fit(x_train,y_train)


#predict on test data and train data
 
y_predtestd = dt.predict(x_test)
y_predtraind = dt.predict(x_train)

print('*'*35)

#accuracy on training and testing data

print('the accuracy on testing data',accuracy_score(y_test,y_predtestd))
print('the accuracy on training data',accuracy_score(y_train,y_predtraind))
train0 = accuracy_score(y_train,y_predtraind)
test0 = accuracy_score(y_test,y_predtestd)

print('*'*35)


# Code for drawing seaborn heatmaps
class_names = ['absent','present']
cm = pd.DataFrame(confusion_matrix(y_test, y_predtestd.round()), index=class_names, columns=class_names )
fig = plt.figure( )
heatmap = sns.heatmap(cm, annot=True, fmt="d")
dt_probs = dt.predict_proba(x_test)
dt_fpr, dt_tpr, _ = roc_curve(y_test,dt_probs[:,1])
plt.figure()
plt.plot(dt_fpr, dt_tpr, label='Decision Tree')
# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()

original =  ['absent' if x==1 else 'present' for x in y_test[:32]]
predicted = dt.predict(x_test[:32])
pred = []

for i in predicted:
  if i == 1:
    k = "absent"
    pred.append(k)
  else:
    k = "present"
    pred.append(k)
# Creating a data frame
dfr = pd.DataFrame(list(zip(original, pred,)), 
               columns =['original_Classlabel', 'predicted_classlebel'])
dfr

all_model_result = pd.DataFrame(columns=['Algorithm', 'Classifier' , 'Train-Accuracy', 'Test-Accuracy' ])
new = ['Decison tree ','DECISION-TREE-Classifier',train0, test0]
all_model_result.loc[0] = new

"""**Random Forest**"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV

dept = [1, 5, 10, 50, 100, 500, 1000]
n_estimators =  [20, 40, 60, 80, 100, 120]


param_grid={'n_estimators':n_estimators , 'max_depth':dept}
clf = RandomForestClassifier()
model = GridSearchCV(clf,param_grid,scoring='accuracy',n_jobs=-1,cv=5)
model.fit(x_train,y_train)
print("optimal n_estimators",model.best_estimator_.n_estimators)
print("optimal max_depth",model.best_estimator_.max_depth)
optimal_n_estimators = model.best_estimator_.n_estimators
optimal_max_depth = model.best_estimator_.max_depth

#Testing AUC on Test data
rf = RandomForestClassifier(max_depth = optimal_max_depth,n_estimators =optimal_n_estimators)

rf.fit(x_train,y_train)

#predict on test data and train data
 
y_predtest = rf.predict(x_test)
y_predtrain = rf.predict(x_train)

print('*'*35)

#accuracy on training and testing data

print('the accuracy on testing data',accuracy_score(y_test,y_predtest))
print('the accuracy on training data',accuracy_score(y_train,y_predtrain))
train1 = accuracy_score(y_train,y_predtrain)
test1 = accuracy_score(y_test,y_predtest)

print('*'*35)


# Code for drawing seaborn heatmaps
class_names = ['absent','present']
cm = pd.DataFrame(confusion_matrix(y_test, y_predtest.round()), index=class_names, columns=class_names )
fig = plt.figure( )
heatmap = sns.heatmap(cm, annot=True, fmt="d")
rf_probs = rf.predict_proba(x_test)
rf_fpr, rf_tpr, _ = roc_curve(y_test,rf_probs[:,1])
plt.figure()
plt.plot(rf_fpr, rf_tpr, label='Random Forest')
# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()

original =  ['absent' if x==1 else 'present' for x in y_test[:32]]
predicted = rf.predict(x_test[:32])
pred = []

for i in predicted:
  if i == 1:
    k = "absent"
    pred.append(k)
  else:
    k = "present"
    pred.append(k)
# Creating a data frame
dfr = pd.DataFrame(list(zip(original, pred,)), 
               columns =['original_Classlabel', 'predicted_classlebel'])
dfr

new = ['Random Forest','RandomForestClassifier',train1, test1]
all_model_result.loc[1] = new

all_model_result

# Logistic Regression
LR = LogisticRegression()
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
score_LR = LR.score(x_test,y_test)
print('The accuracy of the Logistic Regression model is', score_LR)
print(classification_report(y_test, y_pred))

# Support Vector Classifier (SVM/SVC)
from sklearn.svm import SVC
svc = SVC(gamma=0.22)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
score_svc = svc.score(x_test,y_test)
print('The accuracy of SVC model is', score_svc)
print(classification_report(y_test, y_pred))

from sklearn.ensemble import AdaBoostClassifier
ad = AdaBoostClassifier()

# K-Nearest Neighbors
ad.fit(x_train, y_train)
y_pred = ad.predict(x_test)
score_ad = ad.score(x_test,y_test)
print('The accuracy of the AdaBoostClassifier Model is',score_ad)
print(classification_report(y_test, y_pred))

from sklearn.ensemble import GradientBoostingClassifier
ad = GradientBoostingClassifier()

# K-Nearest Neighbors
ad.fit(x_train, y_train)
y_pred = ad.predict(x_test)
score_ad = ad.score(x_test,y_test)
print('The accuracy of the AdaBoostClassifier Model is',score_ad)
print(classification_report(y_test, y_pred))

# logistic regression
model1 = GradientBoostingClassifier()
# knn
model2 = AdaBoostClassifier()

# fit model
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)

# predict probabilities
pred_prob1 = model1.predict_proba(x_test)
pred_prob2 = model2.predict_proba(x_test)

from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])

print(auc_score1, auc_score2)

# matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='GradientBoostingClassifier')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='AdaBoost')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();