import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics, svm

df = pd.read_csv("loan.csv")
df.head()
df.info()

df.isnull().sum()

df['LoanAmount_log'] = np.log1p(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

df.isnull().sum()

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log1p(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)

df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.LoanAmount_log = df.LoanAmount_log.fillna(df.LoanAmount_log.mean())

df['LoanAmount_Term'].fillna(df['LoanAmount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

x = df.iloc[:, np.r_[1:5, 9:11, 13:15]].values
y = df.iloc[:, 12].values

x
print(y)

print("per of missing gender is %2f%%" % (df ['Gender']. isnull().sum()/df.shape[0]*100))

print("number of people who take loan as group by gender : ")
print(df['Gender'].value_counts())
sns.countplot(x = 'Gender', data = df, palette = 'set1')


print("number of people who take loan as group by materal status : ")
print(df['Married'].value_counts())
sns.countplot(x = 'Married', data = df, palette = 'set1')

print("number of people who take loan as group by dependents : ")
print(df['Dependents'].value_counts())
sns.countplot(x = 'Dependents', data = df, palette = 'set1')

print("number of people who take loan as group by Self_Employed :  ")
print(df['Self_Employed'].value_counts())
sns.countplot(x = 'Self_Employed', data = df, palette = 'set1')

print("number of people who take loan as group by Loan Amount : ")
print(df['LoanAmount'].value_counts())
sns.countplot(x = 'LoanAmount', data = df, palette = 'set1')

print("number of people who take loan as group by Credit History : ")
print(df['Credit_History'].value_counts())
sns.countplot(x = 'Credit_History', data = df, palette = 'set1')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

oe = OrdinalEncoder()
x_train[:, :5] = oe.fit_transform(x_train[:, :5])
x_test[:, :5] = oe.transform(x_test[:, :5])


Labelencoder_y = LabelEncoder()
y_train = Labelencoder_y.fit_transform(y_train)
y_test = Labelencoder_y.transform(y_test)
y_test

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(x_train, y_train)

from sklearn import metrics
y_pred = rf_clf.predict(x_test)

print("Accuracy of Random Forest Classifier is : ", metrics.accuracy_score(y_test, y_pred))

y_pred

from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)

y_pred = nb_clf.predict(x_test)
print("Accuracy of Naive Bayes Classifier is : ", metrics.accuracy_score(y_test, y_pred))
y_pred

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train, y_train)

y_pred = dt_clf.predict(x_test)
print("Accuracy of Decision Tree Classifier is : ", metrics.accuracy_score(y_test, y_pred))
y_pred

from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier()
kn_clf.fit(x_train, y_train)

y_pred = kn_clf.predict(x_test)
print("Accuracy of KNeighbors Classifier is : ", metrics.accuracy_score(y_test, y_pred))
y_pred
