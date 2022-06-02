لimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 16)

train = pd.read_csv('Titanic dataset.csv')  # 12 columns
test = pd.read_csv('New passengers.csv')  # 11 columns

print("The training data : \n", train.describe(), "\n")  # Display Some details about Dataset (Max , count , mean ..etc)

print("\n\n", train.columns)  # Display list of the features names within the dataset

print(train.head(10))  # Display an Ordered Sample from the dataset

print(train.sample(10))  # Display random 10 samples

Positive_People = train[train['Survived'].isin([1])]
Negative_People = train[train['Survived'].isin([0])]

print(' Survived People Dataset \n', Positive_People)
print('-----------------------------------------------')
print(' UnSurvived People Dataset\n\n', Negative_People)

print("Data types for each feature : -")
print(train.dtypes)

# Observation : Cabin Feature has alot of missing values and age feature also but with less percentage

print("\n", pd.isnull(train).sum())  # Display the sum of null values in each column

# Now we will show how different survival chance between features and plot them
# Data Visualization, so we can estimate some predictions

sbn.barplot(x="Sex", y="Survived", data=train)  # Draw a bar plot of survival by Sex Feature
plt.show()

print("Percentages of Females Vs. Males who \n")
print(train["Survived"][train["Sex"] == 'female'])
print("---------------------------------\n\n")
print(train["Survived"][train["Sex"] == 'female'].value_counts())
print("====================================\n\n")
print("Percentage of Females Who Survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True)[1]*100)
print("Percentage of Males Who Survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True)[1]*100)
print("---------------------------------\n\n")

sbn.barplot(x="Pclass", y="Survived", data=train)  # Draw a bar plot of Survival by Pclass Feature
plt.show()

print("Percentage of Pclass = 1 who survived: \n", train["Survived"][train["Pclass"] == 1].value_counts(normalize=True)[1] * 100)
print("Percentage of Pclass = 2 who survived: \n", train["Survived"][train["Pclass"] == 2].value_counts(normalize=True)[1] * 100)
print("Percentage of Pclass = 3 who survived: \n", train["Survived"][train["Pclass"] == 3].value_counts(normalize=True)[1] * 100)

sbn.barplot(x="Parch", y="Survived", data=train)  # Draw a bar plot of Survival by Parch Feature
plt.show()

# Some Observations from above output
# People with less than four parents or children aboard are more likely to survive than those with four or more.
# people traveling alone are less likely to survive than those with 1-3 parents or children.

# First , We sort the ages into logical categories

train["Age"] = train["Age"].fillna(train["Age"].mean())  # Fill Age Null Values by the avg value
test["Age"] = test["Age"].fillna(-1)  # Give for Age Null Val any values that show that is unknown -1 or -2 the same
bins = [-2, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels=labels)    # Create Column AgeGroup with the new names
test['AgeGroup'] = pd.cut(test["Age"], bins, labels=labels)
print(train)

sbn.barplot(x="AgeGroup", y="Survived", data=train)  # Draw a bar plot of Age Feature
plt.show()

# Some Observations from above output
# Babies are more likely to survive than any other age group.

# People with cabin numbers more likely to survive as they high class people
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

print("----------------------------------\n\n")

print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(
          normalize=True)[1] * 100)

print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(
          normalize=True)[1] * 100)


sbn.barplot(x="CabinBool", y="Survived", data=train)  # Draw a bar plot of CabinBool Feature
plt.show()

# Cleaning Data from missing values and unnecessary information!

print(test.describe(include="all"))     # We have a total of 418 passengers.


# Drop the Cabin , Name and Ticket Features since not useful information can be extracted from it.
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)
train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)

print("SHAPE[0] = ", train[train["Embarked"] == "S"].shape[0])  # Display # of rows , SHAPE[1] :Display # of Col.

print("Number of people embarking in (s):")
s = train[train["Embarked"] == "S"].shape[0]
print(s)

print("Number of people embarking in C (C):")
c = train[train["Embarked"] == "C"].shape[0]
print(c)

print("Number of people embarking in Q (Q):")
q = train[train["Embarked"] == "Q"].shape[0]
print(q)

# Major of People are embarked in S , So we ' ll fill the missing values of it
train = train.fillna({"Embarked": "S"})  # replacing the missing values in the Embarked feature with S


age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3,   # map each Age value to a numerical value
               'Student': 4, 'Young Adult': 5,
               'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
print("\n", train)

sex_mapping = {"male": 0, "female": 1}   # map each Sex value to a numerical value
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)
print("\n", train)

embarked_mapping = {"S": 1, "C": 2, "Q": 3}  # map each Embarked value to a numerical value
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)
print(train)

train = train.drop(['Fare'], axis=1)  # Drop Fare values
test = test.drop(['Fare'], axis=1)    # Drop Fare values
print("\n\nFare column dropped\n")
print(train)

print(test.head(10), "\n\n")  # Check Test Data

# Choosing the Best Model .. Splitting the Training Data


from sklearn.model_selection import train_test_split
I_Pred = train.drop(['Survived', 'PassengerId'], axis=1)
O_Pred = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(I_Pred, O_Pred, test_size=0.20, random_state=1)

# _______________________________Four models to Determine Accuracy score________________________________________________
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import svm

# ____________________________________________KNN Model(1)_____________________________________________________________
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print("MODEL-1: Accuracy of k-Nearest Neighbors : ", acc_knn)

# Calculating error for K values between 1 and 40
error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):     # K Value = 40
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_val)
    error.append(np.mean(pred_i != y_val))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# ________________________________________DecisionTreeClassifier Model(2)___________________________________________
decisiontree = DecisionTreeClassifier()
# Gini : Min the prob of misclassification m its value between 0 : 1 --> = 1 - SUM(PROB^2)
clf_tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=1)
clf_tree.fit(x_train, y_train)
fig, ax = plt.subplots(figsize=(10, 10))
plot_tree(clf_tree, fontsize=10)
plt.show()
y_pred = clf_tree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("MODEL-2: Accuracy of DecisionTreeClassifier : ", acc_decisiontree)

# ____________________________________________Naïve Bayes oR GaussianNB Model(3)_________________________________________
gaussian = GaussianNB()
result = gaussian.fit(x_train, y_train).predict(x_val)
accuracy = accuracy_score(y_val, result)
print("MODEL-3: Accuracy of GaussianNB : ", accuracy * 100)

# _____________________________________________________Linear SVC Modele(4)________________________________________

classfier = svm.SVC(kernel='linear')  # Linear Kernel
classfier.fit(x_train, y_train)
predsvm = classfier.predict(x_val)
Saccuracy = round(accuracy_score(predsvm, y_val)*100,2)
print("MODEL-4: Accuracy Linear svc : ", Saccuracy)

# MODEL-5) Random Forest
from sklearn.ensemble import RandomForestClassifier
randforest = RandomForestClassifier(random_state=1)
randforest.fit(x_train, y_train)
y_pre = randforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pre, y_val) * 100, 2)
print("MODEL-5: Accuracy of RandomForestClassifier : ", acc_randomforest)
