import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Customer churn prediction dataset overview:
# This dataset contains detailed information about customers, including demographic attributes, account details, service subscriptions, and billing data. Each row represents an individual customer, capturing their relationship and usage history with the company. The dataset is designed to analyze customer behavior and identify patterns related to customer retention and churn.
df=pd.read_csv("customer_churn_data.csv")
df
df.head()
df.info()
df.describe()
df["InternetService"].unique()
df.isnull().sum()
df["InternetService"]=df["InternetService"].fillna("")
df.isna().sum()
df.duplicated().sum()
df.head()
df.groupby("Churn")["Tenure"].mean()
df.groupby("Churn")["Tenure"].mean().plot(kind="bar")
plt.show()
df.head()
df["Churn"].value_counts().plot(kind="pie")
plt.show()

df.groupby("Churn")["MonthlyCharges"].mean()
df.groupby("Churn")["Age"].mean()
df.groupby(["Churn","Gender"])["MonthlyCharges"].mean()
num_cat=df.select_dtypes(include="number")
num_cat.corr()
sns.heatmap(num_cat.corr(),annot=True)
plt.show()
df.head()
df.groupby("ContractType")["MonthlyCharges"].mean().plot(kind="bar")
plt.title("contract type average price")
plt.xlabel("contract type")
plt.ylabel("Mean Price")
plt.show()

plt.hist(df["MonthlyCharges"])
plt.title("histogram of monthly charges")
plt.show()
plt.hist(df["Tenure"])
plt.title("histogram of tenure")
plt.show()
df.columns

X=df[["Age","Gender","Tenure","MonthlyCharges"]]
y=df[["Churn"]]
X
y
X["Gender"]=X["Gender"].map({"Female":1,"Male":0})
X["Gender"]
type(X["Gender"][0])
y
y.head()
y["Churn"]=y["Churn"].map({"Yes":1,"No":0})
y["Churn"].unique()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train
X_test
from sklearn.metrics import accuracy_score
def acc_sc(prediction):
    acc=accuracy_score(y_test,prediction)
    print("Accuracy score is ",acc)
# Logistic Regression is a supervised machine learning algorithm used for binary classification problems.  
# It predicts the probability of an outcome using the sigmoid (logistic) function and assigns a class label based on a threshold.

from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model.fit(X_train,y_train)

y_pred=log_model.predict(X_test)
acc_sc(y_pred)

# K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression.  
# It predicts outcomes based on the majority class or average value of the nearest data points in the feature space.

from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier()
param_grid={
    "n_neighbors":[3,5,7,9],
    "weights":["uniform", "distance"]
}
# GridSearchCV is a hyperparameter tuning technique that exhaustively searches through a specified parameter grid.  
# It uses cross-validation to find the best parameter combination that gives the highest model performance.

from sklearn.model_selection import GridSearchCV

grid_knn=GridSearchCV(model_knn,param_grid,cv=5)
grid_knn.fit(X_train,y_train)
grid_knn.best_params_
y_predkn=grid_knn.predict(X_test)
acc_sc(y_predkn)

# Decision Tree Classifier is a supervised machine learning algorithm used for classification tasks.  
# It splits data into branches based on feature conditions to make decisions and predict class labels.

from sklearn.tree import DecisionTreeClassifier
model_dt=DecisionTreeClassifier()
param_grid={
    "criterion":["gini","entropy"],
    "splitter":["best","random"],
    "max_depth":[None,10,20,30],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,4]
    
}
grid_dt=GridSearchCV(model_dt,param_grid,cv=5)
grid_dt.fit(X_train,y_train)
grid_dt.best_params_
y_preddt=grid_dt.predict(X_test)
acc_sc(y_preddt)


# Support Vector Classifier (SVC) is a supervised machine learning algorithm used for classification.  
# It finds the optimal hyperplane that best separates different classes in the feature space.

from sklearn.svm import SVC
model_svm=SVC()
param_grid={
    "C":[0.01,0.1,0.5,1],
    "kernel": ["linear","rbf","poly"]
}

grid_svm=GridSearchCV(model_svm,param_grid,cv=5)
grid_svm.fit(X_train,y_train)
grid_svm.best_params_
y_predsvm=grid_svm.predict(X_test)
y_predsvm
acc_sc(y_predsvm)

# Random Forest Classifier is an ensemble machine learning algorithm used for classification.  
# It builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting.

from sklearn.ensemble import RandomForestClassifier
model_rfc=RandomForestClassifier()
para_grid={
    "n_estimators":[32,64,128,256],
    "max_features":[2,3,4],
    "bootstrap":[True,False]
    
}

grid_rfc=GridSearchCV(model_rfc,para_grid,cv=5)
grid_rfc.fit(X_train,y_train)
grid_rfc.best_params_
y_pred_rfc=grid_rfc.predict(X_test)
acc_sc(y_pred_rfc)
best_model=log_model
best_model

# cross_val_score is a model evaluation method used to assess performance using cross-validation.  
# It splits data into multiple folds and returns scores for each fold to measure model stability.

from sklearn.model_selection import cross_val_score
score=cross_val_score(best_model,X,y,cv=5)
score

score.mean()
X.columns
import joblib
joblib.dump(scaler,"scaler.pkl")
joblib.dump(best_model,"model.pkl")
joblib.dump(X.columns,"columns.pkl")
models={
    "LogisticRegression":LogisticRegression(),
    "KNeighborsClassifier":KNeighborsClassifier(),
    "DecisionTreeClassifier":DecisionTreeClassifier(),
    "SVC":SVC(),
    "RandomForestClassifier":RandomForestClassifier()
}
for name,model in models.items():
    model.fit(X_train,y_train)
    y_p=model.predict(X_test)
    acc=accuracy_score(y_test,y_p)
    print(name,"Accuracy is :",acc)

