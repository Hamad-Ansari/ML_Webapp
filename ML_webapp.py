# Libararies 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# app ki Heading
st.title(''' #Explore ML Models and Datasets
        ## A Streamlit Web App for Machine Learning Model Comparison
        ## and Dataset Exploration
        ## Upload your dataset and explore different ML models
         **With Hammad Ansari**
         ''')
# data set k name ak box may dal ker sidebar dikhana
dataset_name=st.sidebar.selectbox("Select Dataset", 
                                  ("Iris", "Breast Cancer", "Wine Quality"))
# or isika nicha aik box may classification ya regression ka option dikhana
classifier_name=st.sidebar.selectbox(
    "Select Classifier",
    ("KNN", "SVM", "Random Forest")
)

# create funstion for dataset
def get_dataset(dataset_name):
    if dataset_name=="Iris":
        iris=datasets.load_iris()
        X=pd.DataFrame(data=iris.data, columns=iris.feature_names)
        y=pd.DataFrame(data=iris.target, columns=["target"])
    elif dataset_name=="Breast Cancer":
        breast_cancer=datasets.load_breast_cancer()
        X=pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
        y=pd.DataFrame(data=breast_cancer.target, columns=["target"])
    else:
        wine=datasets.load_wine()
        X=pd.DataFrame(data=wine.data, columns=wine.feature_names)
        y=pd.DataFrame(data=wine.target, columns=["target"])
    return X,y
# call function
X,y=get_dataset(dataset_name)

# data set ka size dikhana
st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))

# definde the difreent classifier
def add_parameter_ui(classifier_name):
    params={}
    if classifier_name=="KNN":
        K=st.sidebar.slider("K", 1, 15)
        params["K"]=K # K nearest neighbor 
    elif classifier_name=="SVM":
        C=st.sidebar.slider("C", 0.01, 10.0)
        params["C"]=C # c is the degree of the polynomial kernel
    else:
        max_depth=st.sidebar.slider("max_depth", 2, 15)
        n_estimators=st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"]=max_depth
        params["n_estimators"]=n_estimators
    return params

# call the function
params=add_parameter_ui(classifier_name)


# function for classifier create for classifier_name and params
def get_classifier(classifier_name, params):
    if classifier_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name=="SVM":
        clf=SVC(C=params["C"])
    else:
        clf=RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
    return clf

#call the function
clf=get_classifier(classifier_name, params)

#ab data set ko train or test may divide karna
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=1234)

# fit the model
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
# accuracy score
accuracy=accuracy_score(y_test, y_pred)
st.write("Classifier:", classifier_name)
st.write("Accuracy:", accuracy)
st.write("Parameters:", params)
# confusion matrix
st.write("Confusion Matrix:")
cm=confusion_matrix(y_test, y_pred)
st.write(cm)
# classification report
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))
# plot confusion matrix
fig, ax=plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", ax=ax)
st.pyplot(fig)
# plot the data set
fig, ax=plt.subplots()
ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test.values, cmap="viridis", s=50)

ax.set_xlabel(X.columns[0])
ax.set_ylabel(X.columns[1])
st.pyplot(fig)
# plot the data set with PCA
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X)
fig, ax=plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y.iloc[:, 0], cmap="viridis", s=50)

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
st.pyplot(fig)
# plot the data set with PCA and classifier
fig, ax=plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y.iloc[:,0], cmap="viridis", s=50)
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clf.predict(X), cmap="coolwarm", s=50, alpha=0.5)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
st.pyplot(fig)
# plot the data set with PCA and classifier and decision boundary
fig, ax=plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y.iloc[:,0], cmap="viridis", s=50)
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clf.predict(X), cmap="coolwarm", s=50, alpha=0.5)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")


