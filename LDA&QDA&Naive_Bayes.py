from sklearn.datasets import load_iris
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import math
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_decision_regions


iris = load_iris()
dt=pd.DataFrame(iris.data)
dc=pd.DataFrame(iris.target)
dm=pd.concat([dt,dc],axis=1)
dt.columns = ["Slength","Swidth","Plength","Pwidth"]
dm.columns = ["Slength","Swidth","Plength","Pwidth","dc"]
print(dt)

plt.subplot(1,4,1)
plt.hist(dt["Slength"])
plt.xlabel("Sepal Length")
plt.subplot(1,4,2)
plt.hist(dt["Swidth"])
plt.xlabel("Sepal Width")
plt.subplot(1,4,3)
plt.hist(dt["Plength"])
plt.xlabel("Petal Length")
plt.subplot(1,4,4)
plt.hist(dt["Pwidth"])
#,bins=100,range=[0,2]
plt.xlabel("Petal Width")
plt.show()
print(dt.info())
print(dt.describe())
sns.pairplot(dt)
plt.show()
sns.pairplot(dt[dt['Pwidth'] >= 1])
sns.pairplot(dm, hue ="dc")
plt.show()


filePath = r'C:/Users/WeiTh/Desktop/wine.csv'
wn = pd.read_csv(filePath)
wn.columns = ["class","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids"
    ,"Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
print(wn)
clas=wn["class"]
wnt=wn.drop('class',axis=1)
label2=["Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids"
    ,"Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
for i in range(0,12):
    plt.subplot(3, 5, i+1)
    plt.hist(wnt[label2[i]])
    plt.xlabel(label2[i])
plt.show()
print(wnt.info())
print(wnt.describe())
fileName='note.csv'
#df=wnt.describe()
#df.to_csv(fileName)


# define predictor and response variables
x = dt
y = dc
# Fit the LDA model
model = LinearDiscriminantAnalysis()
model.fit(x, y[0])
# Define method to evaluate model
y_pred = model.predict(x)
# model evaluation
y_pred=pd.DataFrame(y_pred)
ds=pd.concat([y,y_pred], axis=1)
ds=pd.DataFrame(ds)
ds.to_csv(fileName)
accuracy = accuracy_score(y[0], y_pred)
print(f'Accuracy: {accuracy:.3f}')

# evaluate model k fold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, x, y[0], scoring='accuracy', cv=cv, n_jobs=-1)
print(np.mean(scores))



x = iris.data
y = iris.target
y1=pd.DataFrame(y)
#y1.to_csv(fileName)
# define data to plot
data_plot = model.fit(x, y).transform(x)
# dimensionality reduction
#print(data_plot)
target_names = iris.target_names
# create LDA plot
plt.figure()
colors = ['red', 'green', 'blue']
lw = 2
plt.subplot(1,3,1)
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(data_plot[y == i, 0], data_plot[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel("Original Classification")
plt.subplot(1,3,2)
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(data_plot[y_pred[0] == i, 0], data_plot[y_pred[0] == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel("LDA Classification")
# display LDA plot

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(x, y)
y_qda=qda.predict(x)
#dta_plot = qda.fit(x, y).transform(x)
# create LDA plot
plt.subplot(1,3,3)
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(data_plot[y_qda == i, 0], data_plot[y_qda == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel("QDA Classification")
# display LDA plot
plt.show()
y_qda=pd.DataFrame(y_qda)
fa=pd.concat([dc,y_qda], axis=1)
fa.to_csv(fileName)
accuracy = accuracy_score(dc[0], y_qda[0])
print(f'Accuracy: {accuracy:.3f}')


from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
gnb = GaussianNB()
gnb.fit(wnt, clas)
data_plot = pca.fit_transform(wnt)
target_names=["1","2","3"]
# making predictions on the testing set
y_pred = gnb.predict(wnt)
# comparing actual response values (y_test) with predicted response values (y_pred)
plt.subplot(1,2,1)
for color, i, target_name in zip(colors, [1, 2, 3], target_names):
    plt.scatter(data_plot[clas == i, 0], data_plot[clas == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel("Original Classification")
plt.subplot(1,2,2)
for color, i, target_name in zip(colors, [1, 2, 3], target_names):
    plt.scatter(data_plot[y_pred == i, 0], data_plot[y_pred == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel("Naive Bayes Classification")
plt.show()
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(clas, y_pred)*100)





