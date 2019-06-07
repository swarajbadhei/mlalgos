import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing
import matplotlib.ticker as ticker
#https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv
df=pd.read_csv("teleCust1000t.csv")
print(df.head())
print(df['custcat'].value_counts())
print(df.hist(column='income',bins=50))
print(df.columns)
X=df[['region', 'tenure','age','marital','address','income','ed','employ','retire','gender','reside']].values
print(X[0:5])
X=preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])
y=df['custcat'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)
print("train set:",X_train.shape,y_train.shape)
print("test set:",X_test.shape,y_test.shape)
from sklearn.neighbors import KNeighborsClassifier as knc
k=4
neigh=knc(n_neighbors=k).fit(X_train,y_train)
yhat=neigh.predict(X_test)
print(yhat[0:5])
from sklearn import metrics
print("training accuracy=",metrics.accuracy_score(y_train,neigh.predict(X_train)))
print("test accuracy=",metrics.accuracy_score(y_test,yhat))
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    #Train Model and Predict  
    neigh = knc(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
print(mean_acc)
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
