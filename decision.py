import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as dtc
data=pd.read_csv('drug200.csv',delimiter=',')
X=data[['Age','Sex','BP','Cholesterol','Na_to_K']].values
print(X[0:5])
from sklearn import preprocessing as pproc
le_sex=pproc.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1]=le_sex.transform(X[:,1])
le_BP=pproc.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])
X[:,2]=le_BP.transform(X[:,2])
le_chol=pproc.LabelEncoder()
le_chol.fit(['LOW','NORMAL','HIGH'])
X[:,3]=le_chol.transform(X[:,3])
print(X[0:5])
y=data['Drug']
print(y[0:5])
from sklearn.model_selection import train_test_split as tts
X_trn,X_test,y_trn,y_test=tts(X,y,test_size=0.3,random_state=3)
print(X_trn.shape)
print(y_trn.shape)
#modelling from here now
drugtree=dtc(criterion='entropy',max_depth=4)
drugtree.fit(X_trn,y_trn)
predtree=drugtree.predict(X_test)
print(predtree[0:5])
print(y_test[0:5])
#finding the accuracy of model
from sklearn import metrics
import matplotlib.pyplot as plt
print("decision tree accuracy:",metrics.accuracy_score(y_test,predtree))
#visualization
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
dot_data = StringIO()
filename = "drugtree.png"
featureNames =data.columns[0:5]
targetNames =data["Drug"].unique().tolist()
out=tree.export_graphviz(drugtree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trn), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
