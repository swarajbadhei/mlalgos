import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
from sklearn import linear_model
from sklearn.metrics import r2_score
df=pd.read_csv("FuelConsumptionCo2.csv")
df.head()
df.describe()
cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB_MPG','CO2EMISSIONS']]
cdf.head(9)
viz=cdf[['ENGINESIZE','CYLINDERS']]
viz.hist()
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB_MPG,cdf.CO2EMISSIONS,color='red')
plt.xlabel('consumption')
plt.ylabel('emission')
plt.show()

msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.xlabel('engine_size')
plt.ylabel('co2-emissions')
plt.show()
regr=linear_model.LinearRegression()
train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
print("learning complete")
print("coefficient:",regr.coef_)
print("intercept:",regr.intercept_)
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='green')
plt.plot(train_x,regr.coef_[0][0]*train_x+regr.intercept_[0],'-b')
plt.xlabel('engine_size')
plt.ylabel('co2 emissions')
plt.show()
test_x=np.asanyarray(test['ENGINESIZE'])
test_y=np.asanyarray(test['CO2EMISSIONS'])
test_y_=regr.predict(test_x)
print('MAR:%.2f'%np.mean(np.absolute(test_y-test_y_)))
print('MSE:%.2f'%np.mean((test_y-test_y_)**2))
print('R2-SCORE:%.2f'%r2_score(test_y_,test_y))
