import numpy
import pandas as pd
import matplotlib as plot
import numpy as np
df = pd.read_csv('cdk.csv')
x=df.iloc[:,0:24].values
y=df.iloc[:,24].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='Nan', strategy='median', axis =0,copy=False)
imputer = imputer.fit(x[:,0:5])
imputer.fit_transform(x[:,0:5])

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size0=0.2,random_state=0)
pd.DataFrame(x_train)
pd.DataFrame(y_train)
pd.DataFrame(x_test)
pd.DataFrame(x_test)