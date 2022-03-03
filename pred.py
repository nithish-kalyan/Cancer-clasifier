import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df=pd.read_csv('cancer_classification.csv')

sns.countplot(x=df['benign_0__mal_1'])

y=df['benign_0__mal_1']
x=df.drop('benign_0__mal_1',axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)
x_train,x_test,y_train,y_test=np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

y_predict=model.predict(x_test)


from sklearn.metrics import accuracy_score,classification_report
print(classification_report(y_test,y_predict))


print( "ACCURACY : " ,accuracy_score(y_test,y_predict)*100)

