import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
customer_churn = pd.read_csv("customer_churn.csv")
#finding the first few rows
pd.set_option('display.max_columns',None)
customer_churn.head()
#Extracting 5th column
customer_5=customer_churn.iloc[:,4] 
customer_5.head()
#Extracting 15th column
customer_15=customer_churn.iloc[:,14] 
customer_15.head()
#'Extracting male senior citizen with payment method-> electronic check'
senior_male_electronic=customer_churn[(customer_churn['gender']=='Male')
                                      & (customer_churn['SeniorCitizen']==1)
                                      & (customer_churn['PaymentMethod']=='Electronic check')]
senior_male_electronic.head()
#tenure>70 or monthly charges>100
customer_total_tenure=customer_churn[(customer_churn['tenure']>70) | (customer_churn['MonthlyCharges']>100)]
customer_total_tenure.head()
#cotract is 'two year', payment method is 'Mailed Check', Churn is 'Yes'
two_mail_yes=customer_total_tenure=customer_churn[(customer_churn['Contract']=='Two year')
                                                  & (customer_churn['PaymentMethod']=='Mailed check')
                                                  & (customer_churn['Churn']=='Yes')]
two_mail_yes
#Extracting 333 random records
customer_333=customer_churn.sample(n=333)
customer_333.head()

len(customer_333)
#count of levels of churn column
customer_churn['Churn'].value_counts()
%matplotlib inline
#bar-plot for 'InternetService' column
plt.bar(customer_churn['InternetService'].value_counts().keys().tolist(),
        customer_churn['InternetService'].value_counts().tolist(),color='orange')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of categories')
plt.title('Distribution of Internet Service')
#histogram for 'tenure' column
plt.hist(customer_churn['tenure'],color='green',bins=30)
plt.title('Distribution of tenure')
#scatterplot 
plt.scatter(x=customer_churn['tenure'],y=customer_churn['MonthlyCharges'],color='brown')
plt.xlabel('Tenure of Customer')
plt.ylabel('Monthly Charges of Customer')
plt.title('Tenure vs Monthly Charges')
#Box-plot
customer_churn.boxplot(column='tenure',by=['Contract'])
#-----------------------Linear Regresssion----------------------
from sklearn import linear_model
from sklearn.model_selection import train_test_split
x=pd.DataFrame(customer_churn['tenure'])
y=customer_churn['MonthlyCharges']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
#building the model
from sklearn.linear_model import LinearRegression
simpleLinearRegression = LinearRegression()
simpleLinearRegression.fit(x_train,y_train)
#predicting the values
y_pred = simpleLinearRegression.predict(x_test)
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_pred, y_test)
rmse = np.sqrt(mse)
rmse
x=pd.DataFrame(customer_churn['MonthlyCharges'])
#----------------------------------Logistic Regression-------------------------------
y=customer_churn['Churn']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.65,random_state=0)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)
LogisticRegression()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
y_pred = logmodel.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_pred,y_test),accuracy_score(y_pred,y_test)
#--------------Multiple logistic regression-------------------
x=pd.DataFrame(customer_churn.loc[:,['MonthlyCharges','tenure']])
y=customer_churn['Churn']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.80,random_state=0)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)
LogisticRegression()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
y_pred = logmodel.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
#---------------decision tree---------------
x=pd.DataFrame(customer_churn['tenure'])
y=customer_churn['Churn']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)  
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(x_train, y_train)  
DecisionTreeClassifier()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
y_pred = classifier.predict(x_test)  
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test, y_pred))   
print(accuracy_score(y_test, y_pred)) 
#--------------random forest---------------------
x=customer_churn[['tenure','MonthlyCharges']]
y=customer_churn['Churn']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)  
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)
RandomForestClassifier()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
y_pred=clf.predict(x_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
