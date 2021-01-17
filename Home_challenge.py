#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv('home-challenge.xls')


# In[3]:


type(dataset)


# In[4]:


dataset.shape


# In[5]:


dataset.head()


# In[6]:


dataset.describe()


# In[7]:


dataset.info()


# In[8]:


dataset["zeta_disease"].describe()


# ##### Outliers detection

# From describe function we can find "liver_stress_test", "insulin_test", "cardio_stress_test" there are outlier, so it should be treated

# In[10]:


sns.boxplot(dataset["liver_stress_test"])


# In[11]:


sns.boxplot(dataset["cardio_stress_test"])


# After checking the boxplot, it is recommended to treat outliers and build the model for better output.

# ###### Outlier treating using z value

# In[12]:


from scipy import stats
z = np.abs(stats.zscore(dataset))
print(z)


# In[13]:


threshold = 3
print(np.where(z > 3))


# In[14]:


dataset_df = dataset[(z < 3).all(axis=1)]


# Orginal data

# In[15]:


dataset.shape


# Data without Outliers

# In[17]:


dataset_df.shape


# In[18]:


dataset_df.describe()


# In[19]:


X = dataset_df[["age","weight","bmi","blood_pressure","insulin_test","liver_stress_test","cardio_stress_test","years_smoking"]]
y = dataset_df["zeta_disease"]

sns.countplot(y)


# In[20]:


Output = dataset_df.zeta_disease.value_counts()
print(Output)

print("Percentage of patience without zeta_disease: "+str(round(Output[0]*100/699,2)))
print("Percentage of patience with zeta_disease: "+str(round(Output[1]*100/699,2)))


# #### Train Test Split

# In[21]:


from sklearn. model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.2, random_state = 1)


# In[22]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ###### As it is classification use case, it is recomended to build classification models and based on accuracy of different models, we can select one

# ### Logistic Model

# In[40]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()


# In[41]:


log_model.fit(X_train,y_train)


# In[42]:


predictions_log = log_model.predict(X_test)


# In[43]:


from sklearn.metrics import accuracy_score


# In[44]:


score_logmodel = round(accuracy_score(predictions_log,y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_logmodel)+" %")


# ### RandomForest

# In[49]:


from sklearn.ensemble import RandomForestClassifier      
rfc_model = RandomForestClassifier(n_estimators=100)
rfc_model.fit(X_train, y_train)


# In[50]:


predictions_rfc = rfc_model.predict(X_test)


# In[51]:


score_rfcmodel = round(accuracy_score(predictions_rfc,y_test)*100,2)

print("The accuracy score achieved using RandomForest model is: "+str(score_rfcmodel)+" %")


# ### XG Boost

# In[53]:


import xgboost as xgb
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=3)


# In[54]:


xgb_model.fit(X_train, y_train)


# In[55]:


predictions_xgb = xgb_model.predict(X_test)


# In[56]:


score_xgbmodel = round(accuracy_score(predictions_xgb,y_test)*100,2)

print("The accuracy score achieved using XGBoost model is: "+str(score_xgbmodel)+" %")


# #### Final_score

# In[57]:


scores = [score_logmodel,score_rfcmodel,score_xgbmodel]
algorithms = ["Logistic Regression","Random Forest","XGBoost"] 

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


# #### Finally, RandonForest model is recommended

# We can still implement, neural networks using Tensorflow and also use H2o autoML feature as well to build model for above dataset
