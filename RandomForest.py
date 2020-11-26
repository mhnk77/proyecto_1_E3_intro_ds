#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
#import numpy as np
import pickle
#import seaborn as sns
#import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


# In[19]:


from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
#from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit


# In[35]:


from pytictoc import TicToc
t = TicToc()


# In[20]:

infile = open('data_encoded_pickle.pickle','rb')
data_encoded = pickle.load(infile)
infile.close()

# In[21]:

data_encoded

# In[36]:

#Una vez ordenado por dttm, ya podemos eliminar dicha variable
x_data = data_encoded.drop(columns= ['label', 'dttm_creacion'])

x_data

y_data = data_encoded['label']

y_data



# In[38]:
# ### Dividir data en train test

#Importante el Shuffle = False ya que estamos tratando con fechas ordenadas
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42, shuffle = False)

x_train

x_test

y_train

y_test

# In[39]:

y_train.value_counts(normalize = True)

# In[40]:
y_test.value_counts(normalize = True)

#%%
#Definir time series split

tscv = TimeSeriesSplit(n_splits = 3)

#Verificar
for train_index, test_index in tscv.split(x_train['sin_hora']):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x_train['sin_hora'][train_index], x_train['sin_hora'][test_index]
    Y_train, Y_test = y_train[train_index], y_train[test_index] 

# ### Definir RF

# In[50]:

rfc=RandomForestClassifier(random_state=42)

# In[51]:


param_grid = { 
    'n_estimators': [200],
    'max_features': [5,6,7,8,9],
    'max_depth' : [20,25],
    'criterion': ['gini']
}


# In[52]:


param_grid


# In[53]:



CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= tscv, scoring = 'precision', n_jobs= 5)



# In[ ]:


t.tic()
CV_rfc.fit(x_train, y_train)
t.toc()

# In[21]:


CV_rfc.best_params_

CV_rfc.best_score_

CV_rfc.best_estimator_

#%%

pred=CV_rfc.best_estimator_.predict(x_test)


#%%

pd.DataFrame(pred).value_counts()

#rfc1 = RandomForestClassifier(random_state =42, max_features = 5, n_estimators = 200, criterion = 'gini', max_depth=20)

t.tic()
#rfc1.fit(x_train, y_train)
t.toc()

#pred=rfc1.predict(x_test)

pd.DataFrame(pred).value_counts()


# In[45]:
    
pred_train = CV_rfc.predict(x_train)
#pred_train = rfc1.predict(x_train)

print("Precision for Random Forest on train data: ",precision_score(y_train,pred_train))
    
# In[46]:


print("Accuracy for Random Forest on test data: ",accuracy_score(y_test,pred))


# In[47]:


print("Precision for Random Forest on test data: ",precision_score(y_test,pred))


# In[49]:


d_imp = {'feature': x_train.columns, 'importance': CV_rfc.best_estimator_.feature_importances_}
#d_imp = {'feature': x_train.columns, 'importance': rfc1.feature_importances_}

pd.DataFrame(d_imp).sort_values(by = 'importance', ascending = False)


outfile = open('var_importance.pickle','wb')
pickle.dump(d_imp,outfile)

outfile.close()



outfile = open('cv_model.pickle','wb')
pickle.dump(CV_rfc,outfile)

outfile.close()




