# Code Explanation

**Business Problem description**: to figure out whether a potential employee is lying or telling the truth about his previous salary based on what our model predicts his previous salary *should have been* based on what position he held. 

## Data Preprocessing

For more information, see *../Part 1: Data Preprocessing*.

```python
#%%
## Data Preprocessing

# Importing libraries
import numpy as np
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
```

**Note**: there is no need to carry out feature scaling for this dataset

## Building the Regression Model
