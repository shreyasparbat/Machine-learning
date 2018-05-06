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

```python
#%%
## Building the Regression Model

# Fitting the Random Forest Regression to the Dataset
from sklearn.ensemble import RandomForestRegressor
```

Arguments for **RandomForestRegressor**():

1. **n_estimators**: the number of trees in the forest
2. **criterion**: function to measure the quality of a split. Default is *mse = mean square error*, which checks the difference of squares of the Y_pred and Y_actual. We will use this default option
3. **random_state**: the int specified here is used as a seed by the random number generator. This is to improve the randomness

```python
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, Y)

# Predicting a new result
Y_pred = regressor.predict(6.5)
```

## Visualising the Regression result

```python
#%%
## Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
```

![images/Image1.png](C:\Users\ABC\Desktop\Machine-learning\2. Regression\6. Random Forest Regression\images\Image 1.png)

Here, each step is the salary that the regressor has predicted for that position level. For any given position level, there were multiple trees voting on what the salary should be. The Random Forest regressor took the average of all these predictions and stated that as the final answer.

**Note**: Increasing the number of trees has diminishing returns as the *averages* found by each tree will start converging to a certain number, and the graph as a whole will start converging to a certain shape.