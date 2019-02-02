---
title: "Feature importance - Random Forest"
excerpt: "Three methods of estimating feature importance using Random Forests"
date: 2018-12-14
layout: single
classes: wide
tags: [data science, machine learning, python]
categories: [research]
header:
  overlay_image: /images/forest.jpg
  overlay_filter: 0.5
---
# Feature importance using Random Forest
In this blog post I will look at using a random forest for assessing feature importance, running through three different methods of doing so. 
## Mean decrease impurity
This was my go to method for estimating feature importance for the first 6 months of data science projects since it is so easy to implement. Random forests consist of multiple decision trees, each node in a tree is a condition on a single feature, designed to split the dataset into two so that similar response values end up in the same set. The measure on which the optimal condition is chosen is called 'impurity', for the regression model in this example it is variance, for classification it is either 'gini impurity' or 'entropy'. When training a tree the decrease in impurity by each feature can therefore be calculated, this figure can then be averaged over each tree in the forest and the features can be ranked accoriding to this measure.


```python
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

boston = load_boston()
X = boston.data
y = boston.target
feature_names = boston.feature_names

model = RandomForestRegressor(n_estimators=100)
model.fit(X,y)

importances = model.feature_importances_

print('Features sorted by score')
sorted(zip(importances, feature_names),reverse=True)
```

    Features sorted by score





    [(0.47411103815873246, 'RM'),
     (0.3342543076821159, 'LSTAT'),
     (0.06504354546666029, 'DIS'),
     (0.0348036179672585, 'CRIM'),
     (0.021714128314401858, 'NOX'),
     (0.01920821501534186, 'PTRATIO'),
     (0.01432558912156178, 'TAX'),
     (0.01371099835132889, 'AGE'),
     (0.010537793068627774, 'B'),
     (0.0061692697731715275, 'INDUS'),
     (0.004353926085011793, 'RAD'),
     (0.0010991414319102349, 'ZN'),
     (0.0006684295638773144, 'CHAS')]



While simple to implement, this can lead to misinterpretation of the data as demonstrated by the following example. The target is simply the sum of the three input features, therefore the relative importance of each feature should be roughly equal.


```python
import numpy as np

np.random.seed(seed=42)
size = 10000
X_seed = np.random.normal(1,1,size)
X0 = X_seed + np.random.normal(1,1,size)
X1 = X_seed + np.random.normal(1,1,size)
X2 = X_seed + np.random.normal(1,1,size)

X = np.array([X0,X1,X2]).T
y = X0 + X1 + X2
```


```python
model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X,y)
importances = model.feature_importances_

sorted(zip(importances, ['X0','X1','X2']),reverse=True)
```




    [(0.4834802759752004, 'X1'),
     (0.2619672580916795, 'X0'),
     (0.2545524659331203, 'X2')]



While all having true importances roughly equal to each other, X1 has been ranked almost twice as important as the other two features despite a decent sample size. This is a result of the models behaviour, with two or more correlated features the impurity removed by the model from the first feature cannot then be removed again from further feature(s). Thus giving the first feature a higher ranking when in reality any of the correlated features could have been used as the initial predictor.
While this would be helpful to remove features to reduce overfitting it can lead to confusion when interpreting data.

It should be noted that while this is often not a real life solution, increasing the size of the training set will reduce this bias in this situation.


```python
np.random.seed(seed=42)
size = 100000
X_seed = np.random.normal(1,1,size)
X0 = X_seed + np.random.normal(1,1,size)
X1 = X_seed + np.random.normal(1,1,size)
X2 = X_seed + np.random.normal(1,1,size)

X = np.array([X0,X1,X2]).T
y = X0 + X1 + X2

model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X,y)
importances = model.feature_importances_

sorted(zip(importances, ['X0','X1','X2']),reverse=True)
```




    [(0.3535161503005193, 'X1'),
     (0.3401914590843465, 'X0'),
     (0.3062923906151344, 'X2')]



## Mean decrease accuracy
Another method for assessing feature importance is mean decrease accuracy, the idea is to directly measure the impact of each feature on the accuracy of the model. To implement we measure a baseline score, shuffle the values of the feature in question and return another score. If the feature is pure noise then the shuffle will not affect the score, an important feature would return a reduced score. 

In this example I performed 3 iterations of the test returning an average score.


```python
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score

X = boston.data
y = boston.target
model = RandomForestRegressor(n_estimators=100, random_state=42)
totals = np.zeros(X.shape[1])

ss = ShuffleSplit(n_splits=3, test_size=0.2)

for train_idx, test_idx in ss.split(X):
    scores = []
    model.fit(X[train_idx],y[train_idx])
    predictions = model.predict(X[test_idx])
    r2 = r2_score(y[test_idx], predictions)
    for i in range(X.shape[1]):
        X_permute = X.copy()
        np.random.shuffle(X_permute[:,i])
        model.fit(X_permute[train_idx], y[train_idx])
        pred = model.predict(X_permute[test_idx])
        r2_test = r2_score(y[test_idx], pred)
        scores.append((r2-r2_test)/r2)
    totals += scores

mean_scores = totals/3
sorted(zip(mean_scores,feature_names),reverse=True)
```




    [(0.08076726253763289, 'RM'),
     (0.07027845322156301, 'LSTAT'),
     (0.023446138596186262, 'NOX'),
     (0.019253253610203375, 'DIS'),
     (0.018660143734972285, 'PTRATIO'),
     (0.006566053552545124, 'CRIM'),
     (0.005039995681832424, 'TAX'),
     (0.004787854141582798, 'AGE'),
     (0.003940480957968398, 'INDUS'),
     (0.001813284130196573, 'B'),
     (0.00042021776333498985, 'CHAS'),
     (-0.00032027375904764157, 'ZN'),
     (-0.000773357338529237, 'RAD')]



These results are consistent with the first test, with the greatest loss in performance being produced by the two best performing features in the first method.

## Single feature testing
An alternative approach is to test each feature on its own, the features with greater impact should yield an improved score. 


```python
from sklearn.model_selection import cross_val_score

X = boston.data
y = boston.target
model = RandomForestRegressor(n_estimators=100, random_state=42)

n_features = X.shape[1]

for i in range(n_features):
    score = cross_val_score(model,X[:,i].reshape(-1,1),y)
    print('%s    %d' %(feature_names[i], score.mean()))
```

    CRIM    0
    ZN    0
    INDUS    0
    CHAS    0
    NOX    -2
    RM    0
    AGE    0
    DIS    -1
    RAD    -1
    TAX    0
    PTRATIO    0
    B    0
    LSTAT    0


While clearly either a poor choice of metric or unsuitable application of this method, it could show meaningful insights in other situations.
## Conclusion
All methods can provide valuable insight into the data, however it is important to be aware of potential pitfalls in your methodology. The code for mean decreased accuracy could be applied to any model, but SKlearns built in ranking will remain my first choice due to how quick is it to implement. Backing up one measure with a different approach can provide another way of validating your results and some peace of mind when interpretabality is in question.
