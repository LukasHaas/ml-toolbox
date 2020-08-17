# The Machine Learning Toolbox

This is a repository containing python utilities for machine learning and deep learning. 

## 1. Rule-Based Machine Learning: *RuleAugmentedEstimator Class*

**Combine domain knowledge facts with machine learning.**

This class is a wrapper class for sklearn estimators with the additional possibility of adding rule-based logic to the underlying estimator.
The provided rules are hard-coded and take precedence over the underlying estimator's predictions.
    
**Example Usage:**

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()

# Assume GradientBoostingRegressor was trained on data

rules = {"House Type": [
            ("=", "Penthouse", 1.0),
            ("=", "Shack", 0.0)
         ],
         "House Price": [
            ("<", 1000.0, 0.0),
            (">=", 500000.0, 0.0)
         ]}
         
ra_estimator = RuleAugmentedEstimator(gbr, rules)

# Now, the RuleAugmentedEstimator can be used similarily to the underlying base estimator, assuming data X is defined

predictions = ra_estimator.predict(X)
```
In the above example, whenever in the `House Type` pandas column the value "Penthouse" appears, the value `1.0` is returned by the estimator.
Similarily, if any value in the `House Price` pandas column is greater or equal to `500000.0`, the value `0.0` will be returned.

## 2. Regression at Confidence Intervals (Quantile Regression): *QuantileRegressor Class*

**Evaluate regression predictions at varying levels of confidence.**

This class is a wrapper class for sklearn regressor with the additional possibility of adding multiple quantiles.
These can be specificed to evaluate the prediction problem's result at varying confidence levels, all at the same time.
    
**Example Usage:**

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
q_gbr = QuantileRegressor(gbr, fit_quantiles=[0.4, 0.5, 0.55]) # The class also allows setting an interval and step size

# Now the quantile regressor can be trained at multiple confidence levels at the same time, assuming data X is defined
q_gbr.fit(X)

# The QuantileRegressor class can be used to predict regression outcomes at varying levels of confidence
predictions = q_gbr.predict(X)

predictions
>>> {'0.4' : array([0.09000266, 0.1899997 , 0.2099997 ]),
     '0.5' : array([0.10000266, 0.1999997 , 0.2199997 ]),
     '0.55': array([0.11000266, 0.2099997 , 0.2299997 ])}
     
# The QuantileRegressor class can also be used to predict confidence intervals (first fit it on the required quantiles).
confidence_intervals = q_gbr.predict_confidence_interval(data, confidence=0.2)  # Predicts the 20% confidence interval

confidence_intervals
>>> [(0.10000265613988876, 0.2),
     (0.19999970487334573, 0.2),
     (0.21999970487334575, 0.29999734386011123)]
```

